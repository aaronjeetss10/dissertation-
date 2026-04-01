"""
core/priority_ranker.py
=======================
Multi-stream fusion and priority scoring for SARTriage.

Ranking pipeline
----------------
1. **Temporal binning** — events are mapped onto a uniform time grid.
2. **Per-bin max aggregation** — for each time bin the highest
   normalised score across all streams is kept.
3. **Cross-stream validation boost** — bins where ≥ *k* distinct
   streams report events receive a multiplicative boost (default 1.2×).
4. **Temporal Criticality Evolution (TCE)** — a state machine tracks
   per-track motion states (MOVING → SLOWING → STILL → SUSTAINED_STILL
   → COLLAPSED) and applies a *logarithmic escalation* bonus when a
   track remains in SUSTAINED_STILL or COLLAPSED.
5. **Final ranking** — events are re-scored from their enriched bin
   values and sorted descending.

TCE replaces the earlier simple persistence bonus with a physics-aware
Hidden-Markov-inspired model.  The key insight: a person lying still
for 30 s is far more critical than one still for 2 s, but the urgency
increase is sub-linear (the first few seconds carry the most signal).

All thresholds are read from the ``ranker`` section of ``config.yaml``.
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from streams.base_stream import SAREvent, EventSeverity


# ════════════════════════════════════════════════════════════════════════
# TCE State Machine — Temporal Criticality Evolution
# ════════════════════════════════════════════════════════════════════════

class TCEState(enum.Enum):
    """Motion states for the Temporal Criticality Evolution model.

    8-state model per dissertation spec::

        MOVING_FAST ──(deceleration)──► DECELERATING
             ▲                              │
             │                   (velocity near-zero)
        MOVING_SLOW                         ▼
             ▲                            STOPPED
             │                              │
          (motion                  (still > threshold_1)
           resumes)                         ▼
             │                       SUSTAINED_STILL
             │                              │
             └──────────────── (still > threshold_2)
                                            ▼
                                     CRITICAL_STATIC

        COLLAPSED  ← (sudden velocity→0 + aspect change)
        ERRATIC    ← (high speed_cv + dir_change_rate)
    """
    MOVING_FAST     = "moving_fast"
    MOVING_SLOW     = "moving_slow"
    DECELERATING    = "decelerating"
    STOPPED         = "stopped"
    SUSTAINED_STILL = "sustained_still"
    CRITICAL_STATIC = "critical_static"
    COLLAPSED       = "collapsed"
    ERRATIC         = "erratic"


# Per-state base criticality scores (from spec)
_TCE_BASE_SCORES = {
    TCEState.MOVING_FAST:     0.3,
    TCEState.MOVING_SLOW:     0.2,
    TCEState.DECELERATING:    0.4,
    TCEState.STOPPED:         0.3,
    TCEState.SUSTAINED_STILL: 0.6,
    TCEState.CRITICAL_STATIC: 0.9,
    TCEState.COLLAPSED:       0.95,
    TCEState.ERRATIC:         0.5,
}


# Transition matrix: (current_state, condition) → next_state
# Conditions are evaluated in order; first match wins.
_TCE_TRANSITIONS = {
    TCEState.MOVING_FAST: [
        ("vel_medium", TCEState.MOVING_SLOW),
        ("deceleration", TCEState.DECELERATING),
        ("erratic", TCEState.ERRATIC),
        ("collapse", TCEState.COLLAPSED),
    ],
    TCEState.MOVING_SLOW: [
        ("vel_high", TCEState.MOVING_FAST),
        ("deceleration", TCEState.DECELERATING),
        ("vel_zero", TCEState.STOPPED),
        ("erratic", TCEState.ERRATIC),
        ("collapse", TCEState.COLLAPSED),
    ],
    TCEState.DECELERATING: [
        ("vel_high", TCEState.MOVING_FAST),
        ("vel_medium", TCEState.MOVING_SLOW),
        ("vel_zero", TCEState.STOPPED),
    ],
    TCEState.STOPPED: [
        ("vel_high", TCEState.MOVING_FAST),
        ("vel_medium", TCEState.MOVING_SLOW),
        ("sustained", TCEState.SUSTAINED_STILL),
        ("collapse", TCEState.COLLAPSED),
    ],
    TCEState.SUSTAINED_STILL: [
        ("vel_high", TCEState.MOVING_FAST),
        ("vel_medium", TCEState.MOVING_SLOW),
        ("critical", TCEState.CRITICAL_STATIC),
        ("collapse", TCEState.COLLAPSED),
    ],
    TCEState.CRITICAL_STATIC: [
        # Only strong motion can exit CRITICAL_STATIC
        ("vel_high", TCEState.MOVING_FAST),
        ("vel_medium", TCEState.MOVING_SLOW),
    ],
    TCEState.COLLAPSED: [
        # Only strong motion can exit COLLAPSED
        ("vel_high", TCEState.MOVING_FAST),
    ],
    TCEState.ERRATIC: [
        ("vel_high", TCEState.MOVING_FAST),
        ("vel_medium", TCEState.MOVING_SLOW),
        ("vel_zero", TCEState.STOPPED),
        ("collapse", TCEState.COLLAPSED),
    ],
}


@dataclass
class TCETrackState:
    """Per-track state for the TCE model.

    Maintained across temporal bins for each ``track_id``.
    """
    state: TCEState = TCEState.MOVING_FAST
    time_in_state: float = 0.0     # seconds spent in current state
    total_still_time: float = 0.0  # cumulative seconds in STOPPED+
    entry_bin: int = 0             # bin when this state started
    peak_bonus: float = 0.0        # highest TCE bonus ever awarded
    transitions: int = 0           # total state transitions

    def transition_to(self, new_state: TCEState, current_bin: int) -> None:
        """Execute a state transition."""
        if new_state != self.state:
            self.state = new_state
            self.time_in_state = 0.0
            self.entry_bin = current_bin
            self.transitions += 1
            # Reset total_still_time if returning to movement
            if new_state in (TCEState.MOVING_FAST, TCEState.MOVING_SLOW):
                self.total_still_time = 0.0


def tce_log_escalation(
    time_still: float,
    alpha: float = 0.3,
    tau: float = 30.0,
    cap: float = 2.5,
) -> float:
    r"""Logarithmic escalation function for temporal criticality.

    .. math::

        \text{escalation}(t) = 1 + \alpha \cdot \ln\!\left(1 + \frac{t}{\tau}\right)

    This is a **multiplicative** factor applied to the base_score.

    With spec defaults (α=0.3, τ=30):
        At t=0:    escalation = 1.00
        At t=30s:  escalation = 1.21
        At t=60s:  escalation = 1.33
        At t=300s: escalation = 1.72

    Parameters
    ----------
    time_still : float
        Seconds the track has been in STOPPED / SUSTAINED_STILL /
        CRITICAL_STATIC / COLLAPSED.
    alpha : float
        Scaling coefficient (default 0.3).
    tau : float
        Time constant in seconds (default 30.0).
    cap : float
        Maximum escalation factor (default 2.5).

    Returns
    -------
    float
        The escalation multiplier ∈ [1.0, cap].
    """
    if time_still <= 0:
        return 1.0
    escalation = 1.0 + alpha * math.log(1.0 + time_still / tau)
    return min(escalation, cap)


# ── Scored Event Wrapper ────────────────────────────────────────────────────

@dataclass
class RankedEvent:
    """An event enriched with its fused priority score.

    Attributes
    ----------
    event : SAREvent
        The original event from a detection stream.
    raw_score : float
        Stream-native score (normalised confidence × z-score weight).
    fused_score : float
        Score after max-aggregation across co-temporal events.
    cross_stream_boost : float
        Multiplicative boost applied (1.0 if no boost).
    persistence_bonus : float
        Additive TCE bonus (replaces the old flat persistence bonus).
    tce_state : str
        The TCE state for this event's track at evaluation time.
    emi_multiplier : float
        Ego-Motion Intelligence multiplier (1.0 = neutral).
    final_score : float
        The ultimate ranking score used for ordering.
    contributing_streams : set[str]
        Names of streams that contributed events in overlapping time.
    """

    event: SAREvent
    raw_score: float = 0.0
    fused_score: float = 0.0
    cross_stream_boost: float = 1.0
    persistence_bonus: float = 0.0
    tce_state: str = "moving_fast"
    emi_multiplier: float = 1.0
    final_score: float = 0.0
    contributing_streams: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON / Jinja template consumption."""
        d = self.event.to_dict()
        d.update({
            "raw_score": round(self.raw_score, 4),
            "fused_score": round(self.fused_score, 4),
            "cross_stream_boost": round(self.cross_stream_boost, 4),
            "persistence_bonus": round(self.persistence_bonus, 4),
            "tce_state": self.tce_state,
            "emi_multiplier": round(self.emi_multiplier, 4),
            "final_score": round(self.final_score, 4),
            "contributing_streams": sorted(self.contributing_streams),
        })
        return d


# ── Priority Ranker ─────────────────────────────────────────────────────────

class PriorityRanker:
    """Fuse and rank events from multiple detection streams.

    Parameters
    ----------
    config : dict
        The ``ranker`` section from ``config.yaml``.
    video_duration : float
        Total duration of the source video in seconds.
    bin_resolution : float
        Width of each temporal bin in seconds (default 0.5 s).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        video_duration: float,
        bin_resolution: float = 0.5,
    ) -> None:
        self.config = config
        self.video_duration = max(video_duration, 0.1)
        self.bin_resolution = bin_resolution

        # Config knobs
        self.aggregation = config.get("aggregation", "max")
        self.boost_factor = config.get("cross_stream_boost", 1.2)
        self.boost_min_streams = config.get("cross_stream_min_streams", 2)

        # ── TCE (Temporal Criticality Evolution) config ──
        # Backward-compatible: reads from `temporal_persistence` + `tce`
        persist_cfg = config.get("temporal_persistence", {})
        tce_cfg = config.get("tce", {})
        self.persistence_enabled = persist_cfg.get("enabled", True)

        # TCE velocity thresholds (applied to normalised bin scores)
        self.tce_vel_high = tce_cfg.get("vel_high_threshold", 0.3)
        self.tce_vel_medium = tce_cfg.get("vel_medium_threshold", 0.12)
        self.tce_vel_zero = tce_cfg.get("vel_zero_threshold", 0.02)

        # TCE timing thresholds
        self.tce_sustained_s = tce_cfg.get("sustained_threshold_s", 10.0)
        self.tce_critical_s = tce_cfg.get("critical_threshold_s", 60.0)
        self.tce_collapse_s = tce_cfg.get("collapse_threshold_s", 15.0)

        # TCE erratic thresholds
        self.tce_speed_cv_erratic = tce_cfg.get("speed_cv_erratic", 2.0)
        self.tce_dir_change_erratic = tce_cfg.get("dir_change_erratic", 0.4)
        self.tce_decel_threshold = tce_cfg.get("deceleration_threshold", -2.0)

        # Logarithmic escalation parameters (spec: α=0.3, τ=30)
        self.tce_alpha = tce_cfg.get("alpha", 0.3)
        self.tce_tau = tce_cfg.get("tau", 30.0)
        self.tce_cap = tce_cfg.get("cap", 2.5)

        # Per-track TCE state (keyed by track_id)
        self._tce_states: Dict[Optional[int], TCETrackState] = defaultdict(
            TCETrackState
        )

        # Derived
        self.n_bins = max(1, math.ceil(self.video_duration / self.bin_resolution))

    # ── Public API ───────────────────────────────────────────────────────

    def rank(
        self,
        events: List[SAREvent],
        emi_bin_multipliers: Optional[List[float]] = None,
    ) -> List[RankedEvent]:
        """Execute the full ranking pipeline.

        Parameters
        ----------
        events : list[SAREvent]
            Raw events from all streams (unordered).
        emi_bin_multipliers : list[float], optional
            Per-bin EMI multipliers from the Ego-Motion Intelligence
            module.  Length must equal ``n_bins``.  If ``None``, EMI
            is disabled (all multipliers default to 1.0).

        Returns
        -------
        list[RankedEvent]
            Events sorted by ``final_score`` descending, enriched with
            fusion metadata.
        """
        if not events:
            return []

        # 1. Compute raw scores
        ranked = [self._score_raw(e) for e in events]

        # 2. Temporal binning — map events to bins
        bin_scores, bin_streams, event_bins = self._temporal_bin(ranked)

        # 3. Max aggregation — assign fused score from bin
        self._apply_aggregation(ranked, bin_scores, event_bins)

        # 4. Cross-stream validation boost
        self._apply_cross_stream_boost(ranked, bin_streams, event_bins)

        # 5. Temporal Criticality Evolution (replaces old persistence bonus)
        if self.persistence_enabled:
            self._apply_tce(
                ranked, bin_scores, bin_streams, event_bins,
            )

        # 5.5. Ego-Motion Intelligence multiplier
        if emi_bin_multipliers is not None:
            self._apply_emi(ranked, emi_bin_multipliers, event_bins)

        # 6. Compute final score
        #    final = (fused × cross_stream_boost × emi_multiplier) + tce_bonus
        for r in ranked:
            r.final_score = (
                r.fused_score * r.cross_stream_boost * r.emi_multiplier
            ) + r.persistence_bonus

        # 7. Re-assign severity based on final score
        for r in ranked:
            r.event.severity = self._severity_from_score(r.final_score)

        # 8. Sort descending
        ranked.sort(key=lambda r: r.final_score, reverse=True)

        return ranked

    # ── Internal Stages ──────────────────────────────────────────────────

    def _score_raw(self, event: SAREvent) -> RankedEvent:
        """Compute raw score as normalised ``confidence × z_score``."""
        # z-scores can be 0 for events below threshold; use floor of 0.1
        z = max(event.z_score, 0.1)
        raw = event.confidence * z
        return RankedEvent(event=event, raw_score=raw)

    def _time_to_bin(self, t: float) -> int:
        """Map a timestamp to a bin index."""
        return min(int(t / self.bin_resolution), self.n_bins - 1)

    def _event_bins(self, event: SAREvent) -> List[int]:
        """Return all bin indices that an event spans."""
        start_bin = self._time_to_bin(event.start_time)
        end_bin = self._time_to_bin(event.end_time)
        return list(range(start_bin, end_bin + 1))

    def _temporal_bin(
        self, ranked: List[RankedEvent]
    ) -> Tuple[List[float], List[Set[str]], List[List[int]]]:
        """Map events onto a time grid and compute per-bin aggregates.

        Returns
        -------
        bin_scores : list[float]
            Aggregated score per bin.
        bin_streams : list[set[str]]
            Set of contributing stream names per bin.
        event_bins : list[list[int]]
            Bin indices for each event (parallel to ``ranked``).
        """
        bin_scores: List[float] = [0.0] * self.n_bins
        bin_streams: List[Set[str]] = [set() for _ in range(self.n_bins)]
        event_bins: List[List[int]] = []

        for r in ranked:
            bins = self._event_bins(r.event)
            event_bins.append(bins)

            for b in bins:
                # Max aggregation (default)
                if self.aggregation == "max":
                    bin_scores[b] = max(bin_scores[b], r.raw_score)
                elif self.aggregation == "sum":
                    bin_scores[b] += r.raw_score
                elif self.aggregation == "mean":
                    # Approximate: will normalise later
                    bin_scores[b] += r.raw_score
                else:
                    bin_scores[b] = max(bin_scores[b], r.raw_score)

                bin_streams[b].add(r.event.stream_name)

        # Normalise if using mean aggregation
        if self.aggregation == "mean":
            bin_counts = [0] * self.n_bins
            for bins in event_bins:
                for b in bins:
                    bin_counts[b] += 1
            for b in range(self.n_bins):
                if bin_counts[b] > 0:
                    bin_scores[b] /= bin_counts[b]

        return bin_scores, bin_streams, event_bins

    def _apply_aggregation(
        self,
        ranked: List[RankedEvent],
        bin_scores: List[float],
        event_bins: List[List[int]],
    ) -> None:
        """Assign fused score = max bin score across the event's span."""
        for i, r in enumerate(ranked):
            bins = event_bins[i]
            if bins:
                r.fused_score = max(bin_scores[b] for b in bins)
            else:
                r.fused_score = r.raw_score

    def _apply_cross_stream_boost(
        self,
        ranked: List[RankedEvent],
        bin_streams: List[Set[str]],
        event_bins: List[List[int]],
    ) -> None:
        """Apply multiplicative boost when ≥ k streams agree in a bin.

        The boost is applied once per event, using the *maximum* stream
        overlap found across all bins the event occupies.
        """
        for i, r in enumerate(ranked):
            bins = event_bins[i]
            max_streams: Set[str] = set()
            for b in bins:
                if len(bin_streams[b]) > len(max_streams):
                    max_streams = bin_streams[b]

            r.contributing_streams = max_streams

            if len(max_streams) >= self.boost_min_streams:
                r.cross_stream_boost = self.boost_factor
            else:
                r.cross_stream_boost = 1.0

    # ── EMI: Ego-Motion Intelligence Multiplier ─────────────────────────

    def _apply_emi(
        self,
        ranked: List[RankedEvent],
        emi_bin_multipliers: List[float],
        event_bins: List[List[int]],
    ) -> None:
        """Apply per-bin EMI multipliers to each event.

        For each event, the EMI multiplier is the **maximum** EMI
        multiplier across all bins the event spans.  This ensures
        that even a brief hover during a longer event is captured.
        """
        for i, r in enumerate(ranked):
            bins = event_bins[i]
            if bins:
                # Max EMI multiplier across the event's temporal span
                r.emi_multiplier = max(
                    emi_bin_multipliers[b]
                    for b in bins
                    if b < len(emi_bin_multipliers)
                )
            else:
                r.emi_multiplier = 1.0

    # ── TCE: Temporal Criticality Evolution ──────────────────────────────

    def _compute_bin_velocity(
        self,
        bin_idx: int,
        bin_scores: List[float],
    ) -> float:
        """Estimate the 'motion velocity' of a bin from its score.

        In a real system this would use tracker centroid deltas;
        here we use the bin score as a proxy: high-scoring bins
        from motion/action streams imply motion, low scores imply
        stillness.

        We also look at the score gradient (change between adjacent
        bins) as a secondary signal.
        """
        score = bin_scores[bin_idx]

        # Score gradient (how fast things are changing)
        if bin_idx > 0:
            gradient = abs(score - bin_scores[bin_idx - 1])
        else:
            gradient = 0.0

        # Combined velocity proxy: score + gradient
        return score + gradient * 0.5

    def _evaluate_tce_conditions(
        self,
        track_state: TCETrackState,
        velocity: float,
        has_collapse_label: bool,
    ) -> Optional[str]:
        """Evaluate which TCE transition condition (if any) is met.

        Uses the 8-state model with velocity thresholds, timing,
        and label-based collapse detection.

        Returns the condition string or None if no transition fires.
        """
        state = track_state.state

        # ── COLLAPSED detection (from any state via label) ──
        if has_collapse_label and state not in (TCEState.COLLAPSED,):
            return "collapse"

        if state == TCEState.MOVING_FAST:
            if velocity < self.tce_vel_medium:
                return "vel_medium"

        elif state == TCEState.MOVING_SLOW:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity < self.tce_vel_zero:
                return "vel_zero"

        elif state == TCEState.DECELERATING:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity >= self.tce_vel_medium:
                return "vel_medium"
            if velocity < self.tce_vel_zero:
                return "vel_zero"

        elif state == TCEState.STOPPED:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity >= self.tce_vel_medium:
                return "vel_medium"
            if track_state.time_in_state >= self.tce_sustained_s:
                return "sustained"

        elif state == TCEState.SUSTAINED_STILL:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity >= self.tce_vel_medium:
                return "vel_medium"
            if track_state.total_still_time >= self.tce_critical_s:
                return "critical"

        elif state == TCEState.CRITICAL_STATIC:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity >= self.tce_vel_medium:
                return "vel_medium"

        elif state == TCEState.COLLAPSED:
            if velocity >= self.tce_vel_high:
                return "vel_high"

        elif state == TCEState.ERRATIC:
            if velocity >= self.tce_vel_high:
                return "vel_high"
            if velocity >= self.tce_vel_medium:
                return "vel_medium"
            if velocity < self.tce_vel_zero:
                return "vel_zero"

        return None

    def _tce_bonus_for_state(self, track_state: TCETrackState) -> float:
        """Compute the TCE criticality score.

        Formula (from spec):
            criticality = base_score(state) × escalation(dwell_time)

        where:
            escalation(t) = 1 + α · log(1 + t/τ)

        For moving states, the escalation is always 1.0 (no dwell-time
        benefit). For still/static states, escalation grows logarithmically.
        """
        base = _TCE_BASE_SCORES.get(track_state.state, 0.3)

        # Escalation only applies to states with meaningful dwell time
        if track_state.state in (
            TCEState.MOVING_FAST, TCEState.MOVING_SLOW,
            TCEState.DECELERATING, TCEState.ERRATIC,
        ):
            # Fixed base score, no dwell-time escalation
            escalation = 1.0
        else:
            # STOPPED, SUSTAINED_STILL, CRITICAL_STATIC, COLLAPSED:
            # logarithmic escalation based on total still time
            t_still = track_state.total_still_time
            escalation = tce_log_escalation(
                t_still, alpha=self.tce_alpha,
                tau=self.tce_tau, cap=self.tce_cap,
            )

        criticality = base * escalation
        track_state.peak_bonus = max(track_state.peak_bonus, criticality)
        return criticality

    def _apply_tce(
        self,
        ranked: List[RankedEvent],
        bin_scores: List[float],
        bin_streams: List[Set[str]],
        event_bins: List[List[int]],
    ) -> None:
        """Apply Temporal Criticality Evolution to all ranked events.

        For each event, we:
        1. Identify its track_id (or assign a synthetic one).
        2. Build a **per-track** velocity signal from the event's own
           raw score and its label (not the shared global bin scores).
        3. Walk through its bins chronologically, updating the TCE
           state machine per-bin.
        4. Compute the TCE bonus from the final state.

        The state machine persists across events sharing the same
        ``track_id``, so a person who is first detected as 'falling'
        and later as 'collapsed' accumulates still-time continuously.
        """
        # Labels that indicate collapsed/lying posture
        COLLAPSE_LABELS = {
            "collapsed", "lying_down", "person_collapsed",
            "person_lying", "person_prone", "prone",
        }
        # Labels that indicate low/zero motion
        STILL_LABELS = COLLAPSE_LABELS | {
            "lying", "stationary", "still", "motionless",
        }
        # Labels that indicate high motion
        MOVING_LABELS = {
            "running", "walking", "crawling", "waving", "stumbling",
            "waving_hand", "climbing", "pushing", "pulling",
        }

        # ── Build per-track bin score maps ──
        # Each track gets its own score per bin, derived from its events.
        track_bin_scores: Dict[Optional[int], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for i, r in enumerate(ranked):
            track_id = r.event.track_id
            for b in event_bins[i]:
                # Per-track: use the event's own raw_score
                track_bin_scores[track_id][b] = max(
                    track_bin_scores[track_id][b], r.raw_score
                )

        for i, r in enumerate(ranked):
            bins = event_bins[i]
            if not bins:
                r.persistence_bonus = 0.0
                r.tce_state = TCEState.MOVING_FAST.value
                continue

            track_id = r.event.track_id
            ts = self._tce_states[track_id]

            # Check if this event's label implies collapse or stillness
            label_lower = r.event.label.lower().replace(" ", "_")
            has_collapse = label_lower in COLLAPSE_LABELS
            is_still_label = label_lower in STILL_LABELS
            is_moving_label = label_lower in MOVING_LABELS

            # Walk bins in chronological order
            for b in sorted(bins):
                # Per-track velocity: use this track's own bin score
                per_track_score = track_bin_scores[track_id].get(b, 0.0)

                # Compute velocity with label-awareness:
                # - If the label says "lying_down", override to near-zero
                # - If the label says "running", override to high
                if is_still_label:
                    vel = self.tce_vel_zero * 0.5  # force below vel_zero
                elif is_moving_label:
                    vel = max(per_track_score, self.tce_vel_high)
                else:
                    # Default: use per-track bin score as velocity proxy
                    vel = per_track_score
                    # Add gradient component
                    prev_score = track_bin_scores[track_id].get(b - 1, 0.0)
                    gradient = abs(per_track_score - prev_score)
                    vel = per_track_score + gradient * 0.5

                # Evaluate transition conditions
                condition = self._evaluate_tce_conditions(ts, vel, has_collapse)

                # Execute transition if a condition fired
                if condition is not None:
                    transitions = _TCE_TRANSITIONS.get(ts.state, [])
                    for cond, next_state in transitions:
                        if cond == condition:
                            ts.transition_to(next_state, b)
                            break

                # Accumulate time in current state
                ts.time_in_state += self.bin_resolution

                # Accumulate still time for STOPPED+ states
                if ts.state in (
                    TCEState.STOPPED,
                    TCEState.SUSTAINED_STILL,
                    TCEState.CRITICAL_STATIC,
                    TCEState.COLLAPSED,
                ):
                    ts.total_still_time += self.bin_resolution

            # Compute TCE bonus from final state
            r.persistence_bonus = self._tce_bonus_for_state(ts)
            r.tce_state = ts.state.value

    @staticmethod
    def _severity_from_score(score: float) -> EventSeverity:
        """Map final score to severity tier.

        Thresholds are calibrated for ``conf × z`` products:
        * ``score ≥ 3.0``  →  CRITICAL
        * ``score ≥ 1.5``  →  HIGH
        * ``score ≥ 0.5``  →  MEDIUM
        * ``score < 0.5``  →  LOW
        """
        if score >= 3.0:
            return EventSeverity.CRITICAL
        if score >= 1.5:
            return EventSeverity.HIGH
        if score >= 0.5:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW
