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
4. **Temporal persistence bonus** — bins that are part of a sustained
   run of high-scoring bins receive an additive bonus.
5. **Final ranking** — events are re-scored from their enriched bin
   values and sorted descending.

All thresholds are read from the ``ranker`` section of ``config.yaml``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from streams.base_stream import SAREvent, EventSeverity


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
        Additive bonus from temporal persistence.
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

        persist_cfg = config.get("temporal_persistence", {})
        self.persistence_enabled = persist_cfg.get("enabled", True)
        self.persistence_window = persist_cfg.get("window_seconds", 2.0)
        self.persistence_bonus = persist_cfg.get("bonus", 0.15)

        # Derived
        self.n_bins = max(1, math.ceil(self.video_duration / self.bin_resolution))

    # ── Public API ───────────────────────────────────────────────────────

    def rank(self, events: List[SAREvent]) -> List[RankedEvent]:
        """Execute the full ranking pipeline.

        Parameters
        ----------
        events : list[SAREvent]
            Raw events from all streams (unordered).

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

        # 5. Temporal persistence bonus
        if self.persistence_enabled:
            self._apply_persistence_bonus(ranked, bin_scores, event_bins)

        # 6. Compute final score
        for r in ranked:
            r.final_score = (r.fused_score * r.cross_stream_boost) + r.persistence_bonus

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

    def _apply_persistence_bonus(
        self,
        ranked: List[RankedEvent],
        bin_scores: List[float],
        event_bins: List[List[int]],
    ) -> None:
        """Add a bonus for events in sustained high-scoring regions.

        A bin is considered "active" if its score > 0.  The persistence
        bonus is awarded to events whose bins fall within a contiguous
        run of active bins spanning ≥ ``persistence_window`` seconds.
        """
        window_bins = max(1, int(self.persistence_window / self.bin_resolution))

        # Pre-compute: for each bin, length of the contiguous active run it belongs to
        run_lengths = [0] * self.n_bins
        current_run = 0
        run_start = 0
        for b in range(self.n_bins):
            if bin_scores[b] > 0:
                if current_run == 0:
                    run_start = b
                current_run += 1
            else:
                # Close run — assign length
                if current_run > 0:
                    for rb in range(run_start, run_start + current_run):
                        run_lengths[rb] = current_run
                current_run = 0
        # Close final run
        if current_run > 0:
            for rb in range(run_start, run_start + current_run):
                run_lengths[rb] = current_run

        for i, r in enumerate(ranked):
            bins = event_bins[i]
            max_run = max((run_lengths[b] for b in bins), default=0)
            if max_run >= window_bins:
                r.persistence_bonus = self.persistence_bonus
            else:
                r.persistence_bonus = 0.0

    @staticmethod
    def _severity_from_score(score: float) -> EventSeverity:
        """Map final score to severity tier.

        Thresholds are calibrated for ``conf × z`` products:
        * ``score ≥ 3.0``  →  CRITICAL
        * ``score ≥ 1.5``  →  HIGH
        * ``score ≥ 0.5``  →  MEDIUM
        * ``score <  0.5`` →  LOW
        """
        if score >= 3.0:
            return EventSeverity.CRITICAL
        if score >= 1.5:
            return EventSeverity.HIGH
        if score >= 0.5:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW
