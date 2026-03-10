"""
streams/base_stream.py
======================
Abstract base class for all SARTriage detection streams, plus the
shared data-classes that every stream must produce.

Design notes
------------
* Each concrete stream (Action, Motion, Tracking) sub-classes `BaseStream`
  and implements **`detect()`**.
* All streams emit a list of `SAREvent` objects, giving the orchestrator a
  uniform interface regardless of signal source.
* `SAREvent` carries enough metadata for the **Priority Ranker** to perform
  cross-stream fusion, temporal persistence scoring, and budget-constrained
  timeline building.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ── Severity Enum ────────────────────────────────────────────────────────────

class EventSeverity(Enum):
    """Coarse severity tiers used for UI colour-coding and fast filtering."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Core Event Data-Class ────────────────────────────────────────────────────

@dataclass
class SAREvent:
    """A single triage-relevant event detected by any stream.

    Attributes
    ----------
    event_id : str
        Globally unique identifier (UUID4) assigned at creation time.
    stream_name : str
        Identifier of the originating stream  (e.g. ``"action"``,
        ``"motion"``, ``"tracking"``).
    start_frame : int
        First frame index (0-based) where the event was observed.
    end_frame : int
        Last frame index (inclusive) of the event span.
    start_time : float
        Event start in *seconds* relative to the video start.
    end_time : float
        Event end in seconds.
    confidence : float
        Stream-native confidence / score in **[0, 1]**.
    z_score : float
        How anomalous this event is compared to the running baseline
        (higher ⟹ more unusual).  Computed by each stream internally.
    label : str
        Human-readable short label  (e.g. ``"person_falling"``,
        ``"sudden_motion_spike"``, ``"track_loss"``).
    severity : EventSeverity
        Coarse severity tier derived from z-score / confidence.
    bbox : Optional[List[float]]
        Representative bounding box ``[x1, y1, x2, y2]`` in pixel
        coordinates, if the event is spatially grounded.
    track_id : Optional[int]
        ByteTrack track identifier, if the event is tied to a specific
        tracked object.
    metadata : Dict[str, Any]
        Arbitrary extra data (action probabilities, flow magnitude
        histogram, etc.) that the ranker or UI may consume.
    """

    stream_name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    z_score: float
    label: str
    severity: EventSeverity = EventSeverity.LOW
    bbox: Optional[List[float]] = None
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ── Convenience helpers ──────────────────────────────────────────────

    @property
    def duration_frames(self) -> int:
        """Number of frames spanned by this event (inclusive)."""
        return self.end_frame - self.start_frame + 1

    @property
    def duration_seconds(self) -> float:
        """Duration of the event in seconds."""
        return self.end_time - self.start_time

    @property
    def mid_time(self) -> float:
        """Temporal midpoint in seconds – useful for NMS / merging."""
        return (self.start_time + self.end_time) / 2.0

    def overlaps(self, other: "SAREvent", iou_threshold: float = 0.0) -> bool:
        """Check whether two events overlap temporally.

        Parameters
        ----------
        other : SAREvent
            The other event to compare against.
        iou_threshold : float
            Minimum temporal IoU to consider as overlapping.
            ``0.0`` means any shared frame counts.

        Returns
        -------
        bool
        """
        inter_start = max(self.start_time, other.start_time)
        inter_end = min(self.end_time, other.end_time)
        intersection = max(0.0, inter_end - inter_start)
        if intersection == 0.0:
            return False
        union = (self.duration_seconds + other.duration_seconds) - intersection
        if union <= 0.0:
            return False
        return (intersection / union) >= iou_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary (for Flask responses)."""
        return {
            "event_id": self.event_id,
            "stream": self.stream_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration_seconds, 3),
            "confidence": round(self.confidence, 4),
            "z_score": round(self.z_score, 4),
            "label": self.label,
            "severity": self.severity.value,
            "bbox": self.bbox,
            "track_id": self.track_id,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"SAREvent(id={self.event_id!r}, stream={self.stream_name!r}, "
            f"label={self.label!r}, t=[{self.start_time:.2f}–{self.end_time:.2f}]s, "
            f"conf={self.confidence:.3f}, z={self.z_score:.2f}, "
            f"severity={self.severity.value})"
        )


# ── Frame Metadata Passed to Streams ─────────────────────────────────────────

@dataclass
class FramePacket:
    """Container for a single decoded frame plus pre-computed annotations.

    The orchestrator builds one ``FramePacket`` per frame and passes a
    sequence of them to each stream's ``detect()`` method.

    Attributes
    ----------
    index : int
        0-based frame index within the source video.
    timestamp : float
        Timestamp in seconds from video start.
    image : np.ndarray
        The decoded BGR frame (H × W × 3, ``uint8``).
    detections : Optional[List[Dict[str, Any]]]
        YOLO detections for this frame, each a dict with keys
        ``{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}``.
        ``None`` if YOLO has not yet run on this frame.
    tracks : Optional[List[Dict[str, Any]]]
        ByteTrack-assigned tracks, each extending the detection dict with
        ``{"track_id": int, ...}``.  ``None`` if tracking has not yet run.
    """

    index: int
    timestamp: float
    image: np.ndarray
    detections: Optional[List[Dict[str, Any]]] = None
    tracks: Optional[List[Dict[str, Any]]] = None


# ── Abstract Base Stream ─────────────────────────────────────────────────────

class BaseStream(ABC):
    """Abstract base class that every SARTriage detection stream must extend.

    Subclasses **must** implement:

    * ``detect(packets)``  – analyse a window of frames and return events.

    Subclasses **may** override:

    * ``setup()``          – one-time initialisation (load models, etc.).
    * ``teardown()``       – cleanup (release GPU memory, temp files, etc.).
    * ``name``  property   – human-readable stream identifier.

    Parameters
    ----------
    config : dict
        The *stream-specific* section from ``config.yaml``
        (e.g. ``config["action"]``).
    global_config : dict
        The full parsed ``config.yaml`` — streams can peek at shared
        settings (e.g. ``video.target_fps``) if needed.
    """

    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]) -> None:
        self.config = config
        self.global_config = global_config
        self._is_setup = False

    # ── Identity ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Return a short, unique name for this stream.

        Override in subclasses; default uses the class name.
        """
        return self.__class__.__name__.lower()

    @property
    def enabled(self) -> bool:
        """Whether this stream is switched on in the config."""
        return self.config.get("enabled", True)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def setup(self) -> None:
        """One-time initialisation hook (load model weights, warm-up, etc.).

        Called by the orchestrator *before* the first ``detect()`` call.
        Override freely; the base implementation is a no-op.
        """
        self._is_setup = True

    def teardown(self) -> None:
        """Cleanup hook (release GPU memory, close file handles, etc.).

        Called by the orchestrator when processing is complete.
        Override freely; the base implementation is a no-op.
        """
        self._is_setup = False

    # ── Core contract ────────────────────────────────────────────────────

    @abstractmethod
    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Analyse a window of frames and return detected SAR events.

        Parameters
        ----------
        packets : Sequence[FramePacket]
            Ordered sequence of frame packets.  Each packet contains the
            decoded image, its timestamp, and (optionally) pre-computed
            YOLO detections / ByteTrack tracks.

        Returns
        -------
        List[SAREvent]
            Zero or more events detected within this window.  Events
            **must** have ``stream_name`` set to ``self.name``.
        """
        ...

    # ── Shared Utilities ─────────────────────────────────────────────────

    @staticmethod
    def compute_z_score(value: float, values: Sequence[float]) -> float:
        """Compute how many standard deviations *value* is from the mean.

        Parameters
        ----------
        value : float
            The observation to score.
        values : Sequence[float]
            Historical / population values.

        Returns
        -------
        float
            The z-score.  Returns ``0.0`` when std-dev is near zero.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 2:
            return 0.0
        mean = arr.mean()
        std = arr.std(ddof=1)
        if std < 1e-9:
            return 0.0
        return float((value - mean) / std)

    @staticmethod
    def severity_from_z(z: float) -> EventSeverity:
        """Map a z-score to a coarse severity tier.

        Mapping
        -------
        * ``z < 1.5``  →  LOW
        * ``1.5 ≤ z < 2.5``  →  MEDIUM
        * ``2.5 ≤ z < 3.5``  →  HIGH
        * ``z ≥ 3.5``  →  CRITICAL
        """
        if z >= 3.5:
            return EventSeverity.CRITICAL
        if z >= 2.5:
            return EventSeverity.HIGH
        if z >= 1.5:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    def __repr__(self) -> str:
        status = "ready" if self._is_setup else "not initialised"
        return f"<{self.__class__.__name__} name={self.name!r} ({status})>"
