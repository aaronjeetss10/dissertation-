"""
tests/test_integration.py
============================
Integration tests for the SARTriage pipeline.

Tests cover:
  - Full pipeline produces events from synthetic input
  - BaseStream ABC enforcement
  - Configuration schema validation

Run:
    pytest tests/test_integration.py -v
"""

import sys
from pathlib import Path
from typing import Sequence, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from streams.base_stream import BaseStream, FramePacket, SAREvent, EventSeverity


# ══════════════════════════════════════════════════════════════════════════════
# 1.  test_pipeline_produces_events
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_produces_events():
    """Process synthetic 10-frame video with 1 person detection per frame
    → pipeline should produce at least 1 event."""
    from main import run_pipeline

    # Create a minimal temporary video-like file
    # Pipeline accepts path — for stub mode it generates synthetic frames
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Write minimal data so the file exists (pipeline has synthetic fallback)
        f.write(b"\x00" * 100)
        tmp_path = f.name

    try:
        result = run_pipeline(tmp_path)

        assert isinstance(result, dict), \
            f"Pipeline should return a dict, got {type(result)}"
        assert "events" in result, \
            "Pipeline result should contain 'events' key"
        assert "summary" in result, \
            "Pipeline result should contain 'summary' key"
        assert "processing_time" in result, \
            "Pipeline result should contain 'processing_time' key"

        events = result["events"]
        assert isinstance(events, list), \
            f"Events should be a list, got {type(events)}"
        # With synthetic data and multiple streams, we expect at least 1 event
        assert len(events) >= 1, \
            f"Expected at least 1 event from pipeline, got {len(events)}"

        # Verify event structure
        if events:
            e = events[0]
            assert isinstance(e, dict), "Events should be serialised as dicts"
            required_keys = {"stream", "start_frame", "confidence", "label"}
            present = set(e.keys())
            missing = required_keys - present
            assert not missing, \
                f"Event missing required keys: {missing}. Has: {present}"

    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  test_basestream_interface_enforced
# ══════════════════════════════════════════════════════════════════════════════

def test_basestream_interface_enforced():
    """A subclass of BaseStream that doesn't implement detect()
    should raise TypeError on instantiation."""

    class IncompleteStream(BaseStream):
        """Deliberately incomplete — no detect() method."""

        @property
        def name(self):
            return "incomplete"

    # ABC should prevent instantiation if detect() not implemented
    with pytest.raises(TypeError, match="detect|abstract"):
        IncompleteStream(config={}, global_config={})


def test_basestream_complete_subclass():
    """A properly implemented subclass should instantiate and run."""

    class MockStream(BaseStream):
        @property
        def name(self):
            return "mock_stream"

        def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
            return [SAREvent(
                stream_name=self.name,
                start_frame=0,
                end_frame=len(packets) - 1,
                start_time=0.0,
                end_time=len(packets) / 5.0,
                confidence=0.95,
                z_score=3.0,
                label="test_event",
                severity=EventSeverity.HIGH,
            )]

    stream = MockStream(config={}, global_config={})
    assert stream.name == "mock_stream"

    # Create minimal packets
    packets = [
        FramePacket(index=i, timestamp=i / 5.0,
                    image=np.zeros((100, 100, 3), dtype=np.uint8))
        for i in range(10)
    ]

    events = stream.detect(packets)
    assert len(events) == 1
    assert events[0].stream_name == "mock_stream"
    assert events[0].confidence == 0.95


# ══════════════════════════════════════════════════════════════════════════════
# 3.  test_config_loading
# ══════════════════════════════════════════════════════════════════════════════

def test_config_loading():
    """Config YAML should load and contain all required sections."""
    from main import load_config

    config = load_config()
    assert isinstance(config, dict), "Config should be a dict"

    # Required top-level sections
    required_sections = ["video", "yolo", "action", "motion",
                         "tracking", "concurrency"]
    for section in required_sections:
        assert section in config, \
            f"Config missing required section: '{section}'"

    # Video config
    vid = config["video"]
    assert "target_fps" in vid, "Video config missing 'target_fps'"
    assert isinstance(vid["target_fps"], int), \
        f"target_fps should be int, got {type(vid['target_fps'])}"
    assert 1 <= vid["target_fps"] <= 30, \
        f"target_fps should be 1-30, got {vid['target_fps']}"

    # YOLO config
    yolo = config["yolo"]
    assert "confidence_threshold" in yolo, \
        "YOLO config missing 'confidence_threshold'"
    assert 0 < yolo["confidence_threshold"] < 1, \
        f"confidence_threshold should be (0,1), got {yolo['confidence_threshold']}"

    # Concurrency config
    conc = config["concurrency"]
    assert "gpu_workers" in conc, "Concurrency config missing 'gpu_workers'"
    assert conc["gpu_workers"] >= 1, "gpu_workers must be ≥ 1"
