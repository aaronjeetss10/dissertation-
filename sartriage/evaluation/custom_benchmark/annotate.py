"""
evaluation/custom_benchmark/annotate.py
=========================================
Simple CLI annotation tool for DJI Neo benchmark clips.

For each clip in a directory, plays the video and asks the user to annotate:
  - action_correct: Was the intended action visible? (y/n)
  - person_visible: Was a person detectable? (y/n)
  - start_frame: Frame where action begins
  - end_frame: Frame where action ends
  - quality: 1-5 (1=unusable, 5=perfect)
  - notes: Free-text

Saves annotations to annotations.json in the same directory.

Run:
    python evaluation/custom_benchmark/annotate.py /path/to/clips/
"""

from __future__ import annotations

import argparse, json, sys, re
from pathlib import Path
from typing import List

_FILENAME_RE = re.compile(
    r"^(?P<action>[a-z_]+)_(?P<altitude>\d+)m_(?P<take>\d+)\.mp4$",
    re.IGNORECASE
)


def discover_clips(input_dir: Path) -> List[dict]:
    """Find all benchmark clips."""
    clips = []
    for path in sorted(input_dir.glob("*.mp4")):
        m = _FILENAME_RE.match(path.name)
        if m:
            clips.append({
                "path": str(path),
                "filename": path.name,
                "action": m.group("action").lower(),
                "altitude_m": int(m.group("altitude")),
                "take": int(m.group("take")),
            })
    return clips


def play_clip(path: str):
    """Play a clip using OpenCV (or print message if unavailable)."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"    ⚠ Cannot open {path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"    Playing: {total} frames at {fps:.0f}fps. "
              f"Press 'q' to stop, SPACE to pause.")

        paused = False
        frame_idx = 0
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Add frame counter overlay
                cv2.putText(frame, f"Frame {frame_idx}/{total}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)
                cv2.imshow("Annotate — press Q to continue", frame)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

    except ImportError:
        print(f"    ⚠ OpenCV not available for playback.")
        print(f"    Please view the clip manually: {path}")


def annotate_clip(clip: dict) -> dict:
    """Interactively annotate a single clip."""
    print(f"\n  ── {clip['filename']} ──")
    print(f"    Action: {clip['action']}, Altitude: {clip['altitude_m']}m, "
          f"Take: {clip['take']}")

    play_clip(clip["path"])

    # Collect annotations
    print()
    person_visible = input("    Person visible? (y/n): ").strip().lower() == "y"
    action_correct = False
    if person_visible:
        action_correct = input("    Action correctly performed? (y/n): ").strip().lower() == "y"

    start_frame = 0
    end_frame = 0
    try:
        start_frame = int(input("    Start frame of action (0 if unknown): ") or "0")
        end_frame = int(input("    End frame of action (0 if unknown): ") or "0")
    except ValueError:
        pass

    try:
        quality = int(input("    Quality 1-5 (1=unusable, 5=perfect): ") or "3")
        quality = max(1, min(5, quality))
    except ValueError:
        quality = 3

    notes = input("    Notes (optional): ").strip()

    return {
        "filename": clip["filename"],
        "action": clip["action"],
        "altitude_m": clip["altitude_m"],
        "take": clip["take"],
        "person_visible": person_visible,
        "action_correct": action_correct,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "quality": quality,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Annotate benchmark clips")
    parser.add_argument("input_dir", type=str,
                        help="Directory containing benchmark clips")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing annotations (skip already done)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    clips = discover_clips(input_dir)
    print(f"\nFound {len(clips)} clips to annotate")

    # Load existing annotations
    output_path = input_dir / "annotations.json"
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
            existing = {a["filename"]: a for a in data.get("annotations", [])}
        print(f"Resuming: {len(existing)} already annotated")

    annotations = list(existing.values())

    for i, clip in enumerate(clips):
        if clip["filename"] in existing:
            print(f"  [{i+1}/{len(clips)}] {clip['filename']} — already annotated, skipping")
            continue

        print(f"\n  [{i+1}/{len(clips)}] ", end="")
        ann = annotate_clip(clip)
        annotations.append(ann)

        # Save after each annotation (in case of crash)
        with open(output_path, "w") as f:
            json.dump({"annotations": annotations}, f, indent=2)

    print(f"\n✓ {len(annotations)} annotations saved to {output_path}")


if __name__ == "__main__":
    main()
