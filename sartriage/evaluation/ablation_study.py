"""
evaluation/ablation_study.py
==============================
Systematic ablation study for the SARTriage pipeline.

Tests each stream individually and in combination to measure
contribution.  Generates a LaTeX-ready table and JSON results
for the dissertation.

Usage:
    cd sartriage
    python evaluation/ablation_study.py --video path/to/test.mp4

Output:
    evaluation/results/ablation_results.json
    evaluation/results/ablation_table.txt  (LaTeX-ready)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline_with_config(
    video_path: str,
    config: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    """Run the pipeline and collect metrics."""
    from main import run_pipeline

    t0 = time.time()
    result = run_pipeline(video_path, config=config, task_id=f"ablation_{label}")
    elapsed = time.time() - t0

    events = result.get("events", [])

    # Aggregate metrics
    n_events = len(events)
    n_critical = sum(1 for e in events if e.get("severity") == "critical")
    n_high = sum(1 for e in events if e.get("severity") == "high")
    n_medium = sum(1 for e in events if e.get("severity") == "medium")
    n_low = sum(1 for e in events if e.get("severity") == "low")

    # Stream breakdown
    stream_counts = {}
    for e in events:
        streams = e.get("streams", [e.get("stream", "unknown")])
        if isinstance(streams, str):
            streams = [streams]
        for s in streams:
            stream_counts[s] = stream_counts.get(s, 0) + 1

    # Average confidence and z-score
    confidences = [e.get("confidence", 0) for e in events]
    z_scores = [e.get("z_score", 0) for e in events]

    avg_conf = sum(confidences) / max(len(confidences), 1)
    avg_z = sum(z_scores) / max(len(z_scores), 1)

    # Cross-stream events
    cross_stream = sum(1 for e in events
                       if len(e.get("streams", [])) > 1)

    return {
        "label": label,
        "total_events": n_events,
        "critical": n_critical,
        "high": n_high,
        "medium": n_medium,
        "low": n_low,
        "stream_counts": stream_counts,
        "avg_confidence": round(avg_conf, 4),
        "avg_z_score": round(avg_z, 4),
        "cross_stream_events": cross_stream,
        "processing_time_s": round(elapsed, 1),
        "video_duration_s": result.get("video_duration_s", 0),
    }


def format_latex_table(results: List[Dict[str, Any]]) -> str:
    """Format results as a LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation study results — contribution of each pipeline component}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Configuration & Events & Critical & High & Avg Conf. & Avg Z & Time (s) \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"  {r['label']} & {r['total_events']} & {r['critical']} & "
            f"{r['high']} & {r['avg_confidence']:.2f} & "
            f"{r['avg_z_score']:.2f} & {r['processing_time_s']:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SARTriage Ablation Study")
    parser.add_argument("--video", required=True, help="Path to test video")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    base_config = load_config()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SARTriage — Ablation Study")
    print(f"  Video: {video_path}")
    print("=" * 70)

    # Define ablation configurations
    ablations = []

    # 1. Full pipeline (all 5 streams)
    ablations.append(("Full pipeline (5 streams)", base_config.copy()))

    # 2. Without anomaly detection
    cfg = base_config.copy()
    cfg["anomaly"] = {**cfg.get("anomaly", {}), "enabled": False}
    ablations.append(("Without anomaly", cfg))

    # 3. Without pose estimation
    cfg = base_config.copy()
    cfg["pose"] = {**cfg.get("pose", {}), "enabled": False}
    ablations.append(("Without pose", cfg))

    # 4. Without action classifier
    cfg = base_config.copy()
    cfg["action"] = {**cfg.get("action", {}), "enabled": False}
    ablations.append(("Without action", cfg))

    # 5. Without motion detector
    cfg = base_config.copy()
    cfg["motion"] = {**cfg.get("motion", {}), "enabled": False}
    ablations.append(("Without motion", cfg))

    # 6. Without cross-stream boosting
    cfg = base_config.copy()
    cfg["ranker"] = {**cfg.get("ranker", {}), "cross_stream_boost": 1.0}
    ablations.append(("No cross-stream boost", cfg))

    # 7. Without SAHI
    cfg = base_config.copy()
    cfg["yolo"] = {**cfg.get("yolo", {}), "use_sahi": False}
    ablations.append(("Without SAHI", cfg))

    # 8. Only motion + tracking (no ML)
    cfg = base_config.copy()
    cfg["action"] = {**cfg.get("action", {}), "enabled": False}
    cfg["pose"] = {**cfg.get("pose", {}), "enabled": False}
    cfg["anomaly"] = {**cfg.get("anomaly", {}), "enabled": False}
    ablations.append(("Motion+Tracking only", cfg))

    # Run all configurations
    all_results = []
    for i, (label, config) in enumerate(ablations):
        print(f"\n[{i+1}/{len(ablations)}] Running: {label}")
        print("-" * 50)
        try:
            result = run_pipeline_with_config(video_path, config, label.replace(" ", "_"))
            all_results.append(result)
            print(f"  → {result['total_events']} events "
                  f"(C:{result['critical']}, H:{result['high']}, "
                  f"M:{result['medium']}, L:{result['low']}) "
                  f"in {result['processing_time_s']:.1f}s")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            all_results.append({"label": label, "error": str(exc)})

    # Save results
    results_path = results_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")

    # Generate LaTeX table
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        latex = format_latex_table(valid_results)
        latex_path = results_dir / "ablation_table.txt"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"✓ LaTeX table: {latex_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("  ABLATION STUDY RESULTS")
        print(f"{'='*70}")
        print(f"{'Configuration':<30} {'Events':>7} {'Crit':>6} {'High':>6} "
              f"{'AvgConf':>8} {'Time':>7}")
        print("-" * 70)
        for r in valid_results:
            print(f"  {r['label']:<28} {r['total_events']:>7} "
                  f"{r['critical']:>6} {r['high']:>6} "
                  f"{r['avg_confidence']:>8.3f} {r['processing_time_s']:>6.1f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
