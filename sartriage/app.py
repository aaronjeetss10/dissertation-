"""
app.py
======
Flask web server for SARTriage.

Endpoints
---------
GET  /                  – Upload interface (drag-and-drop)
POST /upload            – Accept video file, enqueue processing
GET  /status/<task_id>  – JSON progress for polling
GET  /results/<task_id> – Ranked timeline view
GET  /health            – Liveness probe

All video processing happens *locally* on the server (zero external API
calls) to satisfy GDPR / data-privacy constraints for SAR footage.
"""

from __future__ import annotations

import os
import uuid
import time
import threading
from pathlib import Path
from typing import Any, Dict

import yaml
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

# ── Load config ──────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as fh:
    CONFIG: Dict[str, Any] = yaml.safe_load(fh)

# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)

UPLOAD_FOLDER = Path(__file__).parent / CONFIG["flask"]["upload_folder"]
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = CONFIG["flask"]["max_content_mb"] * 1024 * 1024

ALLOWED_EXTENSIONS = set(CONFIG["flask"]["allowed_extensions"])

# ── In-memory task store (replaced by DB/Redis in production) ────────────────

tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


def _allowed_file(filename: str) -> bool:
    """Check if the file extension is permitted."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _run_real_pipeline(task_id: str, video_path: str) -> None:
    """Run the real pipeline orchestrator with progress updates."""
    from main import run_pipeline

    def on_progress(stage: str, fraction: float) -> None:
        with tasks_lock:
            tasks[task_id]["stage"] = stage
            tasks[task_id]["progress"] = fraction

    try:
        result = run_pipeline(
            video_path=video_path,
            config=CONFIG,
            on_progress=on_progress,
            task_id=task_id,
        )
        with tasks_lock:
            tasks[task_id]["status"] = "complete"
            tasks[task_id]["events"] = result["events"]
            tasks[task_id]["summary"] = result["summary"]
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with tasks_lock:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["stage"] = f"Error: {exc}"
            tasks[task_id]["progress"] = 0.0


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the drag-and-drop upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a video file upload and start background processing."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed_file(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Invalid file type. Allowed: {exts}"}), 400

    # Save with unique name to avoid collisions
    task_id = uuid.uuid4().hex[:16]
    ext = Path(file.filename).suffix.lower()
    safe_name = f"{task_id}{ext}"
    save_path = UPLOAD_FOLDER / safe_name
    file.save(str(save_path))

    # Initialise task record
    with tasks_lock:
        tasks[task_id] = {
            "status": "processing",
            "stage": "Queued",
            "progress": 0.0,
            "filename": file.filename,
            "path": str(save_path),
            "created": time.time(),
            "events": [],
            "summary": {},
        }

    # Launch pipeline in background thread
    t = threading.Thread(target=_run_real_pipeline, args=(task_id, str(save_path)), daemon=True)
    t.start()

    return jsonify({"task_id": task_id}), 202


@app.route("/status/<task_id>")
def status(task_id: str):
    """Return current processing progress for a task."""
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({
        "task_id": task_id,
        "status": task["status"],
        "stage": task["stage"],
        "progress": task["progress"],
        "filename": task["filename"],
    })


@app.route("/results/<task_id>")
def results(task_id: str):
    """Render the ranked timeline view for a completed task."""
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return "Task not found", 404
    if task["status"] != "complete":
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        task_id=task_id,
        filename=task["filename"],
        events=task["events"],
        summary=task["summary"],
    )


@app.route("/results/<task_id>/json")
def results_json(task_id: str):
    """Return raw event data as JSON (for programmatic access)."""
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({
        "task_id": task_id,
        "status": task["status"],
        "events": task["events"],
        "summary": task["summary"],
    })


@app.route("/health")
def health():
    """Liveness / readiness probe."""
    return jsonify({"status": "ok", "version": "0.1.0"})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
