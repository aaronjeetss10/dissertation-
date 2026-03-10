/**
 * script.js — SARTriage Upload & Progress Polling
 *
 * Handles:
 *  1. Drag-and-drop + click-to-browse file selection
 *  2. Async upload via fetch()
 *  3. Progress polling via /status/<task_id>
 *  4. Automatic redirect to /results/<task_id> on completion
 */

(function () {
    "use strict";

    // ── Config ──
    const POLL_INTERVAL_MS = 1000;
    const MAX_FILE_SIZE_MB = 2048;

    // ── DOM refs ──
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const clearBtn = document.getElementById("clear-file-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadSection = document.getElementById("upload-section");
    const progressSection = document.getElementById("progress-section");
    const progressBar = document.getElementById("progress-bar");
    const progressPercent = document.getElementById("progress-percent");
    const progressStage = document.getElementById("progress-stage");
    const progressFilename = document.getElementById("progress-filename");

    let selectedFile = null;
    let pollTimer = null;

    // ── Helpers ──

    function formatBytes(bytes) {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const units = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + units[i];
    }

    function isAllowed(name) {
        const ext = "." + name.split(".").pop().toLowerCase();
        return [".mp4", ".avi", ".mov", ".mkv"].includes(ext);
    }

    // ── File selection ──

    function selectFile(file) {
        if (!file) return;
        if (!isAllowed(file.name)) {
            alert("Unsupported format. Please use MP4, AVI, MOV, or MKV.");
            return;
        }
        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            alert(`File too large (${formatBytes(file.size)}). Max is ${MAX_FILE_SIZE_MB} MB.`);
            return;
        }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        fileInfo.classList.remove("hidden");
        uploadBtn.classList.remove("hidden");
        uploadBtn.disabled = false;
        dropZone.classList.add("has-file");
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        uploadBtn.classList.add("hidden");
        uploadBtn.disabled = true;
        dropZone.classList.remove("has-file");
    }

    // ── Drag & Drop ──

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) selectFile(fileInput.files[0]);
    });

    clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

    ["dragenter", "dragover"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add("drag-over"); });
    });

    ["dragleave", "drop"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.remove("drag-over"); });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length) selectFile(files[0]);
    });

    // ── Upload ──

    uploadBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        uploadBtn.disabled = true;
        uploadBtn.textContent = "Uploading…";

        const formData = new FormData();
        formData.append("video", selectedFile);

        try {
            const res = await fetch("/upload", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) {
                alert(data.error || "Upload failed.");
                uploadBtn.disabled = false;
                uploadBtn.textContent = "Begin Analysis";
                return;
            }

            // Switch to progress view
            uploadSection.classList.add("hidden");
            progressSection.classList.remove("hidden");
            progressFilename.textContent = selectedFile.name;

            startPolling(data.task_id);
        } catch (err) {
            console.error("Upload error:", err);
            alert("Network error. Please try again.");
            uploadBtn.disabled = false;
            uploadBtn.textContent = "Begin Analysis";
        }
    });

    // ── Progress Polling ──

    function startPolling(taskId) {
        const stages = document.querySelectorAll(".stage-item");
        let completedStages = new Set();

        pollTimer = setInterval(async () => {
            try {
                const res = await fetch(`/status/${taskId}`);
                const data = await res.json();

                if (!res.ok) { clearInterval(pollTimer); return; }

                // Update progress bar
                const pct = Math.round(data.progress * 100);
                progressBar.style.width = pct + "%";
                progressPercent.textContent = pct + "%";
                progressStage.textContent = data.stage;

                // Update stage list
                stages.forEach((item) => {
                    const stageName = item.dataset.stage;
                    const icon = item.querySelector(".stage-icon");
                    if (data.stage === stageName) {
                        icon.className = "stage-icon active";
                        item.classList.add("current");
                    } else if (completedStages.has(stageName)) {
                        icon.className = "stage-icon done";
                        item.classList.remove("current");
                    }
                });

                // Mark all stages before current as completed
                let found = false;
                stages.forEach((item) => {
                    if (item.dataset.stage === data.stage) { found = true; return; }
                    if (!found) { completedStages.add(item.dataset.stage); }
                });

                // Redirect on completion
                if (data.status === "complete") {
                    clearInterval(pollTimer);
                    progressBar.style.width = "100%";
                    progressPercent.textContent = "100%";
                    progressStage.textContent = "Complete — redirecting…";

                    // Mark all done
                    stages.forEach((item) => {
                        item.querySelector(".stage-icon").className = "stage-icon done";
                        item.classList.remove("current");
                    });

                    setTimeout(() => { window.location.href = `/results/${taskId}`; }, 800);
                }
            } catch (err) {
                console.error("Poll error:", err);
            }
        }, POLL_INTERVAL_MS);
    }
})();
