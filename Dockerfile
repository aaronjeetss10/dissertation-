# ============================================================================
# SARTriage — Multi-Stage Docker Build
# ============================================================================
# NFR4: Ubuntu 20.04+, Python 3.9+, Docker deployment
#
# Build:  docker build -t sartriage .
# Run:    docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads sartriage
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="SARTriage <sartriage@example.com>"
LABEL description="SARTriage: Multi-Stream Aerial Video Triage for SAR"
LABEL version="1.0.0"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY sartriage/ /app/sartriage/

# Copy tests (for in-container testing)
COPY tests/ /app/tests/

# Create directories for runtime data
RUN mkdir -p /app/uploads /app/results /app/models

# Copy model files if present (optional — can be mounted as volume)
COPY sartriage/models/*.pt /app/models/ 2>/dev/null || true

# Environment configuration
ENV FLASK_APP=sartriage.app:app \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SARTRIAGE_UPLOAD_DIR=/app/uploads \
    SARTRIAGE_RESULTS_DIR=/app/results \
    SARTRIAGE_MODEL_DIR=/app/models \
    SARTRIAGE_WORKERS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

EXPOSE 5000

# Run with gunicorn for production
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers ${SARTRIAGE_WORKERS} --timeout 300 --access-logfile - sartriage.app:app"]
