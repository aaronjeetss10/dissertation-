# SARTriage — Deployment Guide

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/aaronjeetss10/dissertation-.git
cd dissertation-

# 2. Build and start
make build
make run

# 3. Open in browser
open http://localhost:5000
```

That's it. The application is now running.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Ubuntu 20.04+ / macOS 12+ | Ubuntu 22.04 LTS |
| **Python** | 3.9+ | 3.11 |
| **Docker** | 20.10+ | 24.0+ |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 5 GB (image + models) | 10 GB (with video storage) |
| **GPU** | None (CPU works) | NVIDIA with CUDA 11.8+ |

### GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Uncomment the GPU service in docker-compose.yml
# Then: docker compose up sartriage-gpu -d
```

---

## Configuration

All configuration is via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `SARTRIAGE_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |
| `SARTRIAGE_WORKERS` | `2` | Gunicorn worker processes |
| `SARTRIAGE_UPLOAD_DIR` | `/app/uploads` | Video upload directory |
| `SARTRIAGE_RESULTS_DIR` | `/app/results` | Processing results directory |
| `SARTRIAGE_MODEL_DIR` | `/app/models` | Model weights directory |
| `SARTRIAGE_MAX_UPLOAD_MB` | `500` | Maximum upload file size |
| `FLASK_ENV` | `production` | Flask environment |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Container                                │
│  ┌───────────────────────────────────────────┐  │
│  │  Gunicorn (2 workers)                      │  │
│  │  ├── Flask Web App (port 5000)            │  │
│  │  │   ├── Upload endpoint                  │  │
│  │  │   ├── Processing pipeline              │  │
│  │  │   └── Results viewer                   │  │
│  │  └── SARTriage Pipeline                   │  │
│  │      ├── YOLO11n  (detection)             │  │
│  │      ├── ByteTrack (tracking)             │  │
│  │      ├── MViTv2-S (action classification) │  │
│  │      ├── TMS (trajectory classification)  │  │
│  │      └── Priority Ranker                  │  │
│  └───────────────────────────────────────────┘  │
│  Volumes: /app/uploads, /app/results, /app/models│
└─────────────────────────────────────────────────┘
```

---

## Makefile Commands

| Command | Description |
|---|---|
| `make build` | Build the Docker image |
| `make run` | Start the application (detached) |
| `make stop` | Stop the application |
| `make logs` | Follow application logs |
| `make shell` | Open bash inside the container |
| `make test` | Run smoke tests |
| `make clean` | Remove containers, images, and volumes |
| `make dev` | Run locally without Docker |

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs sartriage

# Common fix: ensure ports aren't in use
lsof -i :5000
```

### Out of memory
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 12G  # Increase from 8G
```

### Slow processing (no GPU)
- CPU inference takes ~30s per 10s video clip
- Set `SARTRIAGE_WORKERS=1` to reduce memory usage
- Consider the NVIDIA GPU variant for production

### Model files missing
```bash
# Models should be in sartriage/models/
ls sartriage/models/
# Expected: yolo11n.pt, action_mvit2_sar.pt

# If missing, the pipeline uses synthetic stubs for demonstration
```

### Upload fails
```bash
# Check upload directory permissions
docker exec sartriage ls -la /app/uploads

# Increase max upload size
# Set SARTRIAGE_MAX_UPLOAD_MB=1000 in docker-compose.yml
```

---

## Production Deployment

For field deployment on a SAR operations laptop:

```bash
# 1. Pre-build image (requires internet)
make build

# 2. Save image for offline use
docker save sartriage:latest | gzip > sartriage-image.tar.gz

# 3. On target machine (no internet needed)
docker load < sartriage-image.tar.gz
make run
```

---

## NFR Compliance

| NFR | Requirement | Status |
|---|---|---|
| NFR4.1 | Ubuntu 20.04+ | ✅ python:3.11-slim (Debian bookworm) |
| NFR4.2 | Python 3.9+ | ✅ Python 3.11 |
| NFR4.3 | Docker deployment | ✅ Dockerfile + docker-compose |
| NFR4.4 | GPU optional | ✅ CPU default, CUDA variant available |
| NFR4.5 | < 5 min setup | ✅ 3 commands: clone, build, run |
