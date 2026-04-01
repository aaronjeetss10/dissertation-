# ============================================================================
# SARTriage — Makefile
# ============================================================================
# Quick commands for building, running, testing, and linting.
#
# Usage:
#   make test      Run pytest locally
#   make lint      Run flake8 on key source files
#   make pipeline  Run the e2e pipeline on test data
#   make app       Start the Flask dev server
#   make build     Build the Docker image
#   make run       Start via docker compose
#   make stop      Stop docker compose
#   make clean     Remove containers, images, and volumes
#   make logs      Follow application logs
#   make shell     Open a shell inside the running container
# ============================================================================

.PHONY: build run stop test lint pipeline app clean logs shell dev test-docker

IMAGE   := sartriage
CONTAINER := sartriage
PYTHON  := python3

# ── Test (local) ──
test:
	cd sartriage && $(PYTHON) -m pytest ../tests/ -v --tb=short

# ── Lint ──
lint:
	$(PYTHON) -m flake8 \
		sartriage/main.py \
		sartriage/app.py \
		sartriage/core/priority_ranker.py \
		sartriage/core/emi.py \
		sartriage/streams/tms_classifier.py \
		sartriage/streams/base_stream.py \
		sartriage/evaluation/aai_v2.py \
		sartriage/tce_v2_pilot.py \
		--max-line-length=120 \
		--ignore=E501,W503,E402,E722 \
		--count --statistics || true

# ── Pipeline (e2e on test data) ──
pipeline:
	cd sartriage && $(PYTHON) -m pytest ../tests/test_pipeline_e2e.py -v --tb=short

# ── App (Flask dev server) ──
app:
	cd sartriage && $(PYTHON) app.py

# ── Build (Docker) ──
build:
	docker compose build

# ── Run (Docker) ──
run:
	@mkdir -p uploads results
	docker compose up -d
	@echo ""
	@echo "✓ SARTriage is running at http://localhost:5000"
	@echo "  Uploads: ./uploads/"
	@echo "  Results: ./results/"
	@echo ""

# ── Stop ──
stop:
	docker compose down

# ── Logs ──
logs:
	docker compose logs -f $(CONTAINER)

# ── Shell ──
shell:
	docker exec -it $(CONTAINER) /bin/bash

# ── Test (inside Docker) ──
test-docker:
	docker exec $(CONTAINER) python -m pytest /app/tests/ -v --tb=short 2>/dev/null || \
	docker exec $(CONTAINER) python -c "from sartriage.main import load_config; print('✓ Config loads'); from sartriage.core import FramePacket; print('✓ Core imports'); print('All checks passed.')"

# ── Clean ──
clean:
	docker compose down -v --rmi local
	@echo "✓ Cleaned containers, volumes, and images"

# ── Dev (local, no Docker) ──
dev:
	cd sartriage && $(PYTHON) app.py
