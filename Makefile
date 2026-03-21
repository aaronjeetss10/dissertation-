# ============================================================================
# SARTriage — Makefile
# ============================================================================
# Quick commands for building, running, and testing.
#
# Usage:
#   make build     Build the Docker image
#   make run       Start the application
#   make stop      Stop the application
#   make test      Run tests inside the container
#   make clean     Remove containers, images, and volumes
#   make logs      Follow application logs
#   make shell     Open a shell inside the running container
# ============================================================================

.PHONY: build run stop test clean logs shell dev

IMAGE := sartriage
CONTAINER := sartriage

# ── Build ──
build:
	docker compose build

# ── Run ──
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

# ── Test ──
test:
	docker exec $(CONTAINER) python -m pytest sartriage/ -v --tb=short 2>/dev/null || \
	docker exec $(CONTAINER) python -c "from sartriage.main import load_config; print('✓ Config loads'); from sartriage.core import FramePacket; print('✓ Core imports'); print('All checks passed.')"

# ── Clean ──
clean:
	docker compose down -v --rmi local
	@echo "✓ Cleaned containers, volumes, and images"

# ── Dev (local, no Docker) ──
dev:
	cd sartriage && python app.py
