# NeuroSim Lab — convenience targets (plain npm/pip/uvicorn underneath).
PY ?= python3

.PHONY: setup test build dev-backend dev-frontend run-prod e2e zip clean

setup: ## create backend venv + install all deps (backend & frontend)
	cd backend && $(PY) -m venv .venv
	cd backend && .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
	cd backend && .venv/bin/pip install --quiet -r requirements.txt
	cd frontend && npm install

test: ## run backend pytest + frontend vitest
	cd backend && .venv/bin/python -m pytest tests -q
	cd frontend && npx vitest run

build: ## production-build the SPA (frontend/dist)
	cd frontend && npm run build

dev-backend: ## backend with hot reload on :8000
	cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## vite dev server on :5173
	cd frontend && npm run dev

run-prod: build ## build SPA then serve everything from :8000
	cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

e2e: ## run the scripted end-to-end verification (backend must be running)
	$(PY) scripts/e2e_check.py

zip: ## produce a clean portable ZIP
	$(PY) scripts/make_zip.py

clean: ## delete runtime data (DB/uploads/weights) and build artifacts
	rm -rf frontend/dist data/*.db data/uploads/* data/runs/*
