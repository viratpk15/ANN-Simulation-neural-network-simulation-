# Troubleshooting — NeuroSim Lab

| Symptom | Likely cause → fix |
|---|---|
| **`Port 8000/5173 already in use`** | Another process holds it. Linux/macOS: `lsof -i :8000` → `kill <pid>`. Windows: `netstat -ano \| findstr :8000` → `taskkill /PID <pid> /F`. Or set `BACKEND_PORT=9000` in `.env` and update the vite proxy. |
| **"● API offline" badge** | Backend not running or crashed. Start it (`uvicorn app.main:app --port 8000` from `backend/`) and check its terminal for the traceback. Health-check manually: `curl http://localhost:8000/api/health`. |
| **CORS errors in browser console** | You opened the SPA from an unexpected origin (e.g. file:// or another port). Fix: use `http://localhost:5173` (dev) or `http://localhost:8000` (single-port). Custom origins: `CORS_ORIGINS=http://yourhost:port` in `.env`. |
| **Vite "Blocked request. This host is not allowed"** | Accessing via a proxy/tunnel host. `server.allowedHosts: true` is set in `vite.config.ts` — if you edited it, restore that line and restart `npm run dev`. |
| **WebSocket drops / training stuck at "queued"** | A proxy in front (nginx, cloud IDE) blocks WS. Dev mode proxies WS automatically. In production frontends behind nginx add `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";` for `/ws`. |
| **PyTorch install failure / huge download** | Use the CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (≈200 MB). Python < 3.10 is unsupported — upgrade Python. |
| **Node/npm build fails** | Requires Node ≥ 18 — `node --version`. `npm ci` for a clean lockfile install. Out of memory on tiny VMs: `NODE_OPTIONS=--max-old-space-size=4096 npm run build`. |
| **"Input layer expects X features but the prepared dataset has Y"** | Exactly what it says: one-hot encoding changed the count (Dataset tab shows model features). Adjust the Input layer or feature selection; the validator also flags this before training. |
| **Training fails: loss/output mismatch (400)** | Follow the task table in the User Guide (cross-entropy → k softmax/linear outputs; BCE → 1 sigmoid; regression → 1 linear output + MSE/MAE). |
| **"Signature did not match" / CSV upload rejected** | Only `.csv`, ≤ `MAX_UPLOAD_MB` (default 10), ≤ 200k×500 cells, parseable by pandas, at least 2 columns. Export from Excel as "CSV UTF-8". |
| **Custom formula rejected** | Only `x`, numbers, `+ - * / ^`, whitelisted functions (`exp log sqrt sin cos tan abs tanh sigmoid relu softplus min max pow clip …`), constants `pi`, `e`. Attributes, imports, names other than `x` are blocked **by design** (security). The error message tells you exactly what was disallowed. |
| **Formula validates but NaN while training** | Look at the f(x) plot — domain holes (log/sqrt of negatives, division by 0). Train with standardized inputs or clip the formula. |
| **Diagnosis has no LLM explanation** | The deterministic findings always work. Open the `▸ providers` list in the AI Diagnosis panel — it shows which of `groq → openrouter → ollama` are configured. A missing key shows as "API key missing". |
| **Ollama fallback doesn't reply** | `ollama serve` running? Model pulled (`ollama pull llama3.1`)? `OLLAMA_BASE_URL` must include the `/v1` suffix. A wrong model name returns a 404 and the chain stops cleanly — deterministic diagnosis is unaffected. |
| **All providers failing / diagnosis slow** | Lower `LLM_TIMEOUT_S`. Each provider gets at most `1 + LLM_MAX_RETRIES` attempts and the chain is walked once, so total failure is bounded by `providers × (1 + retries) × timeout`. Set `LLM_ENABLED=false` to skip the layer entirely. |
| **Everything slow** | Keep epochs ≤ a few hundred, neurons ≤ 128/layer for live demos; datasets are downsampled by design in views but training uses everything. Max 3 concurrent jobs. |
| **Where is my data?** | `data/neurosim.db` (SQLite), `data/uploads/*.csv`, `data/runs/<run_id>.pt`. Delete the `data/` directory to factory-reset. **Never commit it.** |
| **Tests fail after pulling updates** | `cd backend && python -m pytest tests -q` — failures print exact assertions; most common: stale `data/` from an older schema → delete `data/`. |
