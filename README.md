# GAN Studio — Image + Text Generation (Flask + PyTorch)

A full-stack app that trains and serves two Generative Adversarial Networks
from a browser UI: a fully-connected GAN on MNIST digits, and an
LSTM-based GAN for short text sequences. Training runs in a background
thread on the Flask server; the frontend polls for live status and plots
loss curves in real time.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.9-red)

## Architecture

```
┌──────────────────┐   same-origin REST (JSON / polling)   ┌──────────────────┐
│  index.html       │ ─────────────────────────────────▶   │   Flask (app.py)  │
│  (vanilla JS,     │ ◀─────────────────────────────────   │   /api/image/*     │
│   Chart.js)       │      served BY the same Flask app     │   /api/text/*      │
└──────────────────┘                                        └─────────┬────────┘
                                                                        │ background thread
                                                              ┌─────────▼────────┐
                                                              │  PyTorch models   │
                                                              │  Image: MLP GAN   │
                                                              │  Text: LSTM GAN   │
                                                              └──────────────────┘
```

Flask now serves `frontend/index.html` directly (in addition to the
`/api/*` routes), so the whole app is one process on one port — locally
that's `http://localhost:5000`, and in production it's whatever URL your
host gives the single web service.

- **Image GAN** — fully-connected GAN (generator: 100 → 256 → 512 → 1024 →
  784, BatchNorm + LeakyReLU; discriminator: mirrored with Dropout),
  trained on real MNIST via `torchvision.datasets.MNIST`.
- **Text GAN** — generator is an LSTM over a learned latent-conditioned
  sequence, discriminator is a small Conv1D network, trained with
  **Gumbel-Softmax + straight-through estimator** so gradients can reach
  the generator through discrete token sampling.

## Setup (local)

Requires Python 3.10+.

```bash
cd backend
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py                 # → http://localhost:5000
```

Then open **http://localhost:5000** — the backend now serves the frontend
directly, so there's nothing else to open or configure.

Or use the provided `start.sh` (macOS/Linux) / `start.bat` (Windows) to do
all of the above in one step:

```bash
./start.sh
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server + device (CPU/GPU) check |
| `/api/models` | GET | Whether trained checkpoints exist |
| `/api/image/train` | POST | Start image GAN training (`{"epochs": 30}`) |
| `/api/image/stop` | POST | Stop the current image training run |
| `/api/image/status` | GET | Live epoch / loss / status |
| `/api/image/generate` | GET | Sample a grid of generated digits (`?n=16`) |
| `/api/text/train` | POST | Start text GAN training (`{"epochs": 60}`) |
| `/api/text/stop` | POST | Stop the current text training run |
| `/api/text/status` | GET | Live epoch / loss / status |
| `/api/text/generate` | GET | Sample generated sentences (`?n=8&temperature=1.0`) |

## Environment variables

See `.env.example`. All optional — sensible defaults are used if unset.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Port Flask/gunicorn binds to (Render sets this automatically) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for `/api/*`. Only matters if you host the frontend separately from this backend. |
| `FLASK_DEBUG` | `false` | Enable Flask debug/reload mode locally |

## Deployment (Render)

This repo includes `render.yaml`, so you can deploy with Render's
"Blueprint" flow:

1. Push this repo to GitHub (see commands below).
2. In Render: **New → Blueprint**, point it at your GitHub repo.
3. Render reads `render.yaml` and provisions a single free web service that
   runs `pip install -r backend/requirements.txt` then
   `gunicorn --chdir backend app:app`.
4. Once deployed, visit the URL Render gives you — the same URL serves
   both the UI and the API (`/api/...`).

No separate frontend host is needed — Flask serves `frontend/index.html`
itself, so this is a single-service deployment.

**Manual (Render, Railway, or any Python host, without the blueprint):**

```
Build command: pip install -r backend/requirements.txt
Start command: gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

`--workers 1` is intentional: training progress is kept in an in-process
Python dict, so multiple worker processes would each show different
(incomplete) status. `--threads 4` still lets the app handle concurrent
requests (health checks, status polling, training) within that one worker.

## Local commands (copy/paste)

```bash
# Setup + run
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → open http://localhost:5000
```

## Push to GitHub

```bash
git init                                   # already done if you got this repo via git
git add .
git commit -m "Initial commit: working GAN Studio app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## What was fixed

This project was audited end-to-end and the following issues were fixed:

1. **Hardcoded frontend API URL** — the frontend always called
   `http://localhost:5000/api`, which would 404/CORS-fail once deployed
   anywhere else. It now auto-detects: same-origin relative `/api` when
   Flask is serving the page itself (local *or* production), falling back
   to `http://localhost:5000/api` only when opened as a raw file or from a
   separate local dev server.
2. **Broken nav-highlight bug** — `showPage()`'s index map
   (`{dashboard:0, image:2, text:3, about:5}`) didn't match the actual
   `.nav-item` DOM order, so clicking "Image GAN"/"Text GAN" highlighted
   the wrong sidebar entry, and clicking "Architecture" threw a
   `TypeError` in the console (`items[5]` was `undefined`). Fixed to
   `{dashboard:0, image:1, text:2, about:3}`.
3. **No production server config** — `app.run()` had a hardcoded port and
   no host binding, so it couldn't run on a host like Render that assigns
   its own `$PORT`. Now reads `PORT`/`FLASK_DEBUG` from the environment
   and binds `0.0.0.0`.
4. **Frontend and backend were two separate, unconnected pieces** —
   there was no way to deploy this as a single service. Flask now serves
   `frontend/index.html` and returns a JSON 404 for unknown `/api/*`
   routes (instead of Flask's default HTML error page, which the frontend
   can't parse as JSON).
5. **Hardcoded open CORS** — `CORS(app)` allowed all origins on every
   route unconditionally. Scoped to `/api/*` and made configurable via
   `CORS_ORIGINS`.
6. **Windows line endings (CRLF)** — `app.py`, `index.html`, and
   `start.bat` had CRLF line endings, which can cause `no such file or
   directory` errors when running scripts directly on Linux/macOS.
   Normalized to LF.
7. **No production WSGI server / deployment config** — the dev server
   (`app.run`) isn't meant for production. Added `gunicorn` to
   `requirements.txt`, plus `Procfile`, `render.yaml`, and `runtime.txt`
   for Render deployment, and `.env.example` documenting the supported
   environment variables.
8. **PyTorch default wheel is GPU-sized** — plain `pip install torch`
   pulls in the full CUDA toolkit (~4 GB of NVIDIA libraries), which is
   unnecessary (and can fail to fit within free-tier build limits) on a
   GPU-less host. Added `--extra-index-url
   https://download.pytorch.org/whl/cpu` to `requirements.txt` so
   deployments install the much smaller CPU-only build instead.
9. **Incomplete `.gitignore`** — missing `.env`, IDE folders, log files,
   and other common Python-project entries. Expanded.

## Known limitations

- The text GAN's vocabulary and "real" training sentences are a small
  hand-written template set (~100 words), not a real corpus — so outputs
  are template-adjacent, not fluent. Swapping in a real dataset (e.g. a
  sentence corpus + BPE tokenizer) would be the natural next step.
- Training state is kept in-process memory — restarting the server loses
  progress, and (as noted above) the deployment must run as a single
  worker process. Fine for a demo; would need a persistent store
  (Redis/DB) for multi-instance or production-scale use.
- No auth on training endpoints — anyone with network access can trigger
  a training run. Not addressed here since this is a local/demo tool.

## Tech stack

Python, PyTorch, Flask, Flask-CORS, gunicorn, NumPy, Pillow, Chart.js,
vanilla JS/CSS.
