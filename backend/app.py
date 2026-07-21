
import os, io, base64, threading
from functools import wraps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
import torchvision.utils as vutils
import matplotlib; matplotlib.use("Agg")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
DATA_DIR     = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# Flask serves the built frontend directly, so the whole app (API + UI)
# runs from a single process/port — this is what makes `python app.py`
# (or one gunicorn command) a complete one-command deployment.
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# CORS_ORIGINS env var lets you lock this down in production, e.g.
# CORS_ORIGINS="https://myapp.onrender.com". Defaults to "*" for local dev
# and for the common case where the frontend is served by this same app
# (same-origin requests don't need CORS at all — this only matters if the
# frontend is hosted separately, e.g. static hosting + a separate API host).
_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": _origins.split(",") if _origins != "*" else "*"}})

# ──────────────────────────────────────────────────────────
#  AUTH (training endpoints only)
# ──────────────────────────────────────────────────────────
# Lightweight shared-secret gate for the endpoints that actually cost
# compute/time (starting/stopping training). Not a full user/session auth
# system — just enough to stop a random visitor from spinning up training
# jobs on your deployed instance. If API_KEY is unset, these routes stay
# open (matches previous behaviour — convenient for local/demo use).
API_KEY = os.environ.get("API_KEY", "")

def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if API_KEY:
            supplied = request.headers.get("X-API-Key", "")
            if not supplied or supplied != API_KEY:
                return jsonify({"error": "Unauthorized — missing/invalid X-API-Key header"}), 401
        return f(*args, **kwargs)
    return wrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
 
IMG_LATENT = 100
IMG_DIM    = 784
TXT_LATENT = 64
VOCAB_SIZE = 2000
SEQ_LEN    = 15
 
# ──────────────────────────────────────────────────────────
#  IMAGE GAN
# ──────────────────────────────────────────────────────────
class ImageGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, norm=True):
            L = [nn.Linear(i, o)]
            if norm: L.append(nn.BatchNorm1d(o, momentum=0.8))
            L.append(nn.LeakyReLU(0.2, inplace=True))
            return L
        self.model = nn.Sequential(
            *blk(IMG_LATENT, 256, False), *blk(256, 512), *blk(512, 1024),
            nn.Linear(1024, IMG_DIM), nn.Tanh()
        )
    def forward(self, z): return self.model(z).view(-1, 1, 28, 28)
 
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_DIM, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),       nn.Sigmoid()
        )
    def forward(self, x): return self.model(x.view(x.size(0), -1))
 
# ──────────────────────────────────────────────────────────
#  TEXT GAN
# ──────────────────────────────────────────────────────────
class TextGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in  = nn.Linear(TXT_LATENT, 256)
        self.lstm   = nn.LSTM(256, 512, batch_first=True, num_layers=2)
        self.fc_out = nn.Linear(512, VOCAB_SIZE)
    def forward(self, z):
        """Returns raw logits, shape (B, SEQ_LEN, VOCAB_SIZE). Softmax is applied
        by the caller (differently) depending on whether we're training (Gumbel-Softmax,
        differentiable) or generating text (plain softmax + sampling)."""
        x = torch.relu(self.fc_in(z)).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)

    def sample_gumbel(self, z, tau=1.0, hard=True):
        """Differentiable discrete-ish sample used during adversarial training.
        hard=True gives a one-hot forward pass but a soft gradient on the backward
        pass (the 'straight-through' trick), so gradients can actually reach the
        generator's weights through the discriminator."""
        logits = self.forward(z)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
 
class TextDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 64)
        self.conv  = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 3, padding=1), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * SEQ_LEN, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, t):
        # Real samples arrive as LongTensor token ids -> normal embedding lookup.
        # Generated samples arrive as soft/one-hot float vectors (from
        # sample_gumbel) -> use a differentiable "soft embedding lookup" instead,
        # so gradients can flow back through this op into the generator.
        if t.dtype == torch.long:
            e = self.embed(t)                 # (B, SEQ, 64)
        else:
            e = t @ self.embed.weight         # (B, SEQ, VOCAB) @ (VOCAB, 64) -> (B, SEQ, 64)
        return self.fc(self.conv(e.permute(0,2,1)).view(t.size(0), -1))
 
# ──────────────────────────────────────────────────────────
#  VOCABULARY
# ──────────────────────────────────────────────────────────
BASE_WORDS = [
    "<pad>","<unk>","the","a","an","is","are","was","were","be","has","have",
    "can","will","would","could","should","may","might","neural","network",
    "model","data","training","deep","learning","machine","language","natural",
    "image","text","generates","creates","produces","output","input","feature",
    "layer","algorithm","system","artificial","intelligence","compute","pattern",
    "result","accuracy","performance","function","method","research","study",
    "experiment","analysis","classification","detection","generation",
    "transformer","encoder","decoder","attention","weight","gradient","loss",
    "epoch","batch","sample","random","noise","vector","embedding","dimension",
    "hidden","sigmoid","relu","softmax","optimizer","dropout","convolutional",
    "recurrent","generative","adversarial","discriminator","and","or","to","of",
    "in","on","by","for","from","with","this","that","which","its","new","large",
    "small","high","low","better","best","good","great","important","different",
    "complex","simple","efficient","powerful","fast","accurate","robust",
    "flexible","scalable","modern","based","using","trained","improved",
    "enhanced","applied","developed","proposed","demonstrated","evaluated",
    "achieved","shows","learns","predicts","classifies","detects","maps",
    "processes","extracts","builds","trains","optimizes","reduces","increases",
]
WORDS = BASE_WORDS[:]
while len(WORDS) < VOCAB_SIZE: WORDS.append(f"w{len(WORDS)}")
W2I = {w: i for i, w in enumerate(WORDS)}
I2W = {i: w for i, w in enumerate(WORDS)}
 
# ──────────────────────────────────────────────────────────
#  TRAINING STATE
# ──────────────────────────────────────────────────────────
state = {
    "image": {"running": False, "epoch": 0, "total": 0, "d_loss": [], "g_loss": [], "status": "idle"},
    "text":  {"running": False, "epoch": 0, "total": 0, "d_loss": [], "g_loss": [], "status": "idle"},
}
 
# ──────────────────────────────────────────────────────────
#  IMAGE TRAINING
# ──────────────────────────────────────────────────────────
def run_image_training(epochs):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    s = state["image"]
    s.update(running=True, epoch=0, total=epochs, d_loss=[], g_loss=[], status="Downloading MNIST (~11 MB)...")
    try:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
        G  = ImageGenerator().to(DEVICE)
        D  = ImageDiscriminator().to(DEVICE)
        bce = nn.BCELoss()
        oG  = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
        oD  = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
        s["status"] = "Training..."
        for ep in range(1, epochs + 1):
            if not s["running"]: break
            ed = eg = 0.0
            for real, _ in dl:
                if not s["running"]: break
                real = real.to(DEVICE); bs = real.size(0)
                rl = torch.ones(bs,1,device=DEVICE)*0.9
                fl = torch.zeros(bs,1,device=DEVICE)
                oD.zero_grad()
                z  = torch.randn(bs, IMG_LATENT, device=DEVICE)
                ld = (bce(D(real), rl) + bce(D(G(z).detach()), fl)) / 2
                ld.backward(); oD.step()
                oG.zero_grad()
                z  = torch.randn(bs, IMG_LATENT, device=DEVICE)
                lg = bce(D(G(z)), torch.ones(bs,1,device=DEVICE))
                lg.backward(); oG.step()
                ed += ld.item(); eg += lg.item()
            s["epoch"] = ep
            n = len(dl)
            s["d_loss"].append(round(ed/n, 4))
            s["g_loss"].append(round(eg/n, 4))
            s["status"] = f"Epoch {ep}/{epochs} — D:{round(ed/n,3)} G:{round(eg/n,3)}"
        torch.save(G.state_dict(), os.path.join(MODEL_DIR, "image_G.pth"))
        torch.save(D.state_dict(), os.path.join(MODEL_DIR, "image_D.pth"))
        s.update(running=False, status="Done ✓ Model saved")
    except Exception as e:
        s.update(running=False, status=f"Error: {str(e)}")
 
# ──────────────────────────────────────────────────────────
#  TEXT TRAINING
# ──────────────────────────────────────────────────────────
TEMPLATES = [
    "the neural network learns patterns from data efficiently",
    "machine learning model generates new text output samples",
    "deep learning has improved natural language processing systems",
    "the generative model creates realistic image and text samples",
    "artificial intelligence systems can classify and detect patterns",
    "transformer attention mechanism is powerful and flexible for learning",
    "training the model requires large data and significant computation",
    "the discriminator network learns to identify real from fake samples",
    "gradient descent optimizer updates model weights to reduce loss",
    "convolutional neural network extracts important features from images",
    "the encoder maps input data to a compact hidden embedding vector",
    "deep generative adversarial models produce high quality output",
    "a large language model predicts the next word in a sequence",
    "recurrent networks process sequential data using hidden state",
    "the generator and discriminator are trained in an adversarial game",
    "batch normalization stabilizes training of deep neural networks",
    "dropout is a regularization technique that reduces overfitting",
    "the loss function measures how far predictions are from targets",
    "backpropagation computes gradients by applying the chain rule",
    "attention lets a model focus on the most relevant input tokens",
    "the softmax function converts logits into a probability distribution",
    "reinforcement learning trains an agent using reward signals",
    "the embedding layer maps discrete tokens to continuous vectors",
    "supervised learning uses labeled examples to train a model",
    "unsupervised learning finds structure in unlabeled data",
    "the vanishing gradient problem can slow down deep network training",
    "a good model generalizes well to data it has not seen before",
    "hyperparameter tuning improves model performance on validation data",
    "the discriminator tries to distinguish real samples from generated ones",
    "the generator learns to produce samples that fool the discriminator",
]
# NOTE: this is a small, hand-written template set (~30 sentences, ~100
# distinct words) meant to demonstrate the Gumbel-Softmax + straight-through
# training technique, not to produce fluent open-domain text. Swap this for
# a real corpus + tokenizer (and a much larger model) for actual text
# quality — see README "Known limitations".
 
def make_real_batch(bs):
    batch = []
    for _ in range(bs):
        tpl  = TEMPLATES[np.random.randint(len(TEMPLATES))].split()
        ids  = [W2I.get(w, 1) for w in tpl]
        ids  = ids[:SEQ_LEN] + [0] * max(0, SEQ_LEN - len(ids))
        batch.append(ids)
    return torch.tensor(batch, dtype=torch.long, device=DEVICE)
 
def run_text_training(epochs):
    s = state["text"]
    s.update(running=True, epoch=0, total=epochs, d_loss=[], g_loss=[], status="Initialising...")
    try:
        G   = TextGenerator().to(DEVICE)
        D   = TextDiscriminator().to(DEVICE)
        bce = nn.BCELoss()
        oG  = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))
        oD  = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))
        STEPS = 40; BS = 32
        s["status"] = "Training..."
        for ep in range(1, epochs + 1):
            if not s["running"]: break
            ed = eg = 0.0
            for _ in range(STEPS):
                if not s["running"]: break
                rl = torch.ones(BS,1,device=DEVICE)
                fl = torch.zeros(BS,1,device=DEVICE)
                oD.zero_grad()
                real = make_real_batch(BS)
                z    = torch.randn(BS, TXT_LATENT, device=DEVICE)
                with torch.no_grad(): fake = G.sample_gumbel(z, tau=1.0, hard=True)
                ld = (bce(D(real), rl) + bce(D(fake), fl)) / 2
                ld.backward(); oD.step()
                oG.zero_grad()
                z    = torch.randn(BS, TXT_LATENT, device=DEVICE)
                fake = G.sample_gumbel(z, tau=1.0, hard=True)  # NOT detached — gradients must flow to G
                lg = bce(D(fake), torch.ones(BS,1,device=DEVICE))
                lg.backward(); oG.step()
                ed += ld.item(); eg += lg.item()
            s["epoch"] = ep
            s["d_loss"].append(round(ed/STEPS, 4))
            s["g_loss"].append(round(eg/STEPS, 4))
            s["status"] = f"Epoch {ep}/{epochs} — D:{round(ed/STEPS,3)} G:{round(eg/STEPS,3)}"
        torch.save(G.state_dict(), os.path.join(MODEL_DIR, "text_G.pth"))
        torch.save(D.state_dict(), os.path.join(MODEL_DIR, "text_D.pth"))
        s.update(running=False, status="Done ✓ Model saved")
    except Exception as e:
        s.update(running=False, status=f"Error: {str(e)}")
 
# ──────────────────────────────────────────────────────────
#  FRONTEND (served by this same Flask app/process)
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.errorhandler(404)
def not_found(e):
    # Keep API 404s as JSON; let everything else fall back to the SPA shell
    # so client-side routes (if any are added later) don't hard-404.
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found", "path": request.path}), 404
    return send_from_directory(FRONTEND_DIR, "index.html")

# ──────────────────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "auth_required": bool(API_KEY)})
 
@app.route("/api/models")
def models_status():
    files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
    return jsonify({"image_trained": "image_G.pth" in files,
                    "text_trained":  "text_G.pth"  in files})
 
# Image endpoints
@app.route("/api/image/train", methods=["POST"])
@require_api_key
def image_train():
    if state["image"]["running"]:
        return jsonify({"error": "Already training"}), 400
    ep = int((request.json or {}).get("epochs", 30))
    threading.Thread(target=run_image_training, args=(ep,), daemon=True).start()
    return jsonify({"message": f"Image GAN started — {ep} epochs"})
 
@app.route("/api/image/stop", methods=["POST"])
@require_api_key
def image_stop():
    state["image"]["running"] = False
    return jsonify({"message": "Stopped"})
 
@app.route("/api/image/status")
def image_status():
    return jsonify(state["image"])
 
@app.route("/api/image/generate")
def image_generate():
    n = int(request.args.get("n", 16))
    G = ImageGenerator().to(DEVICE)
    p = os.path.join(MODEL_DIR, "image_G.pth")
    trained = os.path.exists(p)
    if trained: G.load_state_dict(torch.load(p, map_location=DEVICE))
    G.eval()
    with torch.no_grad():
        imgs = (G(torch.randn(n, IMG_LATENT, device=DEVICE)).cpu() + 1) / 2
    grid = vutils.make_grid(imgs, nrow=4, padding=2)
    arr  = (grid.permute(1,2,0).numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1: arr = arr[:,:,0]
    mode = "L" if arr.ndim == 2 else "RGB"
    buf  = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    b64  = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"image": f"data:image/png;base64,{b64}", "trained": trained})
 
# Text endpoints
@app.route("/api/text/train", methods=["POST"])
@require_api_key
def text_train():
    if state["text"]["running"]:
        return jsonify({"error": "Already training"}), 400
    ep = int((request.json or {}).get("epochs", 60))
    threading.Thread(target=run_text_training, args=(ep,), daemon=True).start()
    return jsonify({"message": f"Text GAN started — {ep} epochs"})
 
@app.route("/api/text/stop", methods=["POST"])
@require_api_key
def text_stop():
    state["text"]["running"] = False
    return jsonify({"message": "Stopped"})
 
@app.route("/api/text/status")
def text_status():
    return jsonify(state["text"])
 
@app.route("/api/text/generate")
def text_generate():
    n    = int(request.args.get("n", 8))
    temp = float(request.args.get("temperature", 1.0))
    G    = TextGenerator().to(DEVICE)
    p    = os.path.join(MODEL_DIR, "text_G.pth")
    trained = os.path.exists(p)
    if trained: G.load_state_dict(torch.load(p, map_location=DEVICE))
    G.eval()
    sentences = []
    with torch.no_grad():
        logits = G(torch.randn(n, TXT_LATENT, device=DEVICE)) / max(temp, 0.1)  # scale logits, not probs
        probs  = torch.softmax(logits, dim=-1)
        ids    = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(n, SEQ_LEN)
    for row in ids:
        ws = [I2W.get(i.item(), "") for i in row]
        ws = [w for w in ws if w and w not in ("<pad>","<unk>") and not w.startswith("w")]
        sentences.append(" ".join(ws) if ws else "(train first for real output)")
    return jsonify({"sentences": sentences, "trained": trained})
 
if __name__ == "__main__":
    PORT  = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n  [OK]  GAN Studio running -> http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG, threaded=True)