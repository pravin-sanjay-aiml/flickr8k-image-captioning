"""FastAPI demo: Flickr8k image captioning (frozen CLIP ViT-B/32 image tokens + cross-attention transformer decoder).
Run: uvicorn backend.app:app --port 8001      Upload mode needs CLIP weights: set CLIP_WEIGHTS=/path/vit_b_32_laion400m_e32.pt"""
import io, json, os, sys, time
import numpy as np, torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ml.data import Vocab
from ml.model import Captioner

torch.set_num_threads(2)
app = FastAPI(title="Flickr8k Captioning (CLIP + transformer decoder)", version="1.0")
ck = torch.load(os.path.join(ROOT, "model", "decoder_fp16.pt"), weights_only=False)
VOCAB = Vocab(ck["vocab"][4:]); MODEL = Captioner(len(VOCAB)); MODEL.load_state_dict({k: v.float() for k, v in ck["model"].items()}); MODEL.eval()
META = json.load(open(os.path.join(ROOT, "samples", "meta.json"))); TOK = np.load(os.path.join(ROOT, "samples", "tokens.npy"))
_clip = {}


def caption_tokens(tokens, mode="beam", beam=3):
    x = torch.from_numpy(np.asarray(tokens, dtype=np.float32)); t = time.time()
    ids = MODEL.beam(x, beam=beam) if mode == "beam" else MODEL.greedy_batch(x[None])[0]
    return VOCAB.decode(ids[1:] if mode == "beam" else ids), round((time.time() - t) * 1000)


def _clip_model():
    w = os.environ.get("CLIP_WEIGHTS")
    if not w or not os.path.exists(w): return None
    if "m" not in _clip:
        from ml.extract import load_clip
        _clip["m"], _clip["pre"] = load_clip(w)
    return _clip


@app.get("/health")
def health(): return {"status": "ok", "vocab": len(VOCAB), "params_M": round(sum(p.numel() for p in MODEL.parameters()) / 1e6, 2), "upload_enabled": _clip_model() is not None}


@app.get("/samples")
def samples(): return [{"id": i, "image": f"/samples/{m['file']}", "refs": m["refs"]} for i, m in enumerate(META)]


@app.get("/caption/sample/{i}")
def caption_sample(i: int, mode: str = "beam", beam: int = 3):
    if not 0 <= i < len(META): raise HTTPException(404, "unknown sample")
    if mode not in ("beam", "greedy") or not 1 <= beam <= 6: raise HTTPException(422, "mode must be beam|greedy, beam 1..6")
    cap, ms = caption_tokens(TOK[i], mode, beam); return {"id": i, "caption": cap, "mode": mode, "latency_ms": ms, "references": META[i]["refs"]}


@app.post("/caption/upload")
async def caption_upload(file: UploadFile = File(...), mode: str = "beam", beam: int = 3):
    c = _clip_model()
    if c is None: return JSONResponse({"error": "Upload mode disabled: set CLIP_WEIGHTS=/path/to/vit_b_32_laion400m_e32.pt and restart (see README)."}, status_code=503)
    from PIL import Image
    from ml.extract import encode_tokens
    try: img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception: raise HTTPException(422, "not a valid image")
    with torch.no_grad(): tokens, _ = encode_tokens(c["m"], c["pre"](img)[None])
    cap, ms = caption_tokens(tokens[0].numpy(), mode, beam); return {"caption": cap, "mode": mode, "latency_ms": ms}


app.mount("/samples", StaticFiles(directory=os.path.join(ROOT, "samples")), name="samples")


@app.get("/")
def index(): return FileResponse(os.path.join(ROOT, "frontend", "index.html"))
