"""Bootstrap 95% CIs over test images (CIDEr, BLEU-4 proxy via per-image CIDEr/CLIPScore) + reference-free CLIPScore using the CLIP image/text towers.
python scripts/extra_metrics.py --results RES [--preds KEY] -> RES/extra_metrics.json"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import load_flickr8k
from ml.train import load_feats
from ml.metrics import score, simple_tok
from pycocoevalcap.cider.cider import Cider
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--feats", default="features"); ap.add_argument("--results", nargs="+")
ap.add_argument("--weights", default="vit_b_32_laion400m_e32.pt"); ap.add_argument("--B", type=int, default=1000); a = ap.parse_args(); torch.set_num_threads(1)
import open_clip
caps, split, _ = load_flickr8k(a.data); idx, tok, G = load_feats(a.feats); test = split["test"]; cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=a.weights); cm.eval(); tk = open_clip.get_tokenizer("ViT-B-32-quickgelu")
Gt = G[[idx[n] for n in test]]; rng = np.random.default_rng(0); boot = rng.integers(0, len(test), (a.B, len(test)))
def clipscore(texts):   # CLIP-S (Hessel et al. 2021): 2.5 * max(cos, 0), reference-free
    E = []
    with torch.no_grad():
        for i in range(0, len(texts), 128): e = cm.encode_text(tk(texts[i:i + 128])); E.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return 2.5 * np.maximum((Gt * np.concatenate(E)).sum(1), 0)
def ci(x): x = np.asarray(x); b = x[boot].mean(1); return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for r in a.results:
    P = json.load(open(os.path.join(r, "test_predictions.json"))); out = {}
    for k, h in P.items():
        cs = clipscore([h[n] for n in test]); R = {n: [simple_tok(c) for c in caps[n]] for n in test}; H = {n: [simple_tok(h[n])] for n in test}
        per = np.array(Cider().compute_score(R, H)[1])   # per-image CIDEr with corpus-level idf (test order = dict order)
        out[k] = {"CLIPScore": ci(cs), "CIDEr_per_image_mean": ci(per)}
    refs = clipscore([caps[n][0] for n in test]); out["Human reference #1 (CLIPScore only)"] = {"CLIPScore": ci(refs)}
    json.dump(out, open(os.path.join(r, "extra_metrics.json"), "w"), indent=1); print(r, {k: [round(x, 3) for x in v["CLIPScore"]] for k, v in out.items()})
