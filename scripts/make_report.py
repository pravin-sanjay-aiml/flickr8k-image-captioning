"""Sample grid (real test images + predictions + a reference) and training curve from results/."""
import json, os, sys, textwrap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import load_flickr8k
data = sys.argv[1]; caps, split, imgdir = load_flickr8k(data); P = json.load(open("results/test_predictions.json")); sc = json.load(open("results/test_scores.json"))
dk = [k for k in P if k.startswith("Decoder (CLIP")][0]; test = split["test"]
from ml.metrics import score
from ml.metrics import simple_tok; from pycocoevalcap.cider.cider import Cider
R = {n: [simple_tok(c) for c in caps[n]] for n in test}; H = {n: [simple_tok(P[dk][n])] for n in test}; per = dict(zip(test, Cider().compute_score(R, H)[1]))   # per-image CIDEr, corpus-level idf
order = sorted(test, key=lambda n: -per[n]); picks = [("best", order[:4]), ("random", test[100:104]), ("worst", order[-4:])]
fig, ax = plt.subplots(3, 4, figsize=(16, 13))
for r, (tag, ns) in enumerate(picks):
    for c, n in enumerate(ns):
        a = ax[r, c]; a.imshow(Image.open(os.path.join(imgdir, n))); a.axis("off")
        a.set_title("\n".join(textwrap.wrap(f"[{tag}] {P[dk][n]}", 38)) + f"\nref: " + "\n".join(textwrap.wrap(caps[n][0], 38)), fontsize=8, loc="left")
plt.tight_layout(); plt.savefig("results/sample_captions.png", dpi=110)
h = json.load(open("results/history.json")); fig, a = plt.subplots(1, 2, figsize=(10, 3.5)); e = [x["epoch"] for x in h]
a[0].plot(e, [x["train_loss"] for x in h], label="train (label-smoothed)"); a[0].plot(e, [x["val_nll"] for x in h], label="val NLL"); a[0].legend(); a[0].set_xlabel("epoch")
a[1].plot(e, [x["val_cider_greedy"] for x in h]); a[1].set_title("dev CIDEr (greedy, 300 imgs)"); a[1].set_xlabel("epoch"); plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=110)
json.dump({"per_image_cider": per}, open("results/per_image_cider.json", "w"))
