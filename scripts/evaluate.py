"""Evaluate on the 1000 test images: decoder (beam), NN-caption-retrieval baseline, CLIP text retrieval baseline, 'one fixed caption' floor.
python scripts/evaluate.py --data D --feats features --results results --weights W"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data import load_flickr8k, Vocab
from ml.model import Captioner
from ml.metrics import score
from ml.train import load_feats

ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--feats", default="features"); ap.add_argument("--results", default="results"); ap.add_argument("--weights"); ap.add_argument("--beam", type=int, default=3)
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--only_decoder", action="store_true"); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args(); torch.set_num_threads(a.threads)
caps, split, _ = load_flickr8k(a.data); idx, tok, G = load_feats(a.feats); test = split["test"][: a.n]; train = split["train"]; refs = {n: caps[n] for n in test}; out = {}; preds = {}

ck = torch.load(os.path.join(a.results, "decoder.pt"), weights_only=False)
if ck.get("arch") == "clip":
    from ml.clip_text import ClipVocab; init = torch.load("init_cliptext.pt", weights_only=False); m = init["model"]; v = ClipVocab(init["rows"]); m.load_state_dict(ck["model"])
else: v = Vocab(ck["vocab"][4:]); m = Captioner(len(v)); m.load_state_dict(ck["model"])
m.eval()
preds["Decoder (CLIP tokens, beam=%d)" % a.beam] = {n: v.decode(m.beam(torch.from_numpy(np.asarray(tok[idx[n]])).float(), beam=a.beam)) for n in test}
preds["Decoder (greedy)"] = {}
for i in range(0, len(test), 100):
    b = test[i:i + 100]
    for n, s in zip(b, m.greedy_batch(torch.from_numpy(np.stack([tok[idx[x]] for x in b])).float())): preds["Decoder (greedy)"][n] = v.decode(s)

if a.only_decoder:
    res = {k: score(refs, h, use_meteor=True, ptb=True) for k, h in preds.items()}; json.dump(res, open(os.path.join(a.results, 'test_scores.json'), 'w'), indent=1); json.dump(preds, open(os.path.join(a.results, 'test_predictions.json'), 'w')); print(res); sys.exit()
# baseline 1: nearest train image by CLIP image embedding -> copy one of its captions (first caption)
Gtr = G[[idx[n] for n in train]]; Gte = G[[idx[n] for n in test]]; nn = (Gte @ Gtr.T).argmax(1); preds["NN image -> copy caption"] = {n: caps[train[j]][0] for n, j in zip(test, nn)}

# baseline 2: CLIP text-encoder retrieval of the best-matching train caption for the test image (zero-shot style)
import open_clip
cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=a.weights); cm.eval(); tk = open_clip.get_tokenizer("ViT-B-32-quickgelu")
pool = [c for n in train for c in caps[n]]; E = []
with torch.no_grad():
    for i in range(0, len(pool), 256): e = cm.encode_text(tk(pool[i:i + 256])); E.append((e / e.norm(dim=-1, keepdim=True)).numpy())
E = np.concatenate(E); preds["CLIP image -> best train caption"] = {n: pool[j] for n, j in zip(test, (Gte @ E.T).argmax(1))}
# floor: most common caption style (constant)
preds["Constant caption"] = {n: "a man in a blue shirt is standing in front of a building" for n in test}

res = {}
for k, h in preds.items(): res[k] = score(refs, h, use_meteor=True, ptb=True); print(k, {a_: round(b_ * 100, 1) for a_, b_ in res[k].items()}, flush=True)
json.dump(res, open(os.path.join(a.results, "test_scores.json"), "w"), indent=1); json.dump({k: h for k, h in preds.items()}, open(os.path.join(a.results, "test_predictions.json"), "w"))
