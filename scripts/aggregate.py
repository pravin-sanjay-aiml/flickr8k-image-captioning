"""Mean±std over seeds for the scratch decoder + CLIP-text-pretrained decoder; writes results/summary.json"""
import json, os, numpy as np
cols = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]; K = "Decoder (CLIP tokens, beam=3)"; out = {}
runs = {"Scratch decoder (3 seeds)": ["results", "results_seed1", "results_seed2"]}
for name, dirs in runs.items():
    S = [json.load(open(f"{d}/test_scores.json"))[K] for d in dirs]; out[name] = {c: [float(np.mean([s[c] for s in S]) * 100), float(np.std([s[c] for s in S], ddof=1) * 100)] for c in cols}; out[name]["per_seed_CIDEr"] = [float(s["CIDEr"] * 100) for s in S]
if os.path.exists("results_cliptext/test_scores.json"):
    s = json.load(open("results_cliptext/test_scores.json"))[K]; out["CLIP-text-pretrained decoder (1 seed)"] = {c: [float(s[c] * 100), None] for c in cols}
json.dump(out, open("results_summary.json", "w"), indent=1)
for k, v in out.items(): print(k, {c: (round(x[0], 1), None if x[1] is None else round(x[1], 1)) for c, x in v.items() if c in cols})
