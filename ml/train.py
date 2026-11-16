"""Train the caption decoder on cached CLIP tokens. python -m ml.train --data D --feats features --out results [--epochs 25]"""
import argparse, json, os, time, math, numpy as np, torch, torch.nn.functional as F
from .data import load_flickr8k, Vocab, clean, PAD, EOS
from .model import Captioner
from .metrics import score


def load_feats(path):
    names = open(os.path.join(path, "names.txt")).read().split("\n")
    return {n: i for i, n in enumerate(names)}, np.load(os.path.join(path, "tokens.npy")), np.load(os.path.join(path, "global.npy"))


def build_vocab(caps, train, min_freq=3):
    import collections
    cnt = collections.Counter(w for n in train for c in caps[n] for w in clean(c))
    return Vocab(sorted(w for w, k in cnt.items() if k >= min_freq))


def dev_score(model, vocab, caps, names, idx, tok, n=300):
    model.eval(); sel = names[:n]; hyps = {}
    for i in range(0, len(sel), 100):
        b = sel[i:i + 100]; ids = model.greedy_batch(torch.from_numpy(np.stack([tok[idx[x]] for x in b])).float())
        for x, s in zip(b, ids): hyps[x] = vocab.decode(s)
    model.train(); return score({x: caps[x] for x in sel}, hyps, use_meteor=False, ptb=False)["CIDEr"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--feats", default="features"); ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--arch", default="scratch"); ap.add_argument("--init", default="init_cliptext.pt"); ap.add_argument("--weights", default="vit_b_32_laion400m_e32.pt"); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--lr_pre", type=float, default=1e-4); a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
    caps, split, _ = load_flickr8k(a.data); idx, tok, _g = load_feats(a.feats); tok = np.array(tok)  # fp16 copy in RAM (~0.6GB); mmap pages dropped below
    if a.arch == "clip":
        from .clip_text import ClipVocab, ClipTextCaptioner; from .extract import load_clip
        init = torch.load(a.init, weights_only=False); model = init["model"]; vocab = ClipVocab(init["rows"]); del init   # pre-built (CLIP text weights + new cross-attn) so the 150M visual tower is never loaded
        groups = [dict(params=model.pretrained_params(), lr=a.lr_pre), dict(params=model.new_params(), lr=a.lr)]
    else:
        vocab = build_vocab(caps, split["train"]); model = Captioner(len(vocab)); groups = [dict(params=list(model.parameters()), lr=a.lr)]
    print("vocab", len(vocab)); pairs = [(n, vocab.encode(c)) for n in split["train"] for c in caps[n]]; print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(groups, weight_decay=0.05); steps = a.epochs * math.ceil(len(pairs) / a.bs); warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))); hist = []; best = (-1, 0); t0 = time.time()
    def batch(ps):
        L = max(len(p[1]) for p in ps); ids = torch.zeros(len(ps), L, dtype=torch.long)
        for i, p in enumerate(ps): ids[i, :len(p[1])] = torch.tensor(p[1])
        return torch.from_numpy(np.stack([tok[idx[p[0]]] for p in ps])).float(), ids
    for ep in range(a.epochs):
        perm = np.random.permutation(len(pairs)); tl = 0; nb = 0
        for i in range(0, len(pairs), a.bs):
            x, ids = batch([pairs[j] for j in perm[i:i + a.bs]]); logits = model(x, ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = 0; vn = 0
            for i in range(0, 1000, 100):
                pv = [(n, vocab.encode(caps[n][0])) for n in split["val"][i:i + 100]]; x, ids = batch(pv); lg = model(x, ids[:, :-1])
                vl += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), ignore_index=PAD).item(); vn += 1
        model.train(); cd = dev_score(model, vocab, caps, split["val"], idx, tok)
        hist.append(dict(epoch=ep + 1, train_loss=tl / nb, val_nll=vl / vn, val_cider_greedy=cd, sec=time.time() - t0)); print(hist[-1], flush=True)
        if cd > best[0]: best = (cd, ep + 1); torch.save({"model": model.state_dict(), "vocab": getattr(vocab, "itos", None) or vocab.rows, "arch": a.arch, "layers": a.layers}, os.path.join(a.out, "decoder.pt"))
        json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        if ep + 1 - best[1] >= a.patience: print("early stop"); break
    print("best dev CIDEr(greedy, 300 imgs)", best)

if __name__ == "__main__": main()
