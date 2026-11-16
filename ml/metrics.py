"""BLEU-1..4, METEOR, ROUGE-L, CIDEr via pycocoevalcap (PTB tokenizer needs Java), with a plain fallback tokenizer."""
import re
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def simple_tok(s): return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def score(refs, hyps, use_meteor=True, ptb=True):
    """refs {id: [str,...]}, hyps {id: str} -> dict of metrics (x100 not applied)."""
    ids = sorted(hyps)
    if ptb:
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            tk = PTBTokenizer(); R = tk.tokenize({i: [{"caption": c} for c in refs[i]] for i in ids}); H = tk.tokenize({i: [{"caption": hyps[i]}] for i in ids})
        except Exception as e:
            print("[metrics] PTB tokenizer unavailable, using simple tokenizer:", type(e).__name__); R = {i: [simple_tok(c) for c in refs[i]] for i in ids}; H = {i: [simple_tok(hyps[i])] for i in ids}
    else: R = {i: [simple_tok(c) for c in refs[i]] for i in ids}; H = {i: [simple_tok(hyps[i])] for i in ids}
    out = {}; b, _ = Bleu(4).compute_score(R, H, verbose=0); out.update({f"BLEU-{k + 1}": b[k] for k in range(4)})
    out["ROUGE-L"], _ = Rouge().compute_score(R, H); out["CIDEr"], _ = Cider().compute_score(R, H)
    if use_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            out["METEOR"], _ = Meteor().compute_score(R, H)
        except Exception as e: print("[metrics] METEOR unavailable:", type(e).__name__)
    return out
