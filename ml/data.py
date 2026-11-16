"""Flickr8k: image list, 5 captions per image, official train/dev/test split (6000/1000/1000), word vocabulary."""
import os, re, collections
import numpy as np

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def clean(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def load_flickr8k(data_dir):
    """data_dir contains text/ (Flickr8k.token.txt, Flickr_8k.{train,dev,test}Images.txt) and images/Flicker8k_Dataset/*.jpg"""
    caps = collections.defaultdict(list)
    for line in open(os.path.join(data_dir, "text", "Flickr8k.token.txt"), encoding="utf-8"):
        k, c = line.rstrip("\n").split("\t"); caps[k.split("#")[0]].append(c.strip())
    split = {}
    for name, f in (("train", "trainImages"), ("val", "devImages"), ("test", "testImages")):
        split[name] = [l.strip() for l in open(os.path.join(data_dir, "text", f"Flickr_8k.{f}.txt")) if l.strip()]
    img_dir = os.path.join(data_dir, "images", "Flicker8k_Dataset")
    # the token file has stray keys (e.g. "..._586949ec62.jpg.1"): keep only real images with 5 captions
    caps = {k: v for k, v in caps.items() if os.path.exists(os.path.join(img_dir, k)) and len(v) == 5}
    return caps, split, img_dir


class Vocab:
    def __init__(self, words): self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + words; self.stoi = {w: i for i, w in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def encode(self, text, max_len=22): return [BOS] + [self.stoi.get(w, UNK) for w in clean(text)][:max_len] + [EOS]
    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS: break
            if i not in (PAD, BOS): out.append(self.itos[i])
        return " ".join(out)

    @classmethod
    def build(cls, captions, min_freq=3):
        c = collections.Counter(w for s in captions for w in clean(s))
        return cls(sorted(w for w, n in c.items() if n >= min_freq))
