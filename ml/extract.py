"""Frozen CLIP image encoder -> per-image token features (CLS + 49 patch tokens, 768-d) and the projected, normalised global embedding.
    python -m ml.extract --data DATA_DIR --weights vit_b_32_laion400m_e32.pt --out features"""
import os, sys, time, argparse
import numpy as np, torch
from PIL import Image
from .data import load_flickr8k


def load_clip(weights, name="ViT-B-32-quickgelu"):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=weights); m.eval(); return m, pre


@torch.no_grad()
def encode_tokens(m, x):
    """x (B,3,224,224) -> tokens (B,50,768) after the final LayerNorm, global (B,512) L2-normalised."""
    v = m.visual; box = {}
    h = v.transformer.register_forward_hook(lambda mod, i, o: box.__setitem__("o", o))
    g = m.encode_image(x); h.remove(); o = box["o"]
    if o.shape[0] != x.shape[0]: o = o.permute(1, 0, 2)                      # (N,B,D) -> (B,N,D) in case the transformer is not batch_first
    return v.ln_post(o), torch.nn.functional.normalize(g, dim=-1)


class Imgs(torch.utils.data.Dataset):
    def __init__(self, files, pre): self.files, self.pre = files, pre
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.pre(Image.open(self.files[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", default="features")
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
    caps, split, img_dir = load_flickr8k(a.data); names = sorted(caps)[: a.limit] if a.limit else sorted(caps)
    m, pre = load_clip(a.weights); dl = torch.utils.data.DataLoader(Imgs([os.path.join(img_dir, n) for n in names], pre), batch_size=a.bs, num_workers=1)
    T, G, t0 = [], [], time.time()
    for i, x in enumerate(dl):
        t, g = encode_tokens(m, x); T.append(t.half().numpy()); G.append(g.numpy())
        if i % 10 == 0: print(f"[extract] {min((i + 1) * a.bs, len(names))}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out, "tokens.npy"), np.concatenate(T)); np.save(os.path.join(a.out, "global.npy"), np.concatenate(G))
    with open(os.path.join(a.out, "names.txt"), "w") as f: f.write("\n".join(names))
    print("[extract] done", np.concatenate(T).shape)


if __name__ == "__main__": main()
