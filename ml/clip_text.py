"""Decoder initialised from the PRETRAINED CLIP text transformer (first L layers, token+position embeddings), with new zero-initialised
cross-attention layers to the CLIP image tokens. This gives the language side a pretrained start (CLIP text encoder, causal mask), unlike the scratch decoder."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .data import PAD, BOS, EOS, UNK, clean


class ClipVocab:
    """BPE tokens (CLIP tokenizer) restricted to those seen in train captions; ids 0..3 are PAD,BOS(=SOT),EOS(=EOT),UNK."""
    def __init__(self, rows):
        from open_clip.tokenizer import SimpleTokenizer
        self.tk = SimpleTokenizer(); self.rows = rows; self.pos = {r: i for i, r in enumerate(rows)}
    def __len__(self): return len(self.rows)
    @classmethod
    def build(cls, caps, train, min_freq=2):
        import collections; from open_clip.tokenizer import SimpleTokenizer
        tk = SimpleTokenizer(); cnt = collections.Counter(t for n in train for c in caps[n] for t in tk.encode(" ".join(clean(c))))
        return cls([0, 49406, 49407, 0] + sorted(t for t, k in cnt.items() if k >= min_freq))
    def encode(self, text, max_len=30):
        ids = [self.pos.get(t, UNK) for t in self.tk.encode(" ".join(clean(text)))][:max_len]; return [BOS] + ids + [EOS]
    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS: break
            if i > UNK: out.append(self.rows[i])
        return self.tk.decode(out).strip()


class Block(nn.Module):
    def __init__(self, clip_block, d, nhead, dropout):
        super().__init__(); self.ln1 = clip_block.ln_1; self.ln2 = clip_block.ln_2; self.mlp = nn.Sequential(clip_block.mlp.c_fc, nn.GELU(), clip_block.mlp.c_proj)
        self.sa = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        with torch.no_grad():
            self.sa.in_proj_weight.copy_(clip_block.attn.in_proj_weight); self.sa.in_proj_bias.copy_(clip_block.attn.in_proj_bias)
            self.sa.out_proj.weight.copy_(clip_block.attn.out_proj.weight); self.sa.out_proj.bias.copy_(clip_block.attn.out_proj.bias)
        self.lnc = nn.LayerNorm(d); self.ca = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        nn.init.zeros_(self.ca.out_proj.weight); nn.init.zeros_(self.ca.out_proj.bias)       # starts as the pure pretrained text model
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mem, mask, kpm=None):
        h = self.ln1(x); x = x + self.drop(self.sa(h, h, h, attn_mask=mask, key_padding_mask=kpm, need_weights=False)[0])
        h = self.lnc(x); x = x + self.drop(self.ca(h, mem, mem, need_weights=False)[0])
        return x + self.drop(self.mlp(self.ln2(x)))


class ClipTextCaptioner(nn.Module):
    def __init__(self, clip_model, vocab: ClipVocab, layers=6, dropout=0.1, img_dim=768, n_img=50, max_len=32):
        super().__init__(); d = clip_model.transformer.width; self.d, self.max_len = d, max_len
        self.emb = nn.Embedding(len(vocab), d); self.emb.weight.data.copy_(clip_model.token_embedding.weight.data[torch.tensor(vocab.rows)])
        self.pos = nn.Parameter(clip_model.positional_embedding.data[: max_len + 2].clone())
        self.blocks = nn.ModuleList([Block(clip_model.transformer.resblocks[i], d, 8, dropout) for i in range(layers)]); self.ln_f = clip_model.ln_final
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d), nn.LayerNorm(d)); self.img_pos = nn.Parameter(torch.zeros(1, n_img, d)); nn.init.normal_(self.img_pos, std=0.02); self.drop = nn.Dropout(dropout)
    def pretrained_params(self): return [p for n, p in self.named_parameters() if n.startswith(("emb", "pos", "ln_f")) or (n.startswith("blocks") and not any(k in n for k in ("lnc", ".ca.")))]
    def new_params(self): ids = {id(p) for p in self.pretrained_params()}; return [p for p in self.parameters() if id(p) not in ids]
    def encode_img(self, t): return self.drop(self.img_proj(t) + self.img_pos)
    def _run(self, mem, ids, kpm=None):
        T = ids.size(1); x = self.drop(self.emb(ids) + self.pos[:T]); mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        for b in self.blocks: x = b(x, mem, mask, kpm)
        return F.linear(self.ln_f(x), self.emb.weight)
    def forward(self, tokens, caps_in): return self._run(self.encode_img(tokens), caps_in)  # right-padding + causal mask => no pad mask needed
    @torch.no_grad()
    def beam(self, tokens, beam=3, max_len=None, length_penalty=0.7):
        max_len = max_len or self.max_len; mem = self.encode_img(tokens[None]); seqs = [([BOS], 0.0)]; done = []
        for _ in range(max_len):
            cand = []
            for s, sc in seqs:
                lp = F.log_softmax(self._run(mem, torch.tensor([s]))[0, -1], -1); top = lp.topk(beam)
                for v, i in zip(top.values.tolist(), top.indices.tolist()): cand.append((s + [i], sc + v))
            cand.sort(key=lambda c: -c[1]); seqs = []
            for s, sc in cand[:beam]: (done if s[-1] == EOS else seqs).append((s, sc))
            if not seqs: break
        return max(done or seqs, key=lambda c: c[1] / (len(c[0]) ** length_penalty))[0]
    @torch.no_grad()
    def greedy_batch(self, tokens, max_len=None):
        max_len = max_len or self.max_len; B = tokens.size(0); mem = self.encode_img(tokens); ids = torch.full((B, 1), BOS); fin = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_len):
            nxt = self._run(mem, ids)[:, -1].argmax(-1); nxt[fin] = PAD; ids = torch.cat([ids, nxt[:, None]], 1); fin |= nxt == EOS
            if fin.all(): break
        return ids[:, 1:].tolist()
