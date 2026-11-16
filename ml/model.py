"""Caption decoder over frozen CLIP image tokens (ClipCap/BLIP-style cross-attention, trained from scratch on Flickr8k captions)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .data import PAD, BOS, EOS


class Captioner(nn.Module):
    def __init__(self, vocab, d=384, nhead=6, layers=3, d_ff=1024, dropout=0.3, img_dim=768, n_img=50, max_len=26):
        super().__init__()
        self.d, self.max_len, self.V = d, max_len, vocab
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d), nn.LayerNorm(d)); self.img_pos = nn.Parameter(torch.zeros(1, n_img, d))
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD); self.pos = nn.Embedding(max_len + 2, d); self.drop = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(d, nhead, d_ff, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers); self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, vocab, bias=False); self.out.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, std=d ** -0.5); nn.init.normal_(self.img_pos, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)

    def encode_img(self, tokens): return self.drop(self.img_proj(tokens) + self.img_pos)

    def forward(self, tokens, caps_in):                          # tokens (B,50,768), caps_in (B,T) -> logits (B,T,V)
        mem = self.encode_img(tokens); T = caps_in.size(1)
        x = self.drop(self.tok(caps_in) * math.sqrt(self.d) + self.pos(torch.arange(T)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return self.out(self.norm(self.dec(x, mem, tgt_mask=mask, tgt_key_padding_mask=caps_in == PAD)))

    @torch.no_grad()
    def beam(self, tokens, beam=3, max_len=None, length_penalty=0.7):
        """Beam search for ONE image (tokens (50,768)); returns token ids of the best caption."""
        max_len = max_len or self.max_len; mem = self.encode_img(tokens[None]); seqs = [([BOS], 0.0)]; done = []
        for _ in range(max_len):
            cand = []
            for s, sc in seqs:
                ids = torch.tensor([s]); T = ids.size(1)
                x = self.tok(ids) * math.sqrt(self.d) + self.pos(torch.arange(T)); mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
                lp = F.log_softmax(self.out(self.norm(self.dec(x, mem, tgt_mask=mask)))[0, -1], -1); top = lp.topk(beam)
                for v, i in zip(top.values.tolist(), top.indices.tolist()): cand.append((s + [i], sc + v))
            cand.sort(key=lambda c: -c[1]); seqs = []
            for s, sc in cand[:beam]:
                (done if s[-1] == EOS else seqs).append((s, sc))
            if not seqs: break
        pool = done or seqs
        return max(pool, key=lambda c: c[1] / (len(c[0]) ** length_penalty))[0]

    @torch.no_grad()
    def greedy_batch(self, tokens, max_len=None):                # tokens (B,50,768) -> list of id lists (fast, used during training for dev scores)
        max_len = max_len or self.max_len; B = tokens.size(0); mem = self.encode_img(tokens); ids = torch.full((B, 1), BOS); fin = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_len):
            T = ids.size(1); x = self.tok(ids) * math.sqrt(self.d) + self.pos(torch.arange(T)); mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            nxt = self.out(self.norm(self.dec(x, mem, tgt_mask=mask)))[:, -1].argmax(-1); nxt[fin] = PAD; ids = torch.cat([ids, nxt[:, None]], 1); fin |= nxt == EOS
            if fin.all(): break
        return ids[:, 1:].tolist()
