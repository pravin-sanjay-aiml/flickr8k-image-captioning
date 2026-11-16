# Flickr8k Image Captioning — CLIP image encoder + cross-attention transformer decoder

```
flickr8k-captioning/
├── frontend/            index.html            # web UI (HTML/CSS/JS)
├── backend/             app.py                # FastAPI: /health /samples /caption/sample/{id} /caption/upload
├── ml/                                         # model code
│   ├── data.py          Flickr8k loading, vocab
│   ├── extract.py       frozen CLIP ViT-B/32 -> 50 image tokens (cached)
│   ├── model.py         cross-attention transformer decoder + beam search
│   ├── clip_text.py     variant initialised from pretrained CLIP text transformer
│   ├── train.py         training loop (label smoothing, early stopping)
│   └── metrics.py       BLEU / METEOR / ROUGE-L / CIDEr (pycocoevalcap)
├── scripts/             evaluate.py, extra_metrics.py (bootstrap CI + CLIPScore), aggregate.py, make_report.py, run_seed.sh
├── notebooks/           Flickr8k_Captioning_CLIP.ipynb   (executed, 0 errors)
├── model/               decoder_fp16.pt       # trained decoder used by the app (18 MB)
├── samples/             24 held-out test images + cached CLIP tokens + references
├── results*/            test_scores.json, extra_metrics.json, history.json, plots (seeds 0-2, CLIP-text variant)
├── results_summary.json, Flickr8k_Captioning_Score.xlsx
├── tests/               test_api.py (7 tests)
├── Dockerfile, requirements.txt
```

## Results (1,000 Flickr8k test images, 5 refs; pycocoevalcap, PTB tokenizer, Java METEOR; x100)
| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| **Scratch decoder, beam=3 — mean ± std, 3 seeds** | 70.0 ± 0.8 | 28.1 ± 0.9 | 24.1 ± 0.3 | 52.3 ± 0.6 | **73.8 ± 2.1** |
| CLIP-text-pretrained decoder (1 seed) | 68.8 | 27.4 | 24.4 | 51.8 | 73.0 |
| CLIP image -> best train caption (retrieval) | 53.4 | 12.5 | 19.5 | 40.3 | 42.2 |
| NN train image -> copy caption | 48.8 | 10.9 | 16.9 | 38.0 | 32.1 |
| Constant caption | 42.3 | 8.6 | 12.0 | 33.5 | 9.1 |

Bootstrap 95% CI (seed 0, CIDEr): 71–80. Reference-free CLIPScore: 0.72–0.73 (human reference 0.78; CLIP-retrieval 0.85 is inflated because it is chosen by CLIP similarity).

## Run the web app
```
pip install -r requirements.txt
python -m uvicorn backend.app:app --port 8001      # open http://localhost:8001  (docs: /docs)
```
Sample images work out of the box. "Upload your own" additionally needs the CLIP weights (`vit_b_32_laion400m_e32.pt`, 605 MB): `set CLIP_WEIGHTS=C:\path\vit_b_32_laion400m_e32.pt` before starting. Tests: `python -m pytest -q tests`.

## Reproduce training
```
python -m ml.extract --data DATA --weights vit_b_32_laion400m_e32.pt --out features     # ~5 min on 2 CPU cores
python -m ml.train --data DATA --feats features --out results --seed 0
python scripts/evaluate.py --data DATA --weights vit_b_32_laion400m_e32.pt
```
DATA = folder with `images/Flicker8k_Dataset/*.jpg` and `text/` (Flickr8k.token.txt + split files). Java is needed for PTB tokenizer/METEOR. `features/tokens.npy` (620 MB) is not in the repo; regenerate with the first command.

## Honest limitations
- **Not a fine-tuned BLIP/GIT.** Pretrained generative LM weights (BLIP/GPT-2) were unreachable in my environment, so the decoder language model is trained from scratch; the pretrained-text (CLIP text transformer) variant gave no gain.
- 3 seeds for the scratch decoder, 1 for the CLIP-text variant; dev selection on 300 dev images (noisy).
- Dataset: Flickr8k from the jbrownlee GitHub release, not Kaggle (8,091 usable images; one stray key dropped); official 6000/1000/1000 split.
- Reference-overlap metrics under-reward valid but different captions; no human evaluation. Failure cases: rugby called soccer, generic captions for crowds.
