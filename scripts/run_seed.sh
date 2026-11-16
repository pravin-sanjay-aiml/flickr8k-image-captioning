#!/bin/bash
# usage: run_seed.sh SEED  (1 thread each so two seeds can run in parallel on 2 cores)
cd "$(dirname "$0")/.."; S=$1; D=${DATA:-data}; W=${CLIP_WEIGHTS:-vit_b_32_laion400m_e32.pt}
python -m ml.train --data $D --feats features --out results_seed$S --seed $S --epochs 14 --patience 4 --threads 1 > train_seed$S.log 2>&1
python scripts/evaluate.py --data $D --feats features --results results_seed$S --weights $W --only_decoder --threads 1 > eval_seed$S.log 2>&1
