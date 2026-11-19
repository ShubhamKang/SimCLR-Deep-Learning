#!/bin/bash
# Baseline vs SimCLR across label fractions (run after checkpoints/full.pt exists)
for frac in 1.0 0.2 0.1 0.01; do
  python evaluate.py --mode scratch --frac $frac --tag supervised
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
  python evaluate.py --mode finetune --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
done
