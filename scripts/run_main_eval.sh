#!/bin/bash
# Runs after checkpoints/full.pt exists. For each label fraction it records three accuracies:
#   supervised (random init), SimCLR linear probe, SimCLR fine-tune.
# Same --frac and --seed give identical labelled images to every method, so the comparison is fair.
# Baseline vs SimCLR across label fractions (run after checkpoints/full.pt exists)
for frac in 1.0 0.2 0.1 0.01; do
  python evaluate.py --mode scratch --frac $frac --tag supervised
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
  python evaluate.py --mode finetune --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
done
