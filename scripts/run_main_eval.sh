#!/bin/bash
# Baseline vs SimCLR across label fractions. Run after checkpoints/full.pt exists
# (from: python train_simclr.py --name full --epochs 50 --batch_size 256, or download it from the README link).
# For each label fraction it records three accuracies in results/results.csv:
#   supervised (random init), SimCLR linear probe, SimCLR fine-tune.
# Same --frac and --seed give identical labelled images to every method, so the comparison is fair.

for frac in 1.0 0.2 0.1 0.01; do   # 100%, 20%, 10% and 1% of the 5,000 training labels
  # ResNet-18 trained from random weights on the labels only (no pretraining)
  python evaluate.py --mode scratch --frac $frac --tag supervised
  # Frozen SimCLR encoder + one linear layer
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
  # SimCLR encoder with every layer trained further on the labels
  python evaluate.py --mode finetune --ckpt checkpoints/full.pt --frac $frac --tag simclr_full
done
