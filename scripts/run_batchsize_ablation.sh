#!/bin/bash
# Batch-size ablation: pretrain with batch sizes 64/128/256/512 on the same short schedule
# (same epochs, same unlabelled subset, same augmentations), then linear-probe each at 100% labels.
# Larger batches give more negatives per image in the contrastive loss but need more GPU memory,
# and with a fixed number of epochs they also mean fewer optimisation steps.
# Override the schedule with e.g. EPOCHS=30 SUBSET=30000 bash scripts/run_batchsize_ablation.sh

# Use the values from the environment if given, otherwise 20 epochs and 20,000 images.
EPOCHS=${EPOCHS:-20}; SUBSET=${SUBSET:-20000}

for bs in 64 128 256 512; do
  # 1. pretrain with all augmentations at this batch size (checkpoint: checkpoints/bs_<bs>.pt)
  python train_simclr.py --name bs_$bs --epochs $EPOCHS --subset $SUBSET --batch_size $bs &&
  # 2. only if pretraining succeeded: linear probe, logged in results/results.csv as tag bs_<bs>
  python evaluate.py --mode linear --ckpt checkpoints/bs_$bs.pt --frac 1.0 --tag bs_$bs
done
