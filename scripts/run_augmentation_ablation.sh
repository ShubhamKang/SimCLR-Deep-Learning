#!/bin/bash
# Augmentation ablation on one machine (all 10 settings in a row).
# Every setting uses the same short schedule (EPOCHS epochs on SUBSET unlabelled images, batch 256),
# so the only difference between runs is which augmentations are switched on.
# Each setting is evaluated with a linear probe on 100% of the labels; results go to results/results.csv.
# Change the schedule with e.g.:  EPOCHS=30 SUBSET=30000 bash scripts/run_augmentation_ablation.sh

# Use the values from the environment if given, otherwise 20 epochs and 20,000 images.
EPOCHS=${EPOCHS:-20}; SUBSET=${SUBSET:-20000}

# run NAME AUG1 AUG2 ...
#   1. pretrain SimCLR with only the listed augmentations on (checkpoint: checkpoints/abl_NAME.pt)
#   2. if pretraining succeeded (&&), linear-probe that checkpoint and log it under the tag abl_NAME
run() { name=$1; shift
  python train_simclr.py --name abl_$name --augs "$@" --epochs $EPOCHS --subset $SUBSET --batch_size 256 &&
  python evaluate.py --mode linear --ckpt checkpoints/abl_$name.pt --frac 1.0 --tag abl_$name; }

run full crop flip color gray blur   # all five augmentations (reference point)
# Leave-one-out: remove one augmentation at a time to see how much each one contributes.
run no_crop flip color gray blur
run no_flip crop color gray blur
run no_color crop flip gray blur
run no_gray crop flip color blur
run no_blur crop flip color gray
# Small sets: single augmentations and the crop + colour pair from the SimCLR paper.
run only_crop crop
run only_color color
run crop_color crop color
run none                              # no augmentation at all (images are only resized)
