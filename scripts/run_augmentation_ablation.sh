#!/bin/bash
# Short-schedule ablation: full set, leave-one-out, and single augmentations.
# Same schedule for every run (EPOCHS, SUBSET). Evaluates each with a linear probe at 100% labels.
EPOCHS=${EPOCHS:-20}; SUBSET=${SUBSET:-20000}
run() { name=$1; shift
  python train_simclr.py --name abl_$name --augs "$@" --epochs $EPOCHS --subset $SUBSET --batch_size 256 &&
  python evaluate.py --mode linear --ckpt checkpoints/abl_$name.pt --frac 1.0 --tag abl_$name; }
run full crop flip color gray blur
run no_crop flip color gray blur
run no_flip crop color gray blur
run no_color crop flip gray blur
run no_gray crop flip color blur
run no_blur crop flip color gray
run only_crop crop
run only_color color
run crop_color crop color
run none
