#!/bin/bash
EPOCHS=${EPOCHS:-20}; SUBSET=${SUBSET:-20000}
for bs in 64 128 256 512; do
  python train_simclr.py --name bs_$bs --epochs $EPOCHS --subset $SUBSET --batch_size $bs &&
  python evaluate.py --mode linear --ckpt checkpoints/bs_$bs.pt --frac 1.0 --tag bs_$bs
done
