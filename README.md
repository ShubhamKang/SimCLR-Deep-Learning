# Self-supervised contrastive pretraining when labels are scarce (SimCLR on STL-10)

Deep Learning project. Team: Shubham Kang, Sandhya Parmar, Nidhi.

Question: which SimCLR augmentations matter most on a small compute budget, and how much does
pretraining on 100k unlabelled images help a classifier trained on few labels?

See [TASKS.md](TASKS.md) for who does what.

## Setup
```
pip install -r requirements.txt
```
STL-10 downloads automatically into `data/`.

## Reproduce
```
python train_simclr.py --name full --epochs 100 --batch_size 256   # main pretraining
bash scripts/run_main_eval.sh                                      # supervised vs linear probe vs fine-tune
bash scripts/run_augmentation_ablation.sh                          # augmentation ablation
bash scripts/run_batchsize_ablation.sh                             # batch-size ablation
python make_plots.py                                               # figures in results/
```
All results append to `results/results.csv`; pretraining loss logs go to `results/logs/`.

## Layout
- `src/` augmentations (on/off flags), data and label subsets, ResNet-18 + projection head, NT-Xent loss
- `train_simclr.py` pretraining; `evaluate.py` scratch / linear probe / fine-tune
- `scripts/` experiment runners; `report/` final report

## Results
(to be filled in)

## Contributions
- Shubham Kang: SimCLR implementation, pretraining, augmentation ablation, analysis
- Sandhya Parmar: supervised baseline, linear probe and fine-tune evaluation
- Nidhi: batch-size ablation, plots, documentation

## Reference
Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020. https://arxiv.org/abs/2002.05709
