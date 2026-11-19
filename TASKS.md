# Who does what

Read your section, work on your own branch (`git checkout -b yourname`), commit often, and open a pull request to `main`.
Setup for everyone: `pip install -r requirements.txt`. Use Colab or Kaggle with a GPU. STL-10 downloads automatically.
Keep results in `results/results.csv` (the scripts append to it) and commit it, so we can all see the numbers.

## Shubham (about 60%)
- Smoke test the code, then run main pretraining: `python train_simclr.py --name full --epochs 100 --batch_size 256`
- Run `scripts/run_augmentation_ablation.sh` (the core research question)
- Report: abstract, method, augmentation ablation results, discussion, conclusion. Final merge and PDF check.
- Share `checkpoints/full.pt` with Sandhya (Drive link) once pretraining finishes.

## Sandhya (about 20%): baseline and evaluation
Needs no checkpoint:
- Run the supervised baseline: `python evaluate.py --mode scratch --frac 1.0 --tag supervised`, and again for `--frac 0.2`, `0.1`, `0.01`.
After receiving `full.pt` into `checkpoints/`:
- Run `bash scripts/run_main_eval.sh` (linear probe and fine-tune at each label fraction).
- If time allows, repeat `--frac 0.1` with `--seed 1` and `--seed 2` and report mean and spread.
Report sections to write (in `report/`):
- Experimental setup: dataset (splits, sizes), label fractions, model, training settings, metrics.
- Results: supervised vs SimCLR linear probe vs fine-tune, with a table and the labels-vs-accuracy figure. A short analysis of how much pretraining helps at low labels.

## Nidhi (about 20%): batch size, figures, documentation
Report and setup:
- Write the report introduction (motivation, label scarcity) and related work (SimCLR, contrastive learning, supervised baselines). Cite Chen et al., ICML 2020.
- Check the README reproduces cleanly from a fresh clone.
Experiments:
- Run `bash scripts/run_batchsize_ablation.sh` (batch sizes 64, 128, 256, 512). Note GPU memory and time per run.
Figures and write-up (after all results exist):
- Run `python make_plots.py`. Check the three figures are readable (axis labels, units).
- Write the batch-size analysis section and the references list.
- Fill the Results table in the README.

## Everyone
- Add your name and tasks to the contribution statement at the end of the report.
- Review each other's report sections before submission.
- If a run fails or time is short, tell the group early. Cut order: batch size 512, then `only_crop` and `only_color` runs.
