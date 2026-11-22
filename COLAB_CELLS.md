# Running your ablation part on Colab

1. Open colab.research.google.com, Runtime > Change runtime type > **T4 GPU**. Keep the tab open and the laptop awake.
2. Paste ONE of the cells below and run it (about 25 minutes per setting).
3. When it finishes, open the Files panel (left), download `SimCLR-Deep-Learning/results/ablation_<yourname>.csv`,
   and send it to Shubham (or upload it to the repo's `results/` folder on GitHub with Add file > Upload files).
   The file is saved after every setting, so you can download partial results at any time.

## Sandhya (augmentation ablation, 4 settings, about 1h45)
```
!git clone https://github.com/ShubhamKang/SimCLR-Deep-Learning.git
%cd SimCLR-Deep-Learning
!pip install -q -r requirements.txt
!python scripts/run_ablation.py --who sandhya --aug full no_crop no_color only_crop
```

## Nidhi (batch-size ablation, 3 settings, about 1h30)
```
!git clone https://github.com/ShubhamKang/SimCLR-Deep-Learning.git
%cd SimCLR-Deep-Learning
!pip install -q -r requirements.txt
!python scripts/run_ablation.py --who nidhi --bs 64 128 512
```
(The batch-size-256 point reuses Sandhya's `full` run, which has identical settings.)

## Shubham (remaining augmentation settings, after pretraining finishes)
```
!python scripts/run_ablation.py --who shubham --aug only_color crop_color no_gray no_blur no_flip none
```

## Public pretrained checkpoint (backup / comparison, about 1h on Colab)
Evaluates the published SimCLR ResNet-18 trained on STL-10 for 100 epochs by sthalles/SimCLR (MIT licence).
Credit it in the report as an external checkpoint; it is a comparison, not our own pretraining.
```
!git clone https://github.com/ShubhamKang/SimCLR-Deep-Learning.git
%cd SimCLR-Deep-Learning
!pip install -q -r requirements.txt gdown
!gdown 14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF -O public.zip && unzip -o public.zip -d public
!mkdir -p checkpoints && python scripts/convert_public_checkpoint.py public/checkpoint_0100.pth.tar checkpoints/public_sthalles.pt
!for f in 1.0 0.2 0.1 0.01; do python evaluate.py --mode linear --ckpt checkpoints/public_sthalles.pt --frac $f --tag public_sthalles --no_normalize; done
!for f in 0.1 0.01; do python evaluate.py --mode finetune --ckpt checkpoints/public_sthalles.pt --frac $f --tag public_sthalles --no_normalize; done
!cat results/results.csv
```
