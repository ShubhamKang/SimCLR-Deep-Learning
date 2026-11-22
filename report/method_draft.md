# Draft: Method section (Shubham). Edit numbers after the final runs.

## 3. Method

### 3.1 Contrastive pretraining (SimCLR)
We follow SimCLR (Chen et al., 2020). For each unlabelled image x we draw two random augmentations,
giving two views x_i and x_j. A ResNet-18 encoder f maps each view to a 512-d feature vector h, and a
two-layer MLP projection head g maps h to a 128-d vector z. Views of the same image are a positive pair;
the other 2(N-1) views in the batch of N images are negatives.

The loss for a positive pair (i, j) is the NT-Xent loss:

    L(i, j) = -log( exp(sim(z_i, z_j) / t) / sum over k != i of exp(sim(z_i, z_k) / t) )

where sim is cosine similarity and t = 0.5 is the temperature. The loss is averaged over all 2N views in the batch.
After pretraining the projection head is discarded and only the encoder f is used.

### 3.2 Augmentations
The five augmentations studied are random resized crop (scale 0.2 to 1.0), horizontal flip, colour jitter
(strength 0.5, applied with probability 0.8), random grayscale (p = 0.2) and Gaussian blur (kernel 9, p = 0.5).
Each can be switched on or off independently (src/augmentations.py), which is what the ablation uses.

### 3.3 Training details
ResNet-18 with random initialisation, 96x96 inputs, AdamW (lr 3e-4, weight decay 1e-6), cosine learning-rate
decay, mixed precision. Main model: 100,000 unlabelled STL-10 images, batch size 256, 50 epochs (about 3.1 hours on one Kaggle Tesla T4).
Ablations use a shorter, identical schedule: 20 epochs on a fixed 20,000-image subset (chosen so each ablation run takes about 5 to 15 minutes on Kaggle).

### 3.4 Evaluation
All accuracies are measured on the 8,000-image STL-10 test set, with one seed (seed 0). We compare three classifiers built on a
limited number of labels (100%, 20%, 10%, 1% of the 5,000 training labels, class-balanced, same images for every method):
1. Supervised baseline: ResNet-18 trained from scratch on the labels.
2. Linear probe: frozen SimCLR encoder with one trained linear layer.
3. Fine-tuning: SimCLR encoder with all layers trained at a lower learning rate.

Training settings (evaluate.py):
- Supervised baseline and fine-tuning: 100 epochs, batch size 128 (capped at the number of images for 1% labels),
  AdamW with weight decay 1e-4 and cosine decay; learning rate 1e-3 from scratch and 3e-4 for fine-tuning.
  Light augmentation for both: random crop with 8-pixel padding and horizontal flip.
- Linear probe: 512-d encoder features extracted once without augmentation, standardised with training-set
  statistics, then a single linear layer trained with full-batch Adam (lr 1e-3, weight decay 1e-4, 300 steps).
- Ablations are evaluated with the linear probe on 100% of the labels.

## Placeholders to fill from results
- Pretraining loss curve: results/logs/pretrain_full.csv
- Augmentation ablation: results/results.csv, tags abl_*
- Hardware used (GPU model is printed in the Kaggle log) and wall-clock time per run
