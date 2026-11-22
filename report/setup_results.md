# Experimental Setup and Results (Sandhya)

## 4. Experimental Setup

### 4.1 Dataset
We use STL-10 (Coates et al., 2011), which has 96x96 colour images in 10 classes (airplane, bird, car, cat, deer,
dog, horse, monkey, ship, truck). It has three parts:
- **Unlabelled split:** 100,000 images, used only for SimCLR pretraining. No labels are used at this stage.
- **Train split:** 5,000 labelled images (500 per class), used to train the classifiers.
- **Test split:** 8,000 labelled images (800 per class). Every accuracy in this report is measured on the full test set.

### 4.2 Simulating label scarcity
We train classifiers on 100%, 20%, 10% and 1% of the training labels, which is 500, 100, 50 and 5 images per class
(5,000, 1,000, 500 and 50 images in total). Each subset is class-balanced, so a small fraction cannot miss a class.
It is drawn with a fixed random seed (seed 0), so the supervised baseline, the linear probe and fine-tuning all see
exactly the same labelled images at a given fraction.

### 4.3 Model
All methods use the same ResNet-18 encoder with random initialisation (no ImageNet weights). The final
classification layer is removed, so the encoder outputs a 512-dimensional feature vector. A linear layer
(512 -> 10) on top of the encoder produces the class scores.

### 4.4 Methods compared
1. **Supervised baseline (no pretraining).** The whole network is trained from random weights on the labelled
   subset only. Settings: AdamW (learning rate 1e-3, weight decay 1e-4), 100 epochs, cosine learning-rate decay,
   batch size 128 (capped at the subset size, see 4.6), light augmentation (random crop after 8-pixel padding and
   horizontal flip).
2. **SimCLR linear probe.** The pretrained encoder is frozen. Features are extracted once without augmentation and
   standardised using training-set statistics only. A single linear layer is trained with full-batch Adam
   (learning rate 1e-3, weight decay 1e-4) for 300 steps. This measures how linearly separable the learned features are.
3. **SimCLR fine-tuning.** The pretrained encoder and a new linear layer are trained end to end with the same settings
   as the baseline, except for a smaller learning rate of 3e-4, so that the pretrained features are adjusted gently
   instead of overwritten. Using the same recipe as the baseline makes the comparison fair: the only difference is
   the starting weights.

### 4.5 Pretraining used for the main results
The encoder used for the linear probe and fine-tuning was pretrained with SimCLR on all 100,000 unlabelled images
for **50 epochs** (batch size 256, all five augmentations, temperature 0.5, AdamW with learning rate 3e-4, cosine decay,
mixed precision) on a single Kaggle GPU. One epoch took about 226 seconds, about 3.1 hours in total. The contrastive
loss fell from 5.12 after the first epoch to 4.58 at epoch 50, and changed by less than 0.01 over the last ten epochs
(results/logs/pretrain_full.csv).

### 4.6 Implementation note: batch size at 1% labels
At 1% labels the training subset has only 50 images, fewer than the batch size of 128. The original data loader
used `drop_last=True`, which discards the last incomplete batch. Here that was the only batch, so the model received
zero gradient updates and stayed at chance level (about 10%). We cap the batch size at the size of the training subset,
so every fraction trains for the full 100 epochs.

### 4.7 Metric and compute
The metric is top-1 accuracy on the 8,000 test images. The supervised baseline was trained on a Google Colab
T4 GPU. The SimCLR evaluation runs were done on Kaggle. All results use one seed (seed 0).

## 5. Results

### 5.1 Supervised baseline vs SimCLR

**Table 1.** Test accuracy (%) on STL-10 with different amounts of labelled training data. Best result in each row in bold.

| Labels used (per class) | Supervised (from scratch) | SimCLR linear probe | SimCLR fine-tune |
|---|---|---|---|
| 100% (500) | 75.5 | 72.6 | **81.3** |
| 20% (100) | 53.4 | 68.6 | **71.2** |
| 10% (50) | 44.3 | 65.2 | **65.5** |
| 1% (5) | 26.8 | **53.1** | 43.6 |

**Table 2.** Improvement over the supervised baseline (percentage points).

| Labels used | Linear probe | Fine-tune |
|---|---|---|
| 100% | -2.9 | +5.8 |
| 20% | +15.1 | +17.8 |
| 10% | +20.9 | +21.2 |
| 1% | +26.2 | +16.8 |

Figure 1 (results/labels_vs_acc.png) plots test accuracy against the percentage of labels used, on a log scale.

### 5.2 Analysis

**Pretraining helps most when labels are scarce.** The supervised baseline loses almost two thirds of its accuracy
as labels go from 100% to 1% (75.5% to 26.8%). SimCLR is much more robust: the best SimCLR model drops only from
81.3% to 53.1%. The gain over the baseline grows as labels become scarcer, from +5.8 points with all labels to
+21.2 points at 10% and +26.2 points at 1%.

**Label efficiency.** A linear probe trained on just 5 images per class (1%, 53.1%) matches the supervised model
trained on 100 images per class (20%, 53.4%). That is about 20 times fewer labels for the same accuracy. Fine-tuning
with 20% of the labels (71.2%) comes within 4.3 points of the supervised model trained on all 5,000 labels (75.5%).

**Linear probe vs fine-tuning.** With 10% or more labels, fine-tuning is best. Updating the whole encoder lets it
adapt to the classification task, and with all labels it beats the supervised baseline by 5.8 points. At 1% the
order reverses: the linear probe (53.1%) beats fine-tuning (43.6%) by 9.4 points. With only 50 images, training all
11 million parameters for 100 epochs overfits and damages the pretrained features. Training a single linear layer on
frozen features cannot overfit as badly. So when labels are extremely scarce, freezing the encoder is the safer choice.

**Frozen features with all labels.** With 100% of the labels the linear probe (72.6%) is 2.9 points below the
supervised baseline. Frozen features from a 50-epoch pretraining run are not as task-specific as a network trained
end to end on 5,000 labels. The original SimCLR paper pretrains for hundreds of epochs with much larger batches, and
longer pretraining would likely close this gap. Fine-tuning removes the gap entirely.

**Caveats.** Each number comes from a single run (seed 0), so differences of about 1 point, such as linear probe vs
fine-tune at 10% (65.2% vs 65.5%), should not be over-interpreted. The pretraining was short (50 epochs), so absolute
accuracies are below published SimCLR results. The trends, however, are large and consistent across label fractions.
