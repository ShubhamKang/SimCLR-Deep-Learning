# Abstract, ablation results, discussion, conclusion (Shubham)
All numbers come from results/all_results.csv (one seed, STL-10 test set, 8,000 images).

## Abstract
Labelled data is expensive, while unlabelled images are easy to collect. We study whether self-supervised
contrastive pretraining with SimCLR lets a classifier reach good accuracy with far fewer labels, and which
data augmentations matter most when compute is limited. Using STL-10, we pretrain a ResNet-18 encoder on
100,000 unlabelled images for 50 epochs on a single GPU, then train classifiers on 100%, 20%, 10% and 1%
of the 5,000 labelled training images. With 10% of the labels, a supervised ResNet-18 trained from scratch
reaches 44.3% test accuracy, while a linear probe on the SimCLR encoder reaches 65.2% and fine-tuning reaches
65.5%. With only 1% of the labels (50 images), the linear probe reaches 53.1%, double the baseline's 26.8%.
An ablation over five augmentations on a short 20-epoch schedule shows that random resized crop matters most:
removing it lowers linear-probe accuracy from 57.0% to 38.2%, close to using no augmentation at all (36.9%).
Colour jitter (-7.0 points) and grayscale (-5.8) come next, while flip and blur have no measurable benefit on
this budget. Smaller batches did better at a fixed number of epochs: 59.3% at batch size 64 versus 53.2% at 512.

## Ablation results
Schedule for every run: 20 epochs of SimCLR pretraining on the same 20,000 unlabelled images (seed 0),
batch size 256 unless stated, then a linear probe on all 5,000 labels. Figures: results/aug_ablation.png,
results/batchsize_ablation.png.

### Augmentations
| Setting | Augmentations on | Accuracy (%) | Change vs full |
|---|---|---|---|
| full | crop, flip, colour, gray, blur | 57.0 | - |
| no_crop | flip, colour, gray, blur | 38.2 | -18.8 |
| no_color | crop, flip, gray, blur | 50.0 | -7.0 |
| no_gray | crop, flip, colour, blur | 51.2 | -5.8 |
| no_flip | crop, colour, gray, blur | 56.9 | -0.1 |
| no_blur | crop, flip, colour, gray | 58.6 | +1.6 |
| crop_color | crop, colour | 53.0 | -4.0 |
| only_crop | crop | 45.1 | -11.9 |
| only_color | colour | 42.7 | -14.3 |
| none | none | 36.9 | -20.1 |

Ranking by the drop when one augmentation is removed: crop (18.8) > colour jitter (7.0) > grayscale (5.8)
> flip (0.1) and blur (-1.6, i.e. accuracy went up).

### Batch size
All augmentations on, same 20 epochs and 20,000 images. The batch-256 point is the `full` run above.

| Batch size | Updates per epoch | Accuracy (%) | Pretraining time (s) |
|---|---|---|---|
| 64 | 312 | 59.3 | 930 |
| 128 | 156 | 58.2 | 936 |
| 256 | 78 | 57.0 | 944 |
| 512 | 39 | 53.2 | 979 |

## Discussion

### Does pretraining help when labels are scarce?
- Yes, and more so the fewer labels there are. The gap between the best SimCLR model and the baseline is
  +5.8 points at 100% labels, +17.8 at 20%, +21.2 at 10% and +26.2 at 1% (results/labels_vs_acc.png).
- Linear probe vs fine-tune: fine-tuning is best at 100%, 20% and 10% labels (81.3, 71.2, 65.5%), but at 1%
  (50 images) the linear probe is clearly better (53.1% vs 43.6%). Training all 11M weights on 50 images
  overfits, while the frozen encoder keeps the pretrained features intact.
- At 100% labels the linear probe (72.6%) is below the supervised baseline (75.5%): with 5,000 labels the
  baseline has enough data, and 50 epochs of pretraining give useful but incomplete features.
  Fine-tuning still beats the baseline by 5.8 points.

### Which augmentations matter most on a small budget?
- Random resized crop is by far the most important. Without it accuracy (38.2%) is almost the same as with
  no augmentation (36.9%). Crop alone (45.1%) is better than colour jitter alone (42.7%).
- Colour jitter is second. Adding it to crop raises accuracy from 45.1% to 53.0% (+7.9 points). This agrees with Chen et al. (2020), who found crop plus colour distortion to
  be the key pair: without colour changes, two crops of one image can be matched by their colour
  statistics alone, so the network learns less about shape and content.
- Grayscale also helps (-5.8 points when removed), consistent with it being another colour distortion.
- Flip and blur did not help here. Removing flip changed accuracy by 0.1 points, and removing blur raised it
  by 1.6 points. Possible reasons: 96x96 images are already low in detail, so blur removes useful
  information, and a 20-epoch schedule may be too short to benefit from extra, weaker augmentations.
  Both differences are within what one seed can explain, so we read them as "no benefit", not "harmful".
- Cost: augmentations run on the CPU and data loading limited speed. Within the same Kaggle session,
  20 epochs took 944 s with all augmentations, 645 s without colour jitter and 357 s with crop only.
  The no_blur run took 633 s, but it ran in a different session, so that comparison is only indicative.
- Practical answer for a small budget: use crop, flip, colour jitter and grayscale, and skip blur. That
  exact set (no_blur) scored 58.6%, slightly above the full set, and avoids the cost of blur.

### Batch size
- SimCLR reports that larger batches help because they give more negatives per step. On our budget the
  opposite happened: accuracy fell from 59.3% (batch 64) to 53.2% (batch 512).
- The likely reason is the number of updates. With a fixed 20 epochs, batch 512 makes only 39 updates per
  epoch versus 312 for batch 64, and the paper's large-batch advantage appears with long training (hundreds
  of epochs) and the LARS optimiser, which we did not use.
- Run time was almost the same for every batch size (930-979 s), which again points to CPU data loading,
  not the GPU, as the bottleneck. On a small budget a smaller batch is the better choice.

### Limitations
- Short pretraining (50 epochs vs 100 to 1,000 in the paper), so absolute accuracies are lower than published.
- Ablations use 20 epochs on a 20,000-image subset and one seed, so differences under about 2 points
  (flip, blur) may be noise.
- The ablations ran in two Kaggle sessions; accuracy is unaffected, but run times are only fully
  comparable within one session.
- AdamW instead of the LARS optimiser, and batch sizes up to 512 instead of up to 8,192 in the paper.
- 96x96 images and a ResNet-18 only.

## Conclusion
SimCLR pretraining on unlabelled STL-10 images improved accuracy over a supervised baseline at every label
fraction we tested (fine-tuning at 100%, 20% and 10%; the linear probe at 1%), with the largest gains when
labels were scarcest (+21.2 points at 10% labels, +26.2 at 1%). On a small compute budget, random resized crop
is the augmentation that matters most, followed by colour jitter and grayscale; flip and blur can be dropped
without losing accuracy. Smaller batches worked better than larger ones at a fixed number of epochs. Future
work: longer pretraining, more seeds for the ablations, and the LARS optimiser with larger batches.
