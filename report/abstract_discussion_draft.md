# Draft: Abstract, Discussion, Conclusion (Shubham)
Replace every [X] with the real number from results/ once all runs are done.

## Abstract
Labelled data is expensive, while unlabelled images are easy to collect. We study whether self-supervised
contrastive pretraining with SimCLR lets a classifier reach good accuracy with far fewer labels, and which
data augmentations matter most when compute is limited. Using STL-10, we pretrain a ResNet-18 encoder on
100,000 unlabelled images for [X] epochs on a single GPU, then train classifiers on 100%, 20%, 10% and 1%
of the 5,000 labelled training images. With 10% of the labels, a supervised ResNet-18 trained from scratch
reaches 44.3% test accuracy, while a linear probe on the SimCLR encoder reaches [X]% and fine-tuning reaches
[X]%. An ablation over five augmentations, on a short 20-epoch schedule, shows that [random resized crop and
colour jitter] account for most of the benefit: removing crop lowers linear-probe accuracy by [X] points and
removing colour jitter by [X] points, while flip, grayscale and blur each change it by less than [X] points.
Larger batches [helped / did not help] on our budget: accuracy went from [X]% at batch size 64 to [X]% at 512.

## Discussion

### Does pretraining help when labels are scarce?
- Compare the gap between SimCLR and the supervised baseline at each label fraction (table [X], figure [X]).
- Expected pattern: the gap is largest at 1% and 10% labels and smallest at 100%. State whether our numbers follow it.
- Linear probe vs fine-tune: fine-tuning usually wins when there are enough labels; at 1% the linear probe
  may be as good or better because fine-tuning overfits 50 images. Say which happened.

### Which augmentations matter most on a small budget?
- Rank augmentations by the accuracy drop when each is removed (leave-one-out runs).
- Compare single augmentations (only_crop, only_color) with the pair crop_color and with the full set.
- Link to SimCLR (Chen et al., 2020): the paper found crop + colour jitter to be the key combination,
  because without colour jitter two crops of the same image can be matched by their colour histogram alone.
  State whether our short-schedule results agree.
- Cost: all augmentations run on the CPU. On Colab (2 cores) data loading, not the GPU, limited speed
  (about 8 minutes per epoch on 100k images). Blur is the most expensive augmentation; if it adds little
  accuracy, dropping it is a cheap way to save time.

### Batch size
- Larger batches give more negatives per step. Report accuracy, time per run and GPU memory for 64/128/256/512.
- Note that with a fixed number of epochs, larger batches also mean fewer optimisation steps.

### Limitations
- Short pretraining ([X] epochs vs 100 to 1,000 in the paper), so absolute accuracies are lower than published ones.
- Ablations use 20 epochs on a 20,000-image subset and one seed, so differences under about [1-2] points
  may be noise.
- AdamW instead of the LARS optimiser and the large batches (up to 8,192) used in the paper.
- 96x96 images and a ResNet-18 only.

## Conclusion
SimCLR pretraining on unlabelled STL-10 images improved accuracy over a supervised baseline at every label
fraction we tested, with the largest gain when labels were scarcest ([X] points at 10% labels). On a small
compute budget, [crop and colour jitter] are the augmentations that matter; the others can be dropped with
little loss. Future work: longer pretraining, more seeds for the ablations, and stronger evaluation protocols.
