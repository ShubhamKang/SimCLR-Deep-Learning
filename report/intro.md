# 1. Introduction

Deep learning models for image classification have achieved strong results, but their
success has historically depended on large amounts of labelled data. Collecting and
annotating labels at scale is expensive and time-consuming, which makes this dependence
a practical barrier in many real-world settings — medical imaging, satellite analysis,
and rare-event detection all face situations where unlabelled images are plentiful but
labelled examples number in the tens or hundreds.

This project examines one concrete instantiation of that problem: how accurately can a
classifier be trained on STL-10 when only a small fraction of the available labels is
used, and can self-supervised pretraining on the unlabelled images close the gap?

The STL-10 dataset [Coates et al., 2011] is designed precisely for this setting. It
provides 100,000 unlabelled images alongside a small labelled training set of 5,000
images (500 per class across 10 categories), and a test set of 8,000 images. We study
label fractions of 100%, 20%, 10%, and 1%, corresponding to 5,000 down to just 50
labelled training images — a regime where supervised training from scratch degrades
sharply.

Our approach is SimCLR (Simple Framework for Contrastive Learning of Visual
Representations) [Chen et al., 2020], which learns visual features from unlabelled
images by training a network to produce similar representations for two augmented views
of the same image and dissimilar representations for views of different images. The
learned encoder is then transferred to the classification task with a small labelled set.

We investigate two aspects of SimCLR that are practically important on a limited compute
budget:

1. **Which augmentations matter most?** SimCLR relies heavily on the composition of
   image augmentations to define the self-supervised task. We run a leave-one-out and
   single-augmentation ablation across five augmentations — random crop, horizontal flip,
   colour jitter, grayscale, and Gaussian blur — to measure the contribution of each.

2. **How does batch size affect the learned representations?** The contrastive loss
   treats every other image in the same batch as a negative example, so larger batches
   provide more negatives and potentially stronger learning signal. We train with batch
   sizes of 64, 128, 256, and 512 on a fixed short schedule to isolate the effect of
   batch size under a constant compute budget.

The rest of the report is organised as follows. Section 2 reviews related work on
contrastive self-supervised learning and supervised baselines. Section 3 describes the
experimental setup. Section 4 presents results for the supervised baseline versus
SimCLR linear probe and fine-tuning. Section 5 presents the augmentation ablation.
Section 6 presents the batch-size ablation. Section 7 concludes.
