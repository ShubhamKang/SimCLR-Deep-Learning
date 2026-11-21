# 2. Related Work

## 2.1 Supervised Baselines for Image Classification

Convolutional neural networks trained end-to-end with cross-entropy loss on labelled
data have been the standard approach to image classification since AlexNet [Krizhevsky
et al., 2012]. ResNet architectures [He et al., 2016] introduced residual connections
that allow training very deep networks reliably, and ResNet-18 remains a common choice
for small-to-medium datasets. The core limitation of fully supervised training is data
hunger: accuracy falls quickly when fewer labels are available, particularly in the
low-label regime (< 10% of the training set), because the network overfits to the small
labelled set rather than learning generalisable features.

## 2.2 Self-Supervised and Contrastive Learning

Self-supervised learning sidesteps the label bottleneck by constructing a pretext task
directly from unlabelled data. Early pretext tasks included predicting image rotations
[Gidaris et al., 2018], solving jigsaw puzzles [Noroozi & Favaro, 2016], and
colourisation [Zhang et al., 2016]. These tasks produced transferable features but were
often brittle and task-specific.

Contrastive learning unified these ideas under a principled objective: pull together
representations of semantically related pairs and push apart unrelated ones. Instance
discrimination [Wu et al., 2018] treated each image as its own class and used a memory
bank of negatives. Momentum Contrast (MoCo) [He et al., 2020] replaced the memory bank
with a momentum encoder and a queue, decoupling batch size from the number of
negatives. Contrastive Predictive Coding (CPC) [Oord et al., 2018] learned temporal
structure in sequential data using a similar contrastive objective.

## 2.3 SimCLR

SimCLR [Chen et al., 2020] simplified the contrastive learning pipeline while achieving
state-of-the-art results at the time of publication. Its key contributions are:

- **Two-view augmentation.** Each training image is augmented twice with a stochastic
  pipeline (crop, flip, colour jitter, grayscale, blur) to form a positive pair. All
  other images in the same batch form negative pairs.

- **Projection head.** A two-layer MLP is placed on top of the encoder during
  pretraining. Chen et al. showed that computing the contrastive loss on the projection
  head's output, and then discarding the head after pretraining, consistently improves
  the quality of the encoder's representations for downstream tasks.

- **NT-Xent loss.** The normalised temperature-scaled cross-entropy loss treats the
  contrastive task as a 2N-way classification problem: for each of the 2N views in a
  batch, the network must identify the one correct positive partner among the 2N-1
  remaining views.

- **Large batch size.** SimCLR's original results on ImageNet used batch sizes of 4,096
  or 8,192, exploiting the fact that more negatives per image strengthen the
  contrastive signal. This motivates our ablation study on batch size.

The SimCLR paper demonstrated that the composition of augmentations is the single most
important design choice: removing random cropping or colour jitter caused the largest
accuracy drops on ImageNet linear evaluation. We replicate and extend this finding on
STL-10 under a small compute budget.

## 2.4 Linear Evaluation and Fine-Tuning Protocols

A standard way to measure the quality of self-supervised representations is linear
evaluation: freeze the pretrained encoder and train only a single linear classifier on
top of it using labelled data. A good encoder should produce linearly separable features
without any task-specific fine-tuning. Fine-tuning — initialising from the pretrained
encoder and updating all weights — typically yields higher accuracy but requires more
labelled data to avoid overfitting. Comparing linear probe accuracy to fine-tuning
accuracy reveals how much additional gain comes from adapting the encoder itself versus
using it as a fixed feature extractor.

## 2.5 STL-10 Benchmark

STL-10 [Coates et al., 2011] was designed as a benchmark for self-supervised and
semi-supervised methods. Its 96×96 RGB images are higher resolution than CIFAR-10, and
the 100,000 unlabelled images are drawn from a broader distribution than the 10 labelled
classes, making the pretraining task slightly harder and more realistic. Prior SimCLR
results on STL-10 report linear probe accuracies in the range of 85–90% with long
training schedules on strong hardware [Chen et al., 2020]; our experiments use shorter
schedules appropriate to a student compute budget and focus on relative comparisons
rather than absolute state-of-the-art numbers.
