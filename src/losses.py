"""The contrastive loss used by SimCLR: NT-Xent (report Section 3.1, equation 1).

    loss(i, j) = -log( exp(sim(z_i, z_j) / t) / sum over k != i of exp(sim(z_i, z_k) / t) )

sim = cosine similarity, t = temperature (0.5 in all our runs).

QUESTIONS THIS FILE ANSWERS
Q: What does NT-Xent stand for?
   Normalised Temperature-scaled cross-entropy (Xent). "Normalised": vectors are scaled to
   length 1 so dot products are cosine similarities. "Temperature": similarities are divided
   by t. "Cross-entropy": the task is treated as classification.
Q: What are positives and negatives here?
   In a batch of N images there are 2N views. For each view, its partner (the other view of
   the same image) is the positive; the remaining 2N-2 views are negatives.
Q: Why is it a classification problem?
   Each view must pick its partner out of the other 2N-1 views, like choosing the right
   class out of 2N-1 classes, so cross-entropy is the natural loss.
Q: Why does batch size matter for this loss?
   Bigger batch -> more negatives per view (2N-2). The SimCLR paper found that more negatives
   help with long training. In our 20-epoch ablation larger batches did WORSE (512 was the
   lowest in both runs) because a fixed number of epochs gives far fewer updates
   (780 for batch 512 vs 3,120 for batch 128 on 20,000 images).
Q: What does the temperature do, and why 0.5?
   Dividing by a small t makes the softmax sharper, so the loss focuses more on the hardest
   negatives (the most similar wrong views). 0.5 is a common choice for small images (the
   SimCLR paper's CIFAR-10 experiments used it). We used 0.5 for every run and did not tune it.
Q: What do the loss values mean?
   Lower = the network finds partners more easily. Our main run went from 5.12 (epoch 1) to
   4.58 (epoch 50) and changed by less than 0.01 over the last ten epochs.
"""
import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.5):
    """NT-Xent loss for one batch.

    z1, z2 : [N, d] projections (d = 128). Row i of z1 and row i of z2 come from the two
             views of the SAME image (a positive pair). All other rows are negatives.
    Returns the average loss over all 2N views.
    """
    n = z1.size(0)  # N = number of images in the batch

    # Stack both views into one [2N, d] matrix: rows 0..N-1 are view 1, rows N..2N-1 view 2.
    # L2-normalise each row (length 1) so that a dot product equals cosine similarity.
    # .float() does this in 32-bit even under mixed precision, for numerical stability.
    z = F.normalize(torch.cat([z1, z2]), dim=1).float()

    # [2N, 2N] matrix of cosine similarities between every pair of views, divided by t.
    sim = z @ z.t() / temperature

    # A view is always identical to itself, which would make the task trivial, so remove
    # that option: setting the diagonal to -infinity gives it zero probability in the softmax.
    sim.fill_diagonal_(float("-inf"))

    # The correct "class" for row i (view 1 of image i) is row i+N (view 2 of image i), and
    # for row i+N it is row i. This builds the index of every row's positive partner.
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)

    # Softmax over the 2N-1 candidates in each row + cross-entropy against the true partner.
    # This is exactly the formula at the top of the file, averaged over the 2N rows.
    return F.cross_entropy(sim, targets)
