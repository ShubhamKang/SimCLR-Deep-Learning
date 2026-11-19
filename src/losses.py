"""The contrastive loss used by SimCLR."""
import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.5):
    """NT-Xent loss (normalised temperature-scaled cross entropy).

    z1, z2 : [N, d] projections. Row i of z1 and row i of z2 come from two views
             of the SAME image (a positive pair). All other rows come from other
             images in the batch and act as negatives.

    Idea: for each of the 2N views, the network must pick out its partner view from
    the other 2N-1 views. That is a classification problem, so we use cross entropy.

    Why batch size matters (one of our ablations): a bigger batch gives more
    negatives per image, which usually makes the task harder and the features better.

    temperature : small values sharpen the similarity scores and weigh hard negatives
                  more. 0.5 is a common default for small datasets.
    """
    n = z1.size(0)

    # Stack both views into one [2N, d] matrix, then L2-normalise each row so that a
    # dot product equals cosine similarity. .float() keeps this stable under
    # mixed-precision (fp16) training.
    z = F.normalize(torch.cat([z1, z2]), dim=1).float()

    # Pairwise cosine similarity between all 2N views, divided by the temperature.
    sim = z @ z.t() / temperature

    # An image is trivially identical to itself, so remove that option by setting its
    # score to -infinity (it then gets zero probability in the softmax).
    sim.fill_diagonal_(float("-inf"))

    # The correct answer for row i (view 1 of image i) is row i+N (view 2 of image i),
    # and vice versa. This builds the index of each row's positive partner.
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)

    # Softmax over the 2N-1 candidates + cross entropy against the true partner.
    return F.cross_entropy(sim, targets)
