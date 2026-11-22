"""The contrastive loss used by SimCLR, called NT-Xent.

What is a loss?
    A loss is a single number that says how wrong the network currently is. Training
    changes the weights to make this number smaller.

What does NT-Xent measure?
    In a batch of N images there are 2N views (two per image). For every view, the network
    has to find its partner (the other view of the same image) among the other 2N - 1 views.
    The partner is called the "positive"; the other 2N - 2 views are "negatives".
    This is treated like a classification task where the correct "class" is the partner,
    so it uses cross-entropy, the usual loss for classification.

The name: Normalised (vectors are scaled to length 1), Temperature-scaled (similarities are
divided by a number called the temperature), cross-entropy (Xent).

The formula, for a view i whose partner is view j:
    loss(i, j) = -log( exp(sim(z_i, z_j) / t) / sum over all k other than i of exp(sim(z_i, z_k) / t) )
where sim is the cosine similarity and t is the temperature. The loss is small when view i
is much more similar to its partner j than to every other view.
"""
# torch is PyTorch.
import torch
# torch.nn.functional (short name F) contains functions such as normalize and cross_entropy.
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.5):
    """Calculate the NT-Xent loss for one batch.

    z1 : the projections of the first view of every image, shape [N, 128]
    z2 : the projections of the second view of every image, shape [N, 128]
         Row i of z1 and row i of z2 come from the SAME image.
    temperature : the number the similarities are divided by (0.5 in all our runs).
                  A smaller temperature makes the loss focus more on the hardest negatives,
                  the wrong views that look most similar.
    Returns one number: the average loss over all 2N views.
    """
    # z1.size(0) is the number of rows, which is N, the number of images in the batch.
    n = z1.size(0)

    # torch.cat([z1, z2]) stacks the two [N, 128] tables on top of each other into one
    # [2N, 128] table: rows 0 to N-1 are first views, rows N to 2N-1 are second views.
    # F.normalize(..., dim=1) scales every row to length 1. After that, the dot product of
    # two rows equals their cosine similarity: 1 means pointing the same way (very similar),
    # 0 means unrelated, -1 means opposite.
    # .float() makes sure the numbers are 32-bit, which is more precise than the 16-bit
    # numbers used during mixed-precision training.
    z = F.normalize(torch.cat([z1, z2]), dim=1).float()

    # z.t() is the transposed table (rows become columns). z @ z.t() is a matrix
    # multiplication that gives a [2N, 2N] table: the entry in row a, column b is the cosine
    # similarity between view a and view b. Dividing by the temperature scales all of them.
    sim = z @ z.t() / temperature

    # The diagonal (row a, column a) compares each view with itself, which is always the most
    # similar and would make the task trivial. fill_diagonal_ sets these entries to minus
    # infinity, which means "impossible": they get probability 0 in the next step.
    sim.fill_diagonal_(float("-inf"))

    # For each row we need the column of its correct partner:
    #   row i      (first view of image i)  -> partner is row i + N (second view of image i)
    #   row i + N  (second view of image i) -> partner is row i
    # torch.arange(n, 2 * n) gives N, N+1, ..., 2N-1 (the partners of rows 0 to N-1), and
    # torch.arange(0, n) gives 0, 1, ..., N-1 (the partners of rows N to 2N-1).
    # .to(z.device) puts this list on the same device (GPU or CPU) as the similarities.
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)

    # F.cross_entropy treats each row of `sim` as scores for 2N "classes". It turns the
    # scores into probabilities (softmax) and returns -log(probability of the correct class),
    # averaged over all rows. This is exactly the formula at the top of the file.
    return F.cross_entropy(sim, targets)
