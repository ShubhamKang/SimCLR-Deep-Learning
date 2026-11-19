import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.5):
    """NT-Xent loss (SimCLR). z1, z2: [N, d] projections of two views."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1).float()
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)
