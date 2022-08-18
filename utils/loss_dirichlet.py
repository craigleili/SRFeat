import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_bmm(X, Y):
    assert X.dim() == Y.dim() == 3
    return torch.stack([torch.mm(X[b, ...], Y[b, ...]) for b in range(Y.size(0))], dim=0)


def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)


def mdiag(X, Y):
    assert X.dim() == Y.dim() + 1
    return torch.mul(X, torch.unsqueeze(Y, dim=Y.dim() - 1))


def diagm(X, Y):
    assert X.dim() + 1 == Y.dim()
    return torch.mul(torch.unsqueeze(X, dim=X.dim()), Y)


class DirichletLoss(nn.Module):

    def __init__(self, normalize=True, **kwargs):
        super().__init__()
        self.normalize = normalize
        for k, w in kwargs.items():
            setattr(self, k, w)

    def forward(self, feats, L, evecs, evals, mass):
        assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, sparse_bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return {'dirichlet': loss}
