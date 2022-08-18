import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pairwise_distance_matrix(x, y, squared=False, eps=1e-12):
    x2 = torch.sum(x**2, dim=-1, keepdim=True)
    y2 = torch.sum(y**2, dim=-1, keepdim=True)
    dist2 = -2.0 * torch.matmul(x, torch.transpose(y, -2, -1))
    dist2 += x2
    dist2 += torch.transpose(y2, -2, -1)
    if squared:
        return dist2
    else:
        dist2 = torch.clamp(dist2, min=eps)
        return torch.sqrt(dist2)


def nce_loss(F0, F1, corrs, subsample=None, T=0.07):
    assert F0.dim() == F1.dim() == corrs.dim() == 3
    batch_size = F0.size(0)

    corrs = corrs.long()
    F0_sel = list()
    F1_sel = list()
    if subsample is not None and subsample > 0 and corrs.shape[1] > subsample:
        for bidx in range(batch_size):
            idx_sel = np.random.choice(len(corrs[bidx]), subsample, replace=False)
            corr_sel = corrs[bidx][idx_sel]
            F0_sel.append(F0[bidx][corr_sel[:, 0]])
            F1_sel.append(F1[bidx][corr_sel[:, 1]])
    else:
        for bidx in range(batch_size):
            F0_sel.append(F0[bidx][corrs[bidx, :, 0]])
            F1_sel.append(F1[bidx][corrs[bidx, :, 1]])
    F0 = torch.stack(F0_sel, dim=0)
    F1 = torch.stack(F1_sel, dim=0)

    logits = F1 @ torch.transpose(F0, 1, 2)
    logits /= T

    labels = torch.arange(logits.size(-1), device=logits.device).long()
    labels = labels.view(1, -1).repeat(batch_size, 1)

    loss = F.cross_entropy(logits, labels)

    return loss


class NCELoss(nn.Module):

    def __init__(self, num_pos, **kwargs):
        super().__init__()
        self.num_pos = num_pos
        for k, w in kwargs.items():
            setattr(self, k, w)

    def forward(self, feats0, feats1, corrs):
        feats0 = F.normalize(feats0, p=2, dim=-1)
        feats1 = F.normalize(feats1, p=2, dim=-1)
        loss = nce_loss(feats0, feats1, corrs, self.num_pos)
        return {'contrast': loss}
