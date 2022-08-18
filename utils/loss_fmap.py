import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.misc import lstsq


def frobenius_loss(a, b):
    assert a.dim() == b.dim() == 3
    loss = torch.sum((a - b)**2, axis=(1, 2))
    return torch.mean(loss)


def pmap_to_fmap(evecs0, evecs1, corrs):
    assert evecs0.dim() == evecs1.dim() == corrs.dim() == 3
    assert evecs0.size(0) == evecs1.size(0) == corrs.size(0)

    evecs0_sel, evecs1_sel = list(), list()
    for bidx in range(corrs.size(0)):
        evecs0_sel.append(evecs0[bidx][corrs[bidx, :, 0], :])
        evecs1_sel.append(evecs1[bidx][corrs[bidx, :, 1], :])
    evecs0_sel = torch.stack(evecs0_sel, dim=0)
    evecs1_sel = torch.stack(evecs1_sel, dim=0)
    fmap01_t = lstsq(evecs0_sel, evecs1_sel, False)
    fmap01 = torch.transpose(fmap01_t, 1, 2)
    return fmap01


class SoftFmapLoss(nn.Module):

    def __init__(self, spectral_dims, max_val, T=0.07, **kwargs):
        super().__init__()
        self.spectral_dims = spectral_dims
        self.max_val = max_val
        self.T = T
        for k, w in kwargs.items():
            setattr(self, k, w)

    def forward(self, evecs0, evecs1, evals0, evals1, mass0, mass1, feats0, feats1, corrs):
        assert evecs0.size(0) == evals0.size(0) == mass0.size(0) == \
               feats0.size(0) == corrs.size(0) == 1

        feats0 = F.normalize(feats0, p=2, dim=-1)
        feats1 = F.normalize(feats1, p=2, dim=-1)
        pmap10_soft = feats1 @ torch.transpose(feats0, 1, 2)
        pmap10_soft = torch.softmax(pmap10_soft / self.T, dim=-1)

        loss_dict = dict()
        loss_dict['fmap'] = 0
        for SD in self.spectral_dims:
            with torch.no_grad():
                fmap01_gt = pmap_to_fmap(evecs0[:, :, :SD], evecs1[:, :, :SD], corrs)
            fmap01_est = (torch.transpose(evecs1[:, :, :SD], 1, 2) *
                          torch.unsqueeze(mass1, 1)) @ pmap10_soft @ evecs0[:, :, :SD]
            loss = frobenius_loss(fmap01_est, fmap01_gt)
            loss_dict[f'fmap_{SD}'] = loss
            loss_dict['fmap'] = loss_dict['fmap'] + loss
        loss_dict['fmap'] = torch.clamp(loss_dict['fmap'] / len(self.spectral_dims),
                                        max=self.max_val)
        return loss_dict
