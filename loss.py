import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def lstsq(A, B, require_grad):
    if require_grad:
        return torch.linalg.pinv(A) @ B
    else:
        if A.dim() == 2:
            return torch.lstsq(B, A).solution[:A.size(-1)]
        elif A.dim() == 3:
            sols = [
                torch.lstsq(B[bidx], A[bidx]).solution[:A.size(-1)]
                for bidx in range(A.size(0))
            ]
            return torch.stack(sols, dim=0)
        else:
            raise NotImplementedError


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


def frobenius_loss(a, b):
    assert a.dim() == b.dim() == 3
    loss = torch.sum((a - b)**2, axis=(1, 2))
    return torch.mean(loss)


def sparse_bmm(X, Y):
    assert X.dim() == Y.dim() == 3
    return torch.stack(
        [torch.mm(X[b, ...], Y[b, ...]) for b in range(Y.size(0))], dim=0)


def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)


def mdiag(X, Y):
    assert X.dim() == Y.dim() + 1
    return torch.mul(X, torch.unsqueeze(Y, dim=Y.dim() - 1))


def diagm(X, Y):
    assert X.dim() + 1 == Y.dim()
    return torch.mul(torch.unsqueeze(X, dim=X.dim()), Y)


class SoftFmapLoss(nn.Module):

    def __init__(self, spectral_dims, max_val, T=0.07, **kwargs):
        super().__init__()
        self.spectral_dims = spectral_dims
        self.max_val = max_val
        self.T = T
        for k, w in kwargs.items():
            setattr(self, k, w)

    def forward(self, evecs_s, evals_s, mass_s, has_spectral_s, h_s, batch_s,
                evecs_t, evals_t, mass_t, has_spectral_t, h_t, batch_t, y):
        assert has_spectral_s.shape[0] == has_spectral_t.shape[0]
        batch_size = has_spectral_s.shape[0]

        assert evecs_s.shape[1] == evecs_t.shape[1]
        num_eigs = evecs_s.shape[1]

        loss = 0
        count = 0
        for bidx in range(batch_size):
            if not has_spectral_s[bidx].item() or \
                not has_spectral_t[bidx].item():
                continue

            flags_s = (batch_s == bidx)
            flags_t = (batch_t == bidx)

            evecs0 = torch.unsqueeze(evecs_s[flags_s], 0)
            evecs1 = torch.unsqueeze(evecs_t[flags_t], 0)
            evals0 = torch.unsqueeze(
                evals_s[bidx * num_eigs:(bidx + 1) * num_eigs], 0)
            evals1 = torch.unsqueeze(
                evals_t[bidx * num_eigs:(bidx + 1) * num_eigs], 0)
            mass0 = torch.unsqueeze(mass_s[flags_s], 0)
            mass1 = torch.unsqueeze(mass_t[flags_t], 0)
            feats0 = torch.unsqueeze(h_s[flags_s], 0)
            feats1 = torch.unsqueeze(h_t[flags_t], 0)

            corrs = y[1, flags_s]
            corrs = torch.stack((torch.arange(
                len(corrs), dtype=corrs.dtype, device=corrs.device), corrs), 1)
            corrs = torch.unsqueeze(corrs, 0)

            loss_dict = self._forward_helper(evecs0, evecs1, evals0, evals1,
                                             mass0, mass1, feats0, feats1,
                                             corrs)
            loss = loss + loss_dict['fmap']
            count += 1

        if count > 0:
            return loss / count
        else:
            return 0

    def _forward_helper(self, evecs0, evecs1, evals0, evals1, mass0, mass1,
                        feats0, feats1, corrs):
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
                fmap01_gt = pmap_to_fmap(evecs0[:, :, :SD], evecs1[:, :, :SD],
                                         corrs)
            fmap01_est = (
                torch.transpose(evecs1[:, :, :SD], 1, 2) *
                torch.unsqueeze(mass1, 1)) @ pmap10_soft @ evecs0[:, :, :SD]
            loss = frobenius_loss(fmap01_est, fmap01_gt)
            loss_dict[f'fmap_{SD}'] = loss
            loss_dict['fmap'] = loss_dict['fmap'] + loss
        loss_dict['fmap'] = torch.clamp(loss_dict['fmap'] /
                                        len(self.spectral_dims),
                                        max=self.max_val)
        return loss_dict


class DirichletLoss(nn.Module):

    def __init__(self, normalize=True, **kwargs):
        super().__init__()
        self.normalize = normalize
        for k, w in kwargs.items():
            setattr(self, k, w)

    def forward(self, has_laplacian_s, laplacian_s, h_s, batch_s,
                has_laplacian_t, laplacian_t, h_t, batch_t):
        assert has_laplacian_s.shape[0] == has_laplacian_t.shape[0]
        batch_size = has_laplacian_s.shape[0]
        assert laplacian_s.shape[1] == laplacian_s.shape[1]

        loss = 0
        count = 0
        for bidx in range(batch_size):
            if not has_laplacian_s[bidx].item() or \
                not has_laplacian_t[bidx].item():
                continue

            flags_s = (batch_s == bidx)
            flags_t = (batch_t == bidx)

            L0 = laplacian_s[flags_s]
            L0 = torch.unsqueeze(L0[:, :L0.shape[0]], 0)

            L1 = laplacian_t[flags_t]
            L1 = torch.unsqueeze(L1[:, :L1.shape[0]], 0)

            feats0 = torch.unsqueeze(h_s[flags_s], 0)
            feats1 = torch.unsqueeze(h_t[flags_t], 0)

            loss_dict0 = self._forward_helper(feats0, L0)
            loss_dict1 = self._forward_helper(feats1, L1)

            loss = loss + 0.5 * (loss_dict0['dirichlet'] +
                                 loss_dict1['dirichlet'])
            count += 1
        if count > 0:
            return loss / count
        else:
            return 0

    def _forward_helper(self, feats, L):
        assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, sparse_bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return {'dirichlet': loss}
