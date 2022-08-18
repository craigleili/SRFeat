import os.path as osp
import numpy as np
import torch
import yaml
import omegaconf
import random
import scipy.linalg as sla
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from pathlib import Path


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def pinverse_cpu(x):
    device = x.device
    return torch.pinverse(x.cpu()).to(device)


def svd_cpu(x, eps=1e-4, require_backward=False):
    device = x.device
    x = x.cpu()

    _perturb = lambda _y, _eps: _y + _eps * _y.mean() * torch.rand_like(_y)

    try:
        U, S, Vh = torch.linalg.svd(x)
    except:
        U, S, Vh = torch.linalg.svd(_perturb(x, eps))

    while require_backward and len(torch.unique(S)) != len(S):
        U, S, Vh = torch.linalg.svd(_perturb(x, eps))
        eps *= 2

    return U.to(device), S.to(device), torch.t(Vh).to(device)


def lstsq(A, B, require_grad):
    if require_grad:
        return torch.pinverse(A) @ B
    else:
        if A.dim() == 2:
            return torch.lstsq(B, A).solution[:A.size(-1)]
        elif A.dim() == 3:
            sols = [
                torch.lstsq(B[bidx], A[bidx]).solution[:A.size(-1)] for bidx in range(A.size(0))
            ]
            return torch.stack(sols, dim=0)
        else:
            raise NotImplementedError


def omegaconf_to_dotdict(hparams):

    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if v is None:
                res[k] = v
            elif isinstance(v, omegaconf.DictConfig):
                res.update({k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()})
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            elif isinstance(v, omegaconf.ListConfig):
                res[k] = omegaconf.OmegaConf.to_container(v, resolve=True)
            else:
                raise RuntimeError('The type of {} is not supported.'.format(type(v)))
        return res

    return _to_dot_dict(hparams)


def validate_gradient(model):
    flag = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                flag = False
            if torch.any(torch.isinf(param.grad)):
                flag = False
    return flag


class KNNSearch(object):
    DTYPE = np.float32
    NJOBS = 8

    def __init__(self, data):
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, n_jobs=self.NJOBS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, n_jobs=self.NJOBS)
        return nnindices


def seeding(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    #import torch.multiprocessing
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def run_trainer(trainer_cls):
    # train: config=<config_path>
    # test: skip_train=True test_ckpt=<ckpt_latest.pth>
    cfg_cli = OmegaConf.from_cli()
    if cfg_cli.skip_train:
        cfg = OmegaConf.merge(
            torch.load(cfg_cli.test_ckpt)['cfg'],
            cfg_cli,
        )
        OmegaConf.resolve(cfg)
        seeding(cfg.seed)
        trainer = trainer_cls(cfg)
        trainer.test()
    else:
        cfg = OmegaConf.merge(
            OmegaConf.load(cfg_cli.config),
            cfg_cli,
        )
        OmegaConf.resolve(cfg)
        seeding(cfg.seed)
        trainer = trainer_cls(cfg)
        trainer.train()
        trainer.test()
