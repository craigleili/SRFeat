import os.path as osp
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from collections import defaultdict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from diffusion_net.utils import sparse_np_to_torch
from data.utils import get_transforms


def collate_fn(data_list):
    data_dict = defaultdict(list)
    for sdict in data_list:
        for k, v in sdict.items():
            data_dict[k].append(v)
    return data_dict


def get_faust_set(mode, data_root, num_eigenbasis, augments):
    from data.faustscape import ShapeDataset, ShapePairDataset
    from data.faustscape import FAUST_TRAIN_IDX, FAUST_TEST_IDX

    tsfm = get_transforms(augments)
    dset = ShapeDataset(data_root=data_root,
                        data_name='FAUST_r',
                        num_eigenbasis=num_eigenbasis,
                        transforms=tsfm)
    if mode.startswith('train'):
        dset = Subset(dset, FAUST_TRAIN_IDX)
    elif mode.startswith('test'):
        dset = Subset(dset, FAUST_TEST_IDX)
    else:
        raise NotImplementedError
    dset = ShapePairDataset(mode, dset)
    return dset


def get_scape_set(mode, data_root, num_eigenbasis, augments):
    from data.faustscape import ShapeDataset, ShapePairDataset
    from data.faustscape import SCAPE_TRAIN_IDX, SCAPE_TEST_IDX

    tsfm = get_transforms(augments)
    dset = ShapeDataset(data_root=data_root,
                        data_name='SCAPE_r',
                        num_eigenbasis=num_eigenbasis,
                        transforms=tsfm)
    if mode.startswith('train'):
        dset = Subset(dset, SCAPE_TRAIN_IDX)
    elif mode.startswith('test'):
        dset = Subset(dset, SCAPE_TEST_IDX)
    else:
        raise NotImplementedError
    dset = ShapePairDataset(mode, dset)
    return dset


def get_shrec19_set(mode, data_root, num_eigenbasis, augments):
    from data.shrec19 import ShapeDataset, ShapePairDataset

    tsfm = get_transforms(augments)
    dset = ShapeDataset(data_root=data_root,
                        data_name='SHREC_r',
                        num_eigenbasis=num_eigenbasis,
                        transforms=tsfm)
    dset = ShapePairDataset(mode, dset)
    return dset


def get_smal_sub_inter_set(mode, data_root, num_eigenbasis, augments):
    from data.smal import ShapeDataset
    from data.smal import ShapePairDataset

    tsfm = get_transforms(augments)
    data_root = osp.join(data_root, 'smal')
    mode = f'{mode}_sub_inter'
    if mode.startswith('train'):
        data_name = 'train'
        # 0-felidae(cats); 1-canidae(dogs); 3-bovidae(cows);
        categories = ['0', '1', '3']
    elif mode.startswith('test'):
        data_name = 'test'
        # 2-equidae(horses); 4-hippopotamidae(hippos);
        categories = ['2', '4']
    else:
        raise NotImplementedError
    dset = ShapeDataset(data_root=data_root,
                        data_name=data_name,
                        num_eigenbasis=num_eigenbasis,
                        transforms=tsfm)
    indices = [idx for idx in range(len(dset)) if dset[idx]['id'][0] in categories]
    dset = Subset(dset, indices)
    dset = ShapePairDataset(mode, dset)
    return dset


DATASETS = {
    'faust': get_faust_set,
    'scape': get_scape_set,
    'shrec19': get_shrec19_set,
    'smal_sub_inter': get_smal_sub_inter_set,
}

DATASETDIRS = {
    'faust': 'FAUST_r',
    'scape': 'SCAPE_r',
    'shrec19': 'SHREC_r',
    'smal_sub_inter': 'smal',
}


def get_data_loader(data_names, mode, data_root, num_eigenbasis, augments, batch_size,
                    num_workers):
    if isinstance(data_names, (list, tuple)):
        dset = ConcatDataset([
            DATASETS[dname](mode, data_root, num_eigenbasis, augments) for dname in data_names
        ])
    elif isinstance(data_names, str):
        dset = DATASETS[data_names](mode, data_root, num_eigenbasis, augments)
    else:
        raise NotImplementedError

    loader = DataLoader(
        dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=mode.startswith('train'),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def prepare_batch(data_dict, device):
    for k in data_dict.keys():
        if k.startswith('gradX') or \
           k.startswith('gradY') or \
           k.startswith('L'):
            data_dict[k] = torch.stack([sparse_np_to_torch(st) for st in data_dict[k]],
                                       dim=0).to(device)
        else:
            if isinstance(data_dict[k][0], np.ndarray):
                data_dict[k] = torch.from_numpy(np.stack(data_dict[k], axis=0)).to(device)
    return data_dict
