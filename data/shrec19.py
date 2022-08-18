import os.path as osp
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.faustscape import ShapeDataset
from utils.io import list_files


def load_corr(filepath):
    corr = np.loadtxt(filepath, dtype=np.int32)
    return corr - 1


class ShapePairDataset(Dataset):

    def __init__(self, mode, data, **kwargs):
        super().__init__()
        self.mode = mode
        self.data = data
        for k, w in kwargs.items():
            setattr(self, k, w)

        assert mode.startswith('test')

        self.data_root, self.data_name = self._get_meta()

        self.index_map = dict()
        for idx in range(len(data)):
            shape_dict = data[idx]
            self.index_map[str(shape_dict['id'])] = idx

        self.pair_indices = list()
        corr_root = osp.join(self.data_root, self.data_name, 'correspondences')
        corr_filenames = list_files(corr_root, '*.map', alphanum_sort=True)
        for corr_filename in corr_filenames:
            id0, id1 = corr_filename[:-4].split('_')
            self.pair_indices.append((self.index_map[id1], self.index_map[id0]))

    def _get_meta(self):
        if isinstance(self.data, ShapeDataset):
            data = self.data
        else:
            raise NotImplementedError
        return data.data_root, data.data_name

    def __getitem__(self, item):
        pidx = self.pair_indices[item]
        shape_dict0 = self.data[pidx[0]]
        shape_dict1 = self.data[pidx[1]]
        return self._prepare_pair(shape_dict0, shape_dict1)

    def get_pair_by_ids(self, sid0, sid1):
        shape_dict0 = self.data[self.index_map[sid0]]
        shape_dict1 = self.data[self.index_map[sid1]]
        return self._prepare_pair(shape_dict0, shape_dict1)

    def _prepare_pair(self, shape_dict0, shape_dict1):
        data_dict = dict()
        for idx, sdict in enumerate([shape_dict0, shape_dict1]):
            for k in sdict.keys():
                data_dict[f'{k}{idx}'] = sdict[k]

        corr_path = osp.join(self.data_root, self.data_name, 'correspondences',
                             f"{shape_dict1['id']}_{shape_dict0['id']}.map")
        data_dict['corr'] = np.stack(
            (load_corr(corr_path), np.arange(len(shape_dict1['vertices']))), axis=1)

        return data_dict

    def __len__(self):
        return len(self.pair_indices)
