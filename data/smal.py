import os.path as osp
import sys
import copy
import random
import numpy as np
from torch.utils.data import Dataset, Subset
from pathlib import Path
from collections import defaultdict

try:
    import pickle5 as pickle
except ImportError:
    import pickle

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.faustscape import ShapeDataset

# https://smal.is.tue.mpg.de/


def load_inter_pairs(cache_path=None):
    with open(cache_path, 'rb') as fh:
        saved = pickle.load(fh)
        pairs = saved['pairs']
    print(f'[*] Load inter pairs from {cache_path}')

    return pairs


class ShapePairDataset(Dataset):

    def __init__(self, mode, data, **kwargs):
        super().__init__()
        self.mode = mode
        self.data = data
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.data_root, self.data_name = self._get_meta()

        self.index_map = dict()
        class_dict = defaultdict(list)
        for idx in range(len(data)):
            sid = data[idx]['id']
            self.index_map[str(sid)] = idx
            class_dict[sid.split('_')[0]].append(sid)

        cache_path = osp.join(self.data_root, self.data_name, f'{mode}_pairs.pkl')
        self.pair_indices = load_inter_pairs(cache_path)

    def _get_meta(self):
        if isinstance(self.data, ShapeDataset):
            data = self.data
        elif isinstance(self.data, Subset):
            data = self.data.dataset
            assert isinstance(data, ShapeDataset)
        else:
            raise NotImplementedError
        return data.data_root, data.data_name

    def __getitem__(self, item):
        pidx = self.pair_indices[item]
        return self.get_pair_by_ids(pidx[0], pidx[1])

    def get_pair_by_ids(self, sid0, sid1):
        shape_dict0 = self.data[self.index_map[sid0]]
        shape_dict1 = self.data[self.index_map[sid1]]
        return self._prepare_pair(shape_dict0, shape_dict1)

    def _prepare_pair(self, shape_dict0, shape_dict1):
        data_dict = dict()
        for idx, sdict in enumerate([shape_dict0, shape_dict1]):
            for k in sdict.keys():
                data_dict[f'{k}{idx}'] = sdict[k]

        num_points = len(shape_dict0['vertices'])
        assert num_points == len(shape_dict1['vertices'])
        data_dict['corr'] = np.stack((np.arange(num_points), np.arange(num_points)), axis=1)

        return data_dict

    def __len__(self):
        return len(self.pair_indices)
