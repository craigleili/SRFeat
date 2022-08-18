import os.path as osp
import sys
import copy
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from diffusion_net.geometry import get_operators
from diffusion_net.utils import sparse_torch_to_np
from data.utils import load_shape
from utils.io import list_files
from utils.misc import to_numpy


class ShapeDataset(Dataset):

    def __init__(self,
                 data_root,
                 data_name,
                 num_eigenbasis=128,
                 transforms=None,
                 cache_dir='cache_DiffusionNet_a61ef8',
                 shape_dir_name='shapes',
                 shape_suffix='off'):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.num_eigenbasis = num_eigenbasis
        self.transforms = transforms
        self.cache_dir = cache_dir
        self.shape_dir_name = shape_dir_name
        self.shape_suffix = shape_suffix
        cache_root = osp.join(data_root, data_name, cache_dir)

        print(f'[*] Loading {data_name} data')

        self.shape_list = list()

        shape_dir = osp.join(data_root, data_name, shape_dir_name)
        filelist = list_files(shape_dir, f'*.{shape_suffix}', alphanum_sort=True)
        for filename in filelist:
            sid = filename[:-4]
            vertices_np, faces_np, vertex_normals_np = load_shape(osp.join(shape_dir, filename),
                                                                  True)
            shape_dict = dict(
                type=data_name,
                id=sid,
                vertices=vertices_np,
                faces=faces_np,
            )

            if num_eigenbasis > 0:
                vertices_th = torch.from_numpy(vertices_np)
                faces_th = torch.from_numpy(faces_np).long()
                vertex_normals_th = torch.from_numpy(vertex_normals_np)

                frames_th, mass_th, L_th, evals_th, evecs_th, gradX_th, gradY_th = get_operators(
                    vertices_th,
                    faces_th,
                    k_eig=num_eigenbasis,
                    op_cache_dir=cache_root,
                    normals=vertex_normals_th,
                )
                shape_dict['frames'] = to_numpy(frames_th)
                shape_dict['mass'] = to_numpy(mass_th)
                shape_dict['L'] = sparse_torch_to_np(L_th)
                shape_dict['evals'] = to_numpy(evals_th)
                shape_dict['evecs'] = to_numpy(evecs_th)
                shape_dict['gradX'] = sparse_torch_to_np(gradX_th)
                shape_dict['gradY'] = sparse_torch_to_np(gradY_th)

            self.shape_list.append(shape_dict)

    def __getitem__(self, item):
        data = copy.deepcopy(self.shape_list[item])
        if self.transforms is not None:
            vertices_np = data['vertices']
            faces_np = data['faces']
            vertices_np, faces_np = self.transforms(vertices_np, faces_np)
            data['vertices'] = vertices_np
            data['faces'] = faces_np

        return data

    def __len__(self):
        return len(self.shape_list)


FAUST_TRAIN_IDX, FAUST_TEST_IDX = np.arange(0, 80), np.arange(80, 100)
SCAPE_TRAIN_IDX, SCAPE_TEST_IDX = np.arange(0, 51), np.arange(51, 71)


class ShapePairDataset(Dataset):

    def __init__(self, mode, data, **kwargs):
        super().__init__()
        self.mode = mode
        self.data = data
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.data_root, self.data_name = self._get_meta()

        self.index_map = dict()
        for idx in range(len(data)):
            shape_dict = data[idx]
            self.index_map[str(shape_dict['id'])] = idx

        self.pair_indices = list(itertools.combinations(range(len(data)), 2))

    def _get_meta(self):
        if isinstance(self.data, ShapeDataset):
            data = self.data
        elif isinstance(self.data, Subset):
            data = self.data.dataset
            assert isinstance(data, ShapeDataset)
        else:
            raise NotImplementedError
        return data.data_root, data.data_name

    def _load_corr(self, sid):
        corr_path = osp.join(self.data_root, self.data_name, 'correspondences', f'{sid}.vts')
        corr = np.loadtxt(corr_path, dtype=np.int32)
        return corr - 1

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

        corr0 = self._load_corr(shape_dict0['id'])
        corr1 = self._load_corr(shape_dict1['id'])
        data_dict['corr'] = np.stack((corr0, corr1), axis=1)

        return data_dict

    def __len__(self):
        return len(self.pair_indices)
