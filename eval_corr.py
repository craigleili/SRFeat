import argparse
import os
import os.path as osp
import sys
import numpy as np
import pickle
import multiprocessing
from joblib import Parallel, delayed
from scipy.io import loadmat, savemat
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data import DATASETS, DATASETDIRS
from utils.fmap import p2p_to_FM, zoomout_refine
from utils.io import list_files, list_folders, may_create_folder, read_lines
from data.utils import load_geodist

# https://github.com/LIX-shape-analysis/GeomFmaps/tree/master/eval_scripts


def load_pair_predictions(filepath):
    data = loadmat(filepath)
    pmap10 = np.asarray(data['pmap10'], dtype=np.int32)
    pmap10 = np.squeeze(pmap10)
    return pmap10


def read_csv(filepath, num_lines=None):
    ret = dict()
    lines = read_lines(filepath)
    for idx, line in enumerate(lines):
        if num_lines is not None and idx >= num_lines:
            break
        items = line.split(',')
        ret[items[0]] = float(items[1])
    return ret


def run_exp(cfg, test_root, out_root):
    exp_name = Path(test_root).name
    data_type = exp_name[5:][:-18]
    data_dir = DATASETDIRS[data_type]
    mode = 'test'
    if data_type.startswith('smal'):
        data_dir = f'{data_dir}/test'
    elif data_type.startswith('shrec16'):
        data_dir = f'{data_dir}/null'

    if Path(out_root).is_dir() and not Path(out_root + '.pkl').is_file():
        return

    if Path(out_root + '.pkl').is_file():
        print(f'[*] {exp_name} already evaluated: load from pkl...')
        with open(out_root + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
            all_pair_ids = saved['pair_ids']
            all_geoerrs = saved['geoerrs']
            all_geoerrs_ref = saved['geoerrs_ref']
    else:
        pair_filenames = list_files(test_root, '*.mat', alphanum_sort=True)
        if len(pair_filenames) == 0:
            return
        dset = DATASETS[data_type](mode=mode,
                                   data_root=cfg.data_root,
                                   num_eigenbasis=128,
                                   augments=['centering'])

        SD = cfg.spectral_dim

        may_create_folder(out_root)

        all_pair_ids = list()
        all_geoerrs = list()
        all_geoerrs_ref = list()
        for pid in range(len(pair_filenames)):
            if pid == 0 or (pid + 1) % 10 == 0:
                print(f'Processing {exp_name} : {data_type} {pid + 1}/{len(pair_filenames)}')

            pair_filename = pair_filenames[pid]
            id0, id1 = pair_filename[:-4].split('-')

            pair_dict = dset.get_pair_by_ids(id0, id1)
            evecs0 = pair_dict['evecs0']
            evecs1 = pair_dict['evecs1']
            num_verts0 = evecs0.shape[0]
            num_verts1 = evecs1.shape[0]

            geodist0, sqrt_area0 = load_geodist(
                osp.join(cfg.data_root, data_dir, 'geodist', '{}.mat'.format(id0)))

            pmap10 = load_pair_predictions(osp.join(test_root, pair_filename))
            fmap01 = p2p_to_FM(pmap10, evecs0[:, :SD], evecs1[:, :SD])

            fmap01_ref, pmap10_ref = zoomout_refine(
                evecs0,
                evecs1,
                fmap01,
                nit=cfg.num_zo_iters,
                step=(cfg.max_spectral_dim - SD) // cfg.num_zo_iters,
                return_p2p=True,
            )

            corr0 = pair_dict['corr'][:, 0]
            corr1 = pair_dict['corr'][:, 1]

            match010 = np.stack([corr0, pmap10[corr1]], axis=-1)
            match010 = np.ravel_multi_index(match010.T, dims=[num_verts0, num_verts0])

            match010_ref = np.stack([corr0, pmap10_ref[corr1]], axis=-1)
            match010_ref = np.ravel_multi_index(match010_ref.T, dims=[num_verts0, num_verts0])

            geoerrs = np.take(geodist0, match010) / sqrt_area0
            geoerrs_ref = np.take(geodist0, match010_ref) / sqrt_area0

            geoerrs = np.squeeze(geoerrs)
            geoerrs_ref = np.squeeze(geoerrs_ref)

            all_geoerrs.append(geoerrs)
            all_geoerrs_ref.append(geoerrs_ref)
            all_pair_ids.append((id0, id1))

            to_save = {
                'matches': np.asarray(pmap10, dtype=np.int32),
                'matches_ref': np.asarray(pmap10_ref, dtype=np.int32),
            }
            savemat(osp.join(out_root, '{}.mat'.format(pair_filename[:-4])), to_save)

        with open(out_root + '.pkl', 'wb') as fh:
            to_save = {
                'pair_ids': all_pair_ids,
                'geoerrs': all_geoerrs,
                'geoerrs_ref': all_geoerrs_ref,
            }
            pickle.dump(to_save, fh)

    all_geoerrs = np.concatenate(all_geoerrs)
    all_geoerrs_ref = np.concatenate(all_geoerrs_ref)
    with open(out_root + '.csv', 'w') as fh:
        fh.write('MeanGeoErr,{:.4f}\n'.format(np.mean(all_geoerrs)))
        fh.write('MeanGeoErrRef,{:.4f}\n'.format(np.mean(all_geoerrs_ref)))

        fh.write('StdGeoErr,{:.4f}\n'.format(np.std(all_geoerrs)))
        fh.write('StdGeoErrRef,{:.4f}\n'.format(np.std(all_geoerrs_ref)))

        fh.write('MaxGeoErr,{:.4f}\n'.format(np.amax(all_geoerrs)))
        fh.write('MaxGeoErrRef,{:.4f}\n'.format(np.amax(all_geoerrs_ref)))

        fh.write('MedianGeoErr,{:.4f}\n'.format(np.median(all_geoerrs)))
        fh.write('MedianGeoErrRef,{:.4f}\n'.format(np.median(all_geoerrs_ref)))

        fh.write('MinGeoErr,{:.4f}\n'.format(np.amin(all_geoerrs)))
        fh.write('MinGeoErrRef,{:.4f}\n'.format(np.amin(all_geoerrs_ref)))


def run_model(cfg, model_root):
    if not Path(model_root).is_dir():
        return

    print(f'Evaluating {Path(model_root).name}')

    for folder_name in list_folders(model_root):
        if not folder_name.startswith('test_'):
            continue
        if folder_name.endswith('_eval'):
            continue
        test_root = osp.join(model_root, folder_name)
        out_root = test_root + '_eval'
        run_exp(cfg, test_root, out_root)


def run(cfg):
    num_threads = min(len(cfg.test_roots), 3)
    Parallel(n_jobs=num_threads)(
        delayed(run_model)(cfg, test_root) for test_root in cfg.test_roots)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--test_roots', nargs='+')
    parser.add_argument('--num_zo_iters', type=int, default=10)
    parser.add_argument('--spectral_dim', type=int, default=30)
    parser.add_argument('--max_spectral_dim', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_arguments()
    run(cfg)
