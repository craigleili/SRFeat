import os
import os.path as osp
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import time
import glob
import shutil
import yaml
from scipy.io import savemat
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

os.environ['WANDB_MODE'] = 'offline'

ROOT_DIR = osp.abspath(osp.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data import get_data_loader, prepare_batch
from utils.loss_contrast import NCELoss
from utils.loss_dirichlet import DirichletLoss
from utils.io import may_create_folder
from utils.misc import omegaconf_to_dotdict, validate_gradient, to_numpy
from utils.misc import KNNSearch, run_trainer


class Trainer(object):

    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

        if cfg.model.name == 'DiffusionNet':
            from diffusion_net.layers import DiffusionNet

            model = DiffusionNet(
                C_in=cfg.model.in_channels,
                C_out=cfg.model.out_channels,
                C_width=cfg.model.block_width,
                N_block=cfg.model.num_blocks,
                dropout=cfg.model.dropout,
            )
        else:
            raise NotImplementedError
        self.model = model.to(self.device)

        self.dataloaders = dict()

    def _init_train(self, phase='train'):
        cfg = self.cfg

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        log_dir_prefix = '_'.join(cfg.data.train_type)
        cfg.log_dir = osp.join(cfg.log_dir, f'SRFeat-D_dfn_{log_dir_prefix}_{exp_time}')
        may_create_folder(cfg.log_dir)

        contrast_type = cfg.loss.contrast_type
        if contrast_type == 'nce':
            self.loss_contrast = NCELoss(cfg.loss.num_pos)
        else:
            raise NotImplementedError

        self.loss_dirichlet = DirichletLoss(normalize=cfg.loss.dirichlet_normalize)

        self.start_epoch = 1

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=cfg.optim.lr,
                                          betas=(0.9, 0.99),
                                          weight_decay=cfg.optim.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=cfg.optim.decay_step,
                                                         gamma=cfg.optim.decay_gamma)

        data_names = OmegaConf.to_container(cfg.data.train_type)
        augments = OmegaConf.to_container(cfg.data.train_augments)
        self.dataloaders[phase] = get_data_loader(data_names,
                                                  mode=phase,
                                                  data_root=cfg.data.data_root,
                                                  num_eigenbasis=cfg.model.num_eigenbasis,
                                                  augments=augments,
                                                  batch_size=cfg.data.batch_size,
                                                  num_workers=cfg.data.num_workers)

        wandb.init(project=cfg.project,
                   dir=cfg.log_dir,
                   group=cfg.group,
                   notes=exp_time,
                   tags=[exp_time[:8], contrast_type, 'dirichlet', cfg.model.name] + data_names,
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(omegaconf_to_dotdict(cfg))

        if cfg.train_ckpt != '':
            self._load_ckpt(cfg.train_ckpt)

    def _init_test(self, phase='test'):
        cfg = self.cfg

        if cfg.skip_train:
            assert cfg.test_ckpt != ''
            cfg.log_dir = str(Path(cfg.test_ckpt).parent)
        else:
            assert cfg.log_dir != ''
            may_create_folder(cfg.log_dir)

        data_names = OmegaConf.to_container(cfg.data.test_type)
        augments = OmegaConf.to_container(cfg.data.test_augments)
        for data_name in data_names:
            self.dataloaders[f'{phase}_{data_name}'] = get_data_loader(
                [data_name],
                mode=phase,
                data_root=cfg.data.data_root,
                num_eigenbasis=cfg.model.num_eigenbasis,
                augments=augments,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers)

        if cfg.test_ckpt != '':
            self._load_ckpt(cfg.test_ckpt)

    def _train_epoch(self, epoch, phase='train'):
        cfg = self.cfg

        num_iter = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])

        self.model.train()
        self.optimizer.zero_grad()
        pbar = tqdm(range(num_iter), miniters=int(num_iter / 100))
        for iter_idx in pbar:
            global_step = (epoch - 1) * num_iter + (iter_idx + 1)

            def get_weight(val, silent_steps):
                return val if global_step >= silent_steps else 0

            batch_data = loader_iter.next()
            batch_data = prepare_batch(batch_data, self.device)

            all_feats = list()
            for pidx in range(2):
                feats = self.model(x_in=batch_data[f'vertices{pidx}'].float(),
                                   mass=batch_data[f'mass{pidx}'].float(),
                                   L=None,
                                   evals=batch_data[f'evals{pidx}'].float(),
                                   evecs=batch_data[f'evecs{pidx}'].float(),
                                   gradX=batch_data[f'gradX{pidx}'],
                                   gradY=batch_data[f'gradY{pidx}'],
                                   faces=batch_data[f'faces{pidx}'].long())
                all_feats.append(feats)
            feats0, feats1 = all_feats

            loss_dict_c = self.loss_contrast(feats0, feats1, batch_data['corr'].long())

            loss_dict_s0 = self.loss_dirichlet(feats0,
                                               L=batch_data['L0'],
                                               evecs=batch_data['evecs0'].float(),
                                               evals=batch_data['evals0'].float(),
                                               mass=batch_data['mass0'].float())
            loss_dict_s1 = self.loss_dirichlet(feats1,
                                               L=batch_data['L1'],
                                               evecs=batch_data['evecs1'].float(),
                                               evals=batch_data['evals1'].float(),
                                               mass=batch_data['mass1'].float())

            loss_dict_s = {
                k: 0.5 * (loss_dict_s0[k] + loss_dict_s1[k]) for k in loss_dict_s0.keys()
            }

            loss_dict = {**loss_dict_c, **loss_dict_s}

            loss = cfg.loss.contrast_weight * loss_dict['contrast'] + \
                   get_weight(cfg.loss.dirichlet_weight, cfg.loss.dirichlet_silent_steps) * \
                   loss_dict['dirichlet']

            if (iter_idx + 1) % cfg.log_step == 0 or (iter_idx + 1) == num_iter:
                log_dict = {
                    'global_step': global_step,
                    'loss': loss.item(),
                }
                for k, v in loss_dict.items():
                    log_dict[f'loss_{k}'] = v.item()
                wandb.log(log_dict)

            loss /= float(cfg.loss.accum_step)
            loss.backward()

            pbar.set_description(f'{phase} epoch:{epoch} loss:{loss.item():.5f}')

            if (iter_idx + 1) % cfg.loss.accum_step == 0 or (iter_idx + 1) == num_iter:
                if validate_gradient(self.model):
                    self.optimizer.step()
                else:
                    print('[!] Invalid gradients')
                self.optimizer.zero_grad()

    def _test_epoch(self, epoch=None, phase='test'):
        cfg = self.cfg
        l2norm = True

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        out_root = osp.join(cfg.log_dir, f'{phase}_{exp_time}')
        may_create_folder(out_root)

        num_iter = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])

        if cfg.eval_mode:
            self.model.eval()
        else:
            self.model.train()
        for iter_idx in tqdm(range(num_iter), miniters=int(num_iter / 100), desc=phase):
            batch_data = loader_iter.next()
            batch_data = prepare_batch(batch_data, self.device)

            with torch.no_grad():
                all_feats = list()
                for pidx in range(2):
                    feats = self.model(x_in=batch_data[f'vertices{pidx}'].float(),
                                       mass=batch_data[f'mass{pidx}'].float(),
                                       L=None,
                                       evals=batch_data[f'evals{pidx}'].float(),
                                       evecs=batch_data[f'evecs{pidx}'].float(),
                                       gradX=batch_data[f'gradX{pidx}'],
                                       gradY=batch_data[f'gradY{pidx}'],
                                       faces=batch_data[f'faces{pidx}'].long())
                    all_feats.append(feats)
                feats0, feats1 = all_feats
                if l2norm:
                    feats0 = F.normalize(feats0, p=2, dim=-1)
                    feats1 = F.normalize(feats1, p=2, dim=-1)

            feats0 = to_numpy(torch.squeeze(feats0, 0))
            feats1 = to_numpy(torch.squeeze(feats1, 0))
            id0 = batch_data['id0'][0]
            id1 = batch_data['id1'][0]

            knn_search = KNNSearch(feats0)
            pmap10 = knn_search.query(feats1, 1)

            to_save = {
                'id0': id0,
                'id1': id1,
                'pmap10': pmap10,
            }
            savemat(osp.join(out_root, f'{id0}-{id1}.mat'), to_save)

    def train(self):
        cfg = self.cfg

        print('Start training')
        self._init_train()

        for epoch in range(self.start_epoch, cfg.optim.epochs + 1):
            print(f'Epoch: {epoch}, LR = {self.scheduler.get_last_lr()}')
            self._train_epoch(epoch)
            self.scheduler.step()
            self._save_ckpt(epoch, 'latest')

        print('Training finished')

    def test(self):
        cfg = self.cfg

        print('Start testing')
        self._init_test()

        for mode in self.dataloaders.keys():
            if mode.startswith('test'):
                self._test_epoch(phase=mode)

        print('Testing finished')

    def _save_ckpt(self, epoch, name=None):
        cfg = self.cfg

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'cfg': cfg,
        }
        if name is None:
            filepath = osp.join(cfg.log_dir, f'ckpt_epoch_{epoch}.pth')
        else:
            filepath = osp.join(cfg.log_dir, f'ckpt_{name}.pth')
        torch.save(state, filepath)
        print(f'Saved checkpoint to {filepath}')

    def _load_ckpt(self, filepath, only_weights=True):
        if Path(filepath).is_file():
            state = torch.load(filepath)
            self.model.load_state_dict(state['state_dict'])
            if not only_weights:
                self.start_epoch = state['epoch']
                self.optimizer.load_state_dict(state['optimizer'])
                self.scheduler.load_state_dict(state['scheduler'])
            print(f'Loaded checkpoint from {filepath}')
        else:
            raise RuntimeError(f'Checkpoint {filepath} does not exist!')


if __name__ == '__main__':
    run_trainer(Trainer)
