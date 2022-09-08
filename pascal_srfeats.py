import argparse
import os
import os.path as osp
import sys
import shutil
import random
import pickle
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import PascalVOCKeypoints
from torch_geometric.data import DataLoader
import robust_laplacian
import wandb

os.environ['WANDB_MODE'] = 'offline'

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dgmc.utils import ValidPairDataset
from dgmc.models.dgmc_spectral import DGMC
from dgmc.models.spline import SplineCNN
from loss import SoftFmapLoss


class PascalVOC(PascalVOCKeypoints):

    def download(self):
        from torch_geometric.data import download_url, extract_tar

        path = download_url(self.image_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        #os.unlink(path)
        image_path = osp.join(self.raw_dir, 'TrainVal', 'VOCdevkit', 'VOC2011')
        os.rename(image_path, osp.join(self.raw_dir, 'images'))
        shutil.rmtree(osp.join(self.raw_dir, 'TrainVal'))

        path = download_url(self.annotation_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        #os.unlink(path)

        path = download_url(self.split_url, self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, 'splits.npz'))

    def process(self):
        super().process()


class LaplacianBasis(T.BaseTransform):

    def __init__(self, k_eig, eps=1e-6, normalize_pos=False):
        self.k_eig = k_eig
        self.eps = eps
        self.normalize_pos = normalize_pos

    def __call__(self, data):
        if self.k_eig < 1:
            return data

        dtype = data.pos.dtype
        device = data.pos.device
        num_points = data.pos.shape[0]
        assert data.pos.shape[1] == 2
        if num_points <= self.k_eig:
            data.has_spectral = False
            data.mass = torch.zeros(num_points, dtype=dtype).to(device)
            data.evals = torch.zeros(self.k_eig, dtype=dtype).to(device)
            data.evecs = torch.zeros(num_points, self.k_eig,
                                     dtype=dtype).to(device)
            return data

        assert hasattr(data, 'face') and data.face is not None
        assert data.face.shape[0] == 3
        face = data.face.t().cpu().numpy()

        pos = data.pos.cpu().numpy()
        if self.normalize_pos:
            pos = pos - np.mean(pos, axis=0, keepdims=True)
            pos_norm = np.linalg.norm(pos, axis=1)
            assert pos_norm.ndim == 1 and pos_norm.shape[0] == pos.shape[0]
            pos /= np.amax(pos_norm)

        pos = np.concatenate((pos, np.zeros((num_points, 1), dtype=pos.dtype)),
                             axis=1)

        L, M = robust_laplacian.mesh_laplacian(pos, face)
        eps_L = scipy.sparse.identity(num_points) * self.eps
        eps_M = scipy.sparse.identity(num_points) * self.eps * np.sum(M)
        massvec_np = M.diagonal()

        evals_np, evecs_np = sla.eigsh(L + eps_L,
                                       self.k_eig,
                                       M + eps_M,
                                       sigma=1e-8)

        data.has_spectral = True
        data.mass = torch.from_numpy(massvec_np).float().to(data.pos.device)
        data.evals = torch.from_numpy(evals_np).float().to(data.pos.device)
        data.evecs = torch.from_numpy(evecs_np).float().to(data.pos.device)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def get_pred(S, y):
    if not S.is_sparse:
        pred = S[y[0]].argmax(dim=-1)
    else:
        assert S.__idx__ is not None and S.__val__ is not None
        pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]
    return pred


parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, default='exp/data/PascalVOC2011')
parser.add_argument('--test_root', type=str, default='exp/data/PascalVOC2011')
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--spectral_type', type=str, default='soft_fmap')
parser.add_argument('--spectral_weight', type=float, default=0.01)
parser.add_argument('--spectral_dim', type=int, default=7)
parser.add_argument('--spectral_max_val', type=float, default=1000)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

wandb_tags = ['dgmc']
if args.spectral_weight > 0:
    wandb_tags.append(args.spectral_type)
wandb.init(project='DeepGraphMatchingConsensus',
           dir='exp/log',
           group='pascal',
           tags=wandb_tags,
           settings=wandb.Settings(start_method='fork'))
wandb.config.update(args)

pre_filter = lambda data: data.pos.size(0) > 0
transform = T.Compose([
    T.Delaunay(),
    LaplacianBasis(args.spectral_dim, normalize_pos=False),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
for category in PascalVOC.categories:
    dataset = PascalVOC(args.train_root,
                        category,
                        train=True,
                        transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(args.test_root,
                        category,
                        train=False,
                        transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset,
                          args.batch_size,
                          shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(dataset.num_node_features,
                  args.dim,
                  dataset.num_edge_features,
                  args.num_layers,
                  cat=False,
                  dropout=0.5)
psi_2 = SplineCNN(args.rnd_dim,
                  args.rnd_dim,
                  dataset.num_edge_features,
                  args.num_layers,
                  cat=True,
                  dropout=0.0)
model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train():
    model.train()

    if args.spectral_weight > 0:
        if args.spectral_type == 'soft_fmap':
            spectral_criterion = SoftFmapLoss(
                spectral_dims=[args.spectral_dim],
                max_val=args.spectral_max_val)
        else:
            raise NotImplementedError
    else:
        spectral_criterion = None

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        S_0, S_L, h_s, h_t = model(data.x_s, data.edge_index_s,
                                   data.edge_attr_s, data.x_s_batch, data.x_t,
                                   data.edge_index_t, data.edge_attr_t,
                                   data.x_t_batch)
        y = generate_y(data.y)

        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss_dgmc = loss

        if args.spectral_weight > 0:
            loss_spectral = spectral_criterion(
                data.evecs_s,
                data.evals_s,
                data.mass_s,
                data.has_spectral_s,
                h_s,
                data.x_s_batch,
                data.evecs_t,
                data.evals_t,
                data.mass_t,
                data.has_spectral_t,
                h_t,
                data.x_t_batch,
                y,
            )
            loss = loss + args.spectral_weight * loss_spectral
        else:
            loss_spectral = torch.as_tensor(0)

        wandb.log({
            'loss': loss.item(),
            'loss_dgmc': loss_dgmc.item(),
            'loss_spectral': loss_spectral.item()
        })

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(dataset):
    model.eval()

    loader = DataLoader(dataset,
                        args.batch_size,
                        shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    predictions = list()
    groundtruths = list()
    num_examples = 0
    while (num_examples < args.test_samples):
        for data in loader:
            data = data.to(device)

            S_0, S_L, h_s, h_t = model(data.x_s, data.edge_index_s,
                                       data.edge_attr_s, data.x_s_batch,
                                       data.x_t, data.edge_index_t,
                                       data.edge_attr_t, data.x_t_batch)
            y = generate_y(data.y)

            pred = get_pred(S_L, y)
            assert pred.dim() == 1
            for bidx in range(args.batch_size):
                flags_s = (data.x_s_batch == bidx)
                flags_t = (data.x_t_batch == bidx)

                if torch.sum(flags_s) < 1 or torch.sum(flags_t) < 1:
                    continue

                predictions.append(pred[flags_s].cpu().numpy())
                groundtruths.append(y[1][flags_s].cpu().numpy())
                num_examples += torch.sum(flags_s)

            if num_examples >= args.test_samples:
                return predictions, groundtruths


all_result_list = list()
for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    result_dict = dict()
    for category, test_dataset in zip(PascalVOC.categories, test_datasets):
        result_dict[category] = test(test_dataset)
    all_result_list.append(result_dict)

    for min_num_nodes in [0]:
        all_accu = list()
        for category in PascalVOC.categories:
            correct, count = 0, 0
            for pred, gt in zip(*result_dict[category]):
                if len(gt) < min_num_nodes:
                    continue
                correct += np.sum(pred == gt)
                count += len(gt)
            if count == 0:
                all_accu.append(0)
            else:
                all_accu.append(correct / count)
        all_accu = np.asarray(all_accu) * 100

        print(' '.join([c[:5].ljust(5)
                        for c in PascalVOC.categories] + ['mean']))
        flags = all_accu > 0
        accu_list = all_accu.tolist() + [np.mean(all_accu[flags])]
        print(' '.join([f'{accu:.1f}'.ljust(5) for accu in accu_list]))
    print('\n')

torch.save({'state_dict': model.state_dict()},
           osp.join(wandb.run.dir, 'ckpt.pth'))
with open(osp.join(wandb.run.dir, 'run_log.pkl'), 'wb') as fh:
    to_save = {'log': all_result_list}
    pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)
