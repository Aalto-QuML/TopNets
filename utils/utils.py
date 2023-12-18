import torch
from torch_scatter import scatter
import os
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


def remove_duplicate_edges(batch):
    with torch.no_grad():
        batch = batch.clone().detach()

        device = batch.x.device
        # Computing the equivalent of batch over edges.
        #            edge_slices = torch.tensor(batch._slice_dict["edge_index"], device=device)
        edge_slices = batch._slice_dict["edge_index"].clone().detach()
        edge_slices = edge_slices.to(device)

        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(
            torch.arange(n_batch, device=device), edge_diff_slices
        )

        correct_idx = batch.edge_index[0] <= batch.edge_index[1]
        # batch_e_idx = batch_e[correct_idx]

        n_edges = scatter(correct_idx.long(), batch_e, reduce="sum")

        #           batch.edge_index = batch.edge_index[:,correct_idx]

        new_slices = torch.cumsum(
            torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_edges)), 0
        )

        vertex_slice = batch._slice_dict["x"].clone()
        #           batch._slice_dict['edge_index'] = new_slices
        new_edge_index = batch.edge_index[:, correct_idx]

        return new_edge_index, vertex_slice, new_slices, batch.batch


def train_test_split(dataset,train_ratio,val_ratio,batch_size,prop_idx):
    dataset_len = len(dataset)
    y = torch.cat([dataset[i].y[:,prop_idx] for i in range(dataset_len) ])

    global_mean = y.mean()
    global_std = y.std()
    idxs = np.arange(dataset_len,dtype=int)
    idxs = np.random.default_rng(42).permutation(idxs)
    
    train_size = 110000
    val_size = 10000

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : ]

    train_loader = DataLoader(dataset[idx_train],batch_size=batch_size,shuffle=False)
    val_loader = DataLoader(dataset[idx_val],batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(dataset[idx_test],batch_size=batch_size,shuffle=False)

    dataloader = {"train": train_loader,"valid": val_loader, "test":test_loader}
    return dataloader,global_mean,global_std



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")