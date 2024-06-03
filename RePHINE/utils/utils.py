import torch
from torch_scatter import scatter
import os
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataloader
import pandas as pd
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from joblib import delayed
from sklearn.model_selection import StratifiedShuffleSplit
from torch_cluster import knn_graph
import os.path as osp
import pickle
import sys
from ogb.graphproppred import PygGraphPropPredDataset



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

#from tu import TUData
def load_ogb_graph_dataset(root, name):
    raw_dir = osp.join(root, 'raw')
    dataset = PygGraphPropPredDataset(name, raw_dir)
    idx = dataset.get_idx_split()

    return dataset, idx['train'], idx['valid'], idx['test']


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
