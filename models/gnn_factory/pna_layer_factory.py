import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool

from layers.gin_layer import GINLayer
from layers.graph_convolution_layer import GCNLayer
from models.gnn_factory.gnn_factory_interface import GNNFactoryInterface


class PnaCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm, deg=None):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.deg = deg

    def return_gnn_instance(self, is_last=False):
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        return PNAConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=self.deg,
            towers=int(self.hidden_dim / 16),
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
