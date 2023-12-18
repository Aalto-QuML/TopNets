import torch.nn.functional as F
from torch import nn

from layers.gin_layer import GINLayer
from models.gnn_factory.gnn_factory_interface import GNNFactoryInterface


class GinCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

    def return_gnn_instance(self, is_last=False):
        return GINLayer(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            activation=nn.Identity() if is_last else F.relu,
            batch_norm=self.batch_norm,
        )
