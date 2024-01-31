import torch
import torch.nn as nn
from torch_scatter import scatter


class DeepSetLayer0(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

        self.external_mlp = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x0, batch):
        x0 = self.mlp(x0)
        x = scatter(x0, batch, dim=0, reduce="mean")
        x = self.external_mlp(x)
        return x


class DeepSetLayer1(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

        self.external_mlp = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x1, edge_slices, mask=None):
        edge_diff_slices = (edge_slices[1:] - edge_slices[:-1]).to(x1.device)
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(
            torch.arange(n_batch, device=x1.device), edge_diff_slices
        )
        if mask is not None:
            batch_e = batch_e[mask]

        x1 = self.mlp(x1)
        x1 = scatter(x1, batch_e, dim=0, reduce="mean", dim_size=n_batch)
        x1 = self.external_mlp(x1)
        return x1
