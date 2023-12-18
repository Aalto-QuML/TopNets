import torch
import torch.nn as nn
import sys
sys.path.append('../')
from ph_cpu import compute_persistence_homology_batched_mt
from rephine_mt import compute_rephine_batched_mt

from layers.graph_equivariant_layers import DeepSetLayer0, DeepSetLayer1
from utils.utils import remove_duplicate_edges


class RephineLayer(nn.Module):
    def __init__(
        self,
        n_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type="rephine",
        dim1=True,
        sig_filtrations=True,
        reduce_tuples=False
    ):
        super().__init__()

        final_filtration_activation = nn.Sigmoid() if sig_filtrations else nn.Identity()

        self.persistence_fn = (
            compute_rephine_batched_mt
            if diagram_type == "rephine"
            else compute_persistence_homology_batched_mt
        )

        self.diagram_type = diagram_type
        self.dim1 = dim1
        self.out_dim = out_dim

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, filtration_hidden),
            nn.ReLU(),
            nn.Linear(filtration_hidden, n_filtrations),
            final_filtration_activation,
        )

        if self.diagram_type == "rephine":
            self.edge_filtrations = nn.Sequential(
                nn.Linear(n_features, filtration_hidden),
                nn.ReLU(),
                nn.Linear(filtration_hidden, n_filtrations),
                final_filtration_activation,
            )

        self.num_filtrations = n_filtrations
        self.reduce_tuples = reduce_tuples
        tuple_size = 3 if self.reduce_tuples else 4
        diagram_size = tuple_size if diagram_type == "rephine" else 2
        self.deepset_fn = DeepSetLayer0(
            in_dim=n_filtrations * diagram_size, out_dim=out_dim
        )

        if dim1:
            self.deepset_fn_dim1 = DeepSetLayer1(in_dim=diagram_size, out_dim=out_dim)

        self.out = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices):
        filtered_v = self.filtrations(x)
        if self.diagram_type == "rephine":
            filtered_e = self.edge_filtrations(
                x[edge_index[0]] + x[edge_index[1]]
            )  # multiple ways of doing this.
        elif self.diagram_type == "standard":
            filtered_e, _ = torch.max(
                torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
                axis=0,
            )

        vertex_slices = vertex_slices.cpu().long()
        edge_slices = edge_slices.cpu().long()
        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0, persistence1 = self.persistence_fn(
            filtered_v, filtered_e, edge_index, vertex_slices, edge_slices
        )

        persistence0 = persistence0.to(x.device)
        persistence1 = persistence1.to(x.device)

        if self.diagram_type == "rephine":
            persistence0 = persistence0[:, :, [0, 2, 1]]
            if not self.reduce_tuples:
                persistence0 = torch.cat(
                    (
                        torch.zeros((persistence0.shape[0], persistence0.shape[1], 1)).to(
                            x.device
                        ),
                        persistence0,
                    ),
                    dim=-1,
                )
                persistence1 = torch.cat(
                    (
                        torch.zeros((persistence1.shape[0], persistence1.shape[1], 1)).to(
                            x.device
                        ),
                        persistence1,
                    ),
                    dim=-1,
                )
                persistence1[persistence1[:, :, 1:].nonzero(as_tuple=True)] = 1.0

            persistence0[persistence0.isnan()] = 1.0

        return persistence0, persistence1

    def forward(self, x, data):
        edge_index, vertex_slices, edge_slices, batch = remove_duplicate_edges(data)

        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices
        )
        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)

        # processing x0
        x0 = self.deepset_fn(x0, batch)
        # processing x1
        if self.dim1:
            x1_list = []
            for i in range(self.num_filtrations):
                pers1_reshaped = pers1[i, :, :].reshape(pers1.shape[1], -1)
                pers1_mask = ~((pers1_reshaped == 0).all(-1))
                x1 = pers1_reshaped[pers1_mask]
                x1_list.append(self.deepset_fn_dim1(x1, edge_slices, mask=pers1_mask))
            x1 = torch.stack(x1_list).mean(dim=0)
            x0 = x0 + x1
        x = x0
        x = self.bn(x)
        x = self.out(x)
        return x



class RephineLayer_Equiv(nn.Module):
    def __init__(
        self,
        n_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type="rephine",
        dim1=True,
        sig_filtrations=True,
        reduce_tuples=False
    ):
        super().__init__()

        final_filtration_activation = nn.Sigmoid() if sig_filtrations else nn.Identity()

        self.persistence_fn = (
            compute_rephine_batched_mt
            if diagram_type == "rephine"
            else compute_persistence_homology_batched_mt
        )

        self.diagram_type = diagram_type
        self.dim1 = dim1
        self.out_dim = out_dim

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, filtration_hidden),
            nn.ReLU(),
            nn.Linear(filtration_hidden, n_filtrations),
            final_filtration_activation,
        )

        if self.diagram_type == "rephine":
            self.edge_filtrations = nn.Sequential(
                nn.Linear(n_features+1, filtration_hidden),
                nn.ReLU(),
                nn.Linear(filtration_hidden, n_filtrations),
                final_filtration_activation,
            )

        self.num_filtrations = n_filtrations
        self.reduce_tuples = reduce_tuples
        tuple_size = 3 if self.reduce_tuples else 4
        diagram_size = tuple_size if diagram_type == "rephine" else 2
        self.deepset_fn = DeepSetLayer0(
            in_dim=n_filtrations * diagram_size, out_dim=out_dim
        )

        if dim1:
            self.deepset_fn_dim1 = DeepSetLayer1(in_dim=diagram_size, out_dim=out_dim)

        self.out = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices,pos):
        filtered_v = self.filtrations(x)
        if self.diagram_type == "rephine":
            # Changed to adopt the position coordinates to become equivariant
            euclid_dist = (pos[edge_index[0]] - pos[edge_index[1]])**2
            euclid_dist = torch.sqrt(euclid_dist.sum(dim=1))
            input_feat = torch.cat([x[edge_index[0]] + x[edge_index[1]],euclid_dist.unsqueeze(dim=1)],dim=1)
            filtered_e = self.edge_filtrations(input_feat)
                #x[edge_index[0]] + x[edge_index[1]]
            #)  # multiple ways of doing this.
        elif self.diagram_type == "standard":
            filtered_e, _ = torch.max(
                torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
                axis=0,
            )

        vertex_slices = vertex_slices.cpu().long()
        edge_slices = edge_slices.cpu().long()
        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0, persistence1 = self.persistence_fn(
            filtered_v, filtered_e, edge_index, vertex_slices, edge_slices
        )

        persistence0 = persistence0.to(x.device)
        persistence1 = persistence1.to(x.device)

        if self.diagram_type == "rephine":
            persistence0 = persistence0[:, :, [0, 2, 1]]
            if not self.reduce_tuples:
                persistence0 = torch.cat(
                    (
                        torch.zeros((persistence0.shape[0], persistence0.shape[1], 1)).to(
                            x.device
                        ),
                        persistence0,
                    ),
                    dim=-1,
                )
                persistence1 = torch.cat(
                    (
                        torch.zeros((persistence1.shape[0], persistence1.shape[1], 1)).to(
                            x.device
                        ),
                        persistence1,
                    ),
                    dim=-1,
                )
                persistence1[persistence1[:, :, 1:].nonzero(as_tuple=True)] = 1.0

            persistence0[persistence0.isnan()] = 1.0

        return persistence0, persistence1

    def forward(self, x, data,pos):
        edge_index, vertex_slices, edge_slices, batch = remove_duplicate_edges(data)

        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices,pos
        )
        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)

        # processing x0
        x0 = self.deepset_fn(x0, batch)
        # processing x1
        if self.dim1:
            x1_list = []
            for i in range(self.num_filtrations):
                pers1_reshaped = pers1[i, :, :].reshape(pers1.shape[1], -1)
                pers1_mask = ~((pers1_reshaped == 0).all(-1))
                x1 = pers1_reshaped[pers1_mask]
                x1_list.append(self.deepset_fn_dim1(x1, edge_slices, mask=pers1_mask))
            x1 = torch.stack(x1_list).mean(dim=0)
            x0 = x0 + x1
        x = x0
        x = self.bn(x)
        x = self.out(x)
        return x
    


class RephineLayerToy(RephineLayer):

    def __init__(self,
        n_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type="rephine",
        dim1=True,
        sig_filtrations=True,
        reduce_tuples=False
    ):
        super().__init__(n_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type=diagram_type,
        dim1=dim1,
        sig_filtrations=sig_filtrations,
        reduce_tuples=reduce_tuples)

        self.out = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index, vertex_slices, edge_slices, batch = remove_duplicate_edges(data)
        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices
        )
        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)

        # processing x0
        x0 = self.deepset_fn(x0, batch)
        # processing x1
        if self.dim1:
            x1_list = []
            for i in range(self.num_filtrations):
                pers1_reshaped = pers1[i, :, :].reshape(pers1.shape[1], -1)
                pers1_mask = ~((pers1_reshaped == 0).all(-1))
                x1 = pers1_reshaped[pers1_mask]
                x1_list.append(self.deepset_fn_dim1(x1, edge_slices, mask=pers1_mask))
            x1 = torch.stack(x1_list).mean(dim=0)
            x0 = x0 + x1
        x = x0
        h = x.clone().detach()
        x = self.bn(x)
        x = self.out(x)
        return x, h