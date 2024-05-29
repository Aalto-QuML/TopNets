import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool, MessagePassing
from typing import Tuple, Dict, List
from torch_scatter import scatter_add
from models.utils import compute_invariants_3d
from torch_geometric.data import Data
from torch_geometric.data import DataLoader 
from torch_cluster import knn_graph
from torchdiffeq import odeint as odeint
import sys
sys.path.append('../')
from RePHINE.layers.rephine_layer import RephineLayer_Equiv
from egnn.lib_adatop.pc_representation import PCFeatureNet
from egnn.lib_adatop.pc_representation  import *


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str) -> None:
        super().__init__()

        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.layers = nn.ModuleList(
            [EMPSNLayer(adjacencies, self.max_dim, num_hidden) for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )

    def forward(self, graph: Data) -> Tensor:
        x_ind = {
            str(i): getattr(graph, f'x_{i}').long() for i in range(self.max_dim + 1)
        }

        # compute initial features
        x = {
            str(i): torch.sum(torch.stack([
                graph.x[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        x_batch = {
            str(i): getattr(graph, f'x_{i}_batch') for i in range(self.max_dim + 1)
        }

        adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        inv = compute_invariants_3d(x_ind, graph.pos, adj, inv_ind, graph.pos.device)

        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(state)
        out = torch.squeeze(out)

        return out

    def __str__(self):
        return f"EMPSN ({self.type})"


class EMPSN_Rephine(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str) -> None:
        super().__init__()

        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(num_layers):
            topo = RephineLayer_Equiv(
                n_features=num_hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type='rephine',
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"

        self.layers = nn.ModuleList(
            [EMPSNLayer(adjacencies, self.max_dim, num_hidden) for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden + self.out_ph, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )

    def forward(self, graph: Data) -> Tensor:
        x_ind = {
            str(i): getattr(graph, f'x_{i}').long() for i in range(self.max_dim + 1)
        }

        # compute initial features
        x = {
            str(i): torch.sum(torch.stack([
                graph.x[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        x_batch = {
            str(i): getattr(graph, f'x_{i}_batch') for i in range(self.max_dim + 1)
        }

        adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        inv = compute_invariants_3d(x_ind, graph.pos, adj, inv_ind, graph.pos.device)

        batches = get_batched_data(x['0'],graph.batch,graph.num_graphs)
        # message passing
        ph_vectors = []
        for i,layer in enumerate(self.layers):
            x = layer(x, adj, inv)
            for batch in batches:
                ph_vectors += [self.ph_layers[i](x['0'], batch,graph.pos)]

        # read out
        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(torch.cat([state,ph_embedding],dim=1))
        out = torch.squeeze(out)
        return out

    def __str__(self):
        return f"EMPSN ({self.type})"

class EMPSNLayer(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Simplicial Network.

    A message passing layer is added for each type of adjacency to the message_passing dict. For each simplex, a state is
    found by concatening the messages sent to that simplex, e.g. we update an edge by concatenating the messages from
    nodes, edges, and triangles. The simplex is update by passing this state through an MLP as found in the update dict.
    """
    def __init__(self, adjacencies: List[str], max_dim: int, num_hidden: int) -> None:
        super().__init__()
        self.adjacencies = adjacencies

        dict_temp = {
            '0_0': 3,
            '0_1': 3,
            '1_1': 6,
            '1_2': 6,
        }
        # messages
        self.message_passing = nn.ModuleDict({
            adj: SimplicialEGNNLayer(num_hidden, dict_temp[adj]) for adj in adjacencies
        })

        # updates
        self.update = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.update[str(dim)] = nn.Sequential(
                nn.Linear(factor * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_hidden)
            )

    def forward(self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=index,
                edge_attr=inv[adj_type]
            ) for adj_type, index in adj.items()
        }

        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature] + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim], dim=1
            ) for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        return x


class EMPSNLayer_Cont(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Simplicial Network.

    A message passing layer is added for each type of adjacency to the message_passing dict. For each simplex, a state is
    found by concatening the messages sent to that simplex, e.g. we update an edge by concatenating the messages from
    nodes, edges, and triangles. The simplex is update by passing this state through an MLP as found in the update dict.
    """
    def __init__(self, adjacencies: List[str], max_dim: int, num_hidden: int,num_output:int) -> None:
        super().__init__()
        self.adjacencies = adjacencies

        dict_temp = {
            '0_0': 3,
            '0_1': 3,
            '1_1': 6,
            '1_2': 6,
        }
        # messages
        self.message_passing = nn.ModuleDict({
            adj: SimplicialEGNNLayer(num_hidden , dict_temp[adj]) for adj in adjacencies
        })

        # updates
        self.update = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.update[str(dim)] = nn.Sequential(
                nn.Linear(factor * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_output)
            )

        self.shortcut = nn.Linear(num_hidden, num_output)

    def forward(self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=index,
                edge_attr=inv[adj_type]
            ) for adj_type, index in adj.items()
        }

        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature] + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim], dim=1
            ) for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: self.shortcut(feature) + h[dim] for dim, feature in x.items()}

        return x

class SimplicialEGNNLayer(nn.Module):
    def __init__(self, num_hidden, num_inv):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.edge_inf_mlp = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = scatter_add(messages * edge_weights, index_rec, dim=0, dim_size=x_rec.shape[0])

        return messages_aggr

class EMPSN_PC(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str) -> None:
        super().__init__()

        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.topo  = PCFeatureNet(3, 128, matds=True)

        self.layers = nn.ModuleList(
            [EMPSNLayer(adjacencies, self.max_dim, num_hidden) for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden + 32, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )

    def forward(self, graph: Data) -> Tensor:
        x_ind = {
            str(i): getattr(graph, f'x_{i}').long() for i in range(self.max_dim + 1)
        }

        # compute initial features
        x = {
            str(i): torch.sum(torch.stack([
                graph.x[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        x_batch = {
            str(i): getattr(graph, f'x_{i}_batch') for i in range(self.max_dim + 1)
        }

        adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        inv = compute_invariants_3d(x_ind, graph.pos, adj, inv_ind, graph.pos.device)

        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        topo_rep = self.topo(graph.pos.view(-1,128,3)).to(graph.pos.device)

        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        state = torch.cat([state,topo_rep],dim=1)
        out = self.post_pool(state)
        out = torch.squeeze(out)

        return out

    def __str__(self):
        return f"EMPSN_PC ({self.type})"



class EMPSN_Rephine_Cont(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int, max_com: str, solver : str, nsteps: int) -> None:
        super().__init__()

        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        self.solver = solver
        self.nsteps = nsteps

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(nsteps):
            topo = RephineLayer_Equiv(
                n_features=num_hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type='rephine',
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"

        self.layers = nn.ModuleList(
            [EMPSNLayer_Cont(adjacencies, self.max_dim, num_hidden+1,num_hidden) for _ in range(num_layers)]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.post_pool = nn.Sequential(
            nn.Sequential(nn.Linear((max_dim + 1) * num_hidden + self.out_ph, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_out))
        )
        self.inv = 0
        self.adj = 0




    def ode(self,t,x):
         t = {dim:torch.ones_like(x[dim][:, :1]) * t for dim, feature in enumerate(x)}
         x = {str(dim): torch.cat([feature,t[dim]],1) for dim, feature in enumerate(x)}

         for i,layer in enumerate(self.layers):
            x = layer(x, self.adj, self.inv)

         return tuple(x.values())





    def forward(self, graph: Data) -> Tensor:
        x_ind = {
            str(i): getattr(graph, f'x_{i}').long() for i in range(self.max_dim + 1)
        }

        # compute initial features
        x = {
            str(i): torch.sum(torch.stack([
                graph.x[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        x_batch = {
            str(i): getattr(graph, f'x_{i}_batch') for i in range(self.max_dim + 1)
        }

        self.adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        self.inv = compute_invariants_3d(x_ind, graph.pos, self.adj, inv_ind, graph.pos.device)

        batches = get_batched_data(x['0'],graph.batch,graph.num_graphs)
        # message passing
        ph_vectors = []

        time_steps = torch.linspace(0,1,steps=self.nsteps).to(graph.pos.device)
        ode_rhs  = lambda t,x: self.ode(t,x)
        #print(tuple(x.values()))
        x = odeint(ode_rhs,tuple(x.values()),time_steps,method=self.solver,atol=1e-2,rtol=1e-2)

        for i in range(self.nsteps):
            for batch in batches:
                ph_vectors += [self.ph_layers[i](x[0][-1], batch,graph.pos)]
        # read out
                
        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = {dim: self.pre_pool[str(dim)](feature[-1]) for dim, feature in enumerate(x)}
        x = {dim: global_add_pool(x[dim], x_batch[str(dim)]) for dim, feature in enumerate(x)}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(torch.cat([state,ph_embedding],dim=1))
        out = torch.squeeze(out)
        
        return out

    def __str__(self):
        return f"EMPSN ({self.type})"
    

def get_batched_data(x,batch,batch_size):
    num_neigh =3
    batches = []
    for i in range(batch_size):
        idx = (batch == i).nonzero(as_tuple=True)[0]
        edges = knn_graph(x[idx].cpu(),num_neigh,loop=False).to(x.device)
        batches.append(Data(x=x[idx],edge_index=edges))

    return DataLoader(batches,shuffle=False,batch_size=len(batches))

