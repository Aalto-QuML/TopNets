import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GCNConv,GPSConv,TransformerConv,GINConv
from torchdiffeq import odeint as odeint
from torch_geometric.data import Data
from torch_geometric.data import DataLoader



class GNN(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        global_pooling,
        deg=None,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)

        dim_before_class = hidden_dim
        self.classif = torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x.float())

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)

        x = self.pooling_fun(x, data.batch)
        x = self.classif(x)
        return x


class GNN_cont(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        n_steps,
        solver,
        global_pooling,
        deg=None,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gin_net = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
            layers = [GINConv(gin_net)]
        elif gnn == "gcn":
            layers = [GCNConv(hidden_dim+1,hidden_dim)]

        #build_gnn_layer = gnn_instance.return_gnn_instance

        #self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        #self.ode_embed = torch.nn.Linear(hidden_dim+1, hidden_dim)
        #layers = []
        #for i in range(depth):
        #layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)
        self.n_steps = n_steps
        self.solver = solver

        self.edge_index = 0

    def ode(self,t,data):
        tt = torch.ones_like(data[:, :1]) * t
        x = torch.cat([tt.float(), data], 1)
        for idx,layer in enumerate(self.layers):
            #if idx==0:
            #    x = layer(self.ode_embed(x),edge_index=self.edge_index)
            #else: 
            x = layer(x, edge_index=self.edge_index)

        return x
    

    def forward(self, data):
        x, self.edge_index = data.x, data.edge_index
        time_steps = torch.linspace(0,1,steps=self.n_steps)
        ode_rhs  = lambda t,x: self.ode(t,x)
        x = self.embedding(x.float())
        x = odeint(ode_rhs,x,time_steps,method=self.solver,atol=1e-3,rtol=1e-3)
        #x = self.pooling_fun(x, data.batch)
        #x = self.classif(x)
        return x



if __name__=='__main__':
    input_node_feat = torch.randn(10,15)
    input_pos_feat = torch.randn(10,3)
    input_edge_index = [torch.randint(0,10,(20,)),torch.randint(0,10,(20,))]
    data = Data(x=input_node_feat,edge_index=input_edge_index,pos=input_pos_feat)
    breakpoint()
    print(model)
    final = model(data)
