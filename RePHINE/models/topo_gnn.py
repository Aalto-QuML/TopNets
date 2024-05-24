import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GINConv,GCNConv,GPSConv
from models.gnn import GNN_cont,GNN
import sys
sys.path.append('../')
from layers.rephine_layer import RephineLayer,RephineLayer_Equiv
from torchdiffeq import odeint as odeint
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from TOGL.topognn.models import TopologyLayer

class TopoGNN(GNN):
    def __init__(
        self,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        gnn,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        diagram_type="rephine",
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
        global_pooling="mean",
        deg=None,
        batch_norm=False,
    ):
        super().__init__(
            gnn=gnn,
            hidden_dim=hidden_dim,
            depth=depth,
            num_node_features=num_node_features,
            num_classes=num_classes,
            deg=deg,
            global_pooling=global_pooling,
            batch_norm=batch_norm,
        )

        topo_layers = []
        self.ph_pooling_type = ph_pooling_type
        for i in range(len(self.layers)):
            topo = RephineLayer(
                n_features=hidden_dim,
                n_filtrations=num_filtrations,
                filtration_hidden=filtration_hidden,
                out_dim=out_ph_dim,
                diagram_type=diagram_type,
                dim1=dim1,
                sig_filtrations=sig_filtrations,
            )
            topo_layers.append(topo)

        self.ph_layers = nn.ModuleList(topo_layers)

        final_dim = (
            hidden_dim + len(self.ph_layers) * out_ph_dim
            if self.ph_pooling_type == "cat"
            else hidden_dim + out_ph_dim
        )
        self.classif = torch.nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        if self.ph_pooling_type != "mean":
            self.jump = JumpingKnowledge(mode=self.ph_pooling_type)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x.float())

        ph_vectors = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index)
            ph_vectors += [self.ph_layers[i](x, data)]

        # Pooling GNN embeddings
        x = self.pooling_fun(x, data.batch)

        # Pooling PH diagrams
        if self.ph_pooling_type == "mean":
            ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        else:
            ph_embedding = self.jump(ph_vectors)
        x_pre_class = torch.cat([x, ph_embedding], axis=1)

        # Final classification
        x = self.classif(x_pre_class)
        return x




class TopNN_2D(nn.Module):
    def __init__(
        self,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        gnn,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        n_steps, #time-steps for ODE-system
        solver, # solver used for solving ODE
        diagram_type,
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
        global_pooling="sum",
        deg=None,
        batch_norm=False,
    ):
        super().__init__()
        self.ODE_GNN = GNN_cont(gnn,hidden_dim,depth,num_node_features,num_classes,n_steps,solver,deg,batch_norm)
        topo_layers = []
        self.ph_pooling_type = ph_pooling_type
        if global_pooling == "mean":
            self.pooling_fun = global_mean_pool
        elif global_pooling == "sum":
            self.pooling_fun = global_add_pool

        for i in range(n_steps):
            topo = RephineLayer(
                n_features=hidden_dim,
                n_filtrations=num_filtrations,
                filtration_hidden=filtration_hidden,
                out_dim=out_ph_dim,
                diagram_type=diagram_type,
                dim1=dim1,
                sig_filtrations=sig_filtrations,
            )
            topo_layers.append(topo)

        self.ph_layers = nn.ModuleList(topo_layers)

        final_dim = (
            hidden_dim + len(self.ph_layers) * out_ph_dim
            if self.ph_pooling_type == "cat"
            else hidden_dim + out_ph_dim
        )
        self.classif = torch.nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self.node_readout = torch.nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim))

        if self.ph_pooling_type != "mean":
            self.jump = JumpingKnowledge(mode=self.ph_pooling_type)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        ode_node_embed = self.ODE_GNN(data)
        ph_vectors = []
        for i in range(ode_node_embed.shape[0]):
            ph_vectors += [self.ph_layers[i](ode_node_embed[i],data)]

        # Pooling GNN embeddings
        x = self.pooling_fun(self.node_readout(ode_node_embed[-1]), data.batch)

        # Pooling PH diagrams
        if self.ph_pooling_type == "mean":
            ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        else:
            ph_embedding = self.jump(ph_vectors)
        x_pre_class = torch.cat([x, ph_embedding], axis=1)

        # Final classification
        x = self.classif(x_pre_class)
        return x
    

    

class TopNN_TOGL(nn.Module):
    def __init__(
        self,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        gnn,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        n_steps, #time-steps for ODE-system
        solver, # solver used for solving ODE
        diagram_type='standard',
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
        global_pooling="sum",
        deg=None,
        batch_norm=False,
    ):
        super().__init__()

        if gnn == "gin":
            gin_net = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
            gin_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
            layers = [GINConv(gin_net),GINConv(gin_net_2)]
        elif gnn == "gcn":
            layers = [GCNConv(hidden_dim+1,hidden_dim),GCNConv(hidden_dim,hidden_dim)]

        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        self.layers = nn.ModuleList(layers)
        self.n_steps = n_steps
        self.solver = solver

        self.edge_index = 0

        self.ph_pooling_type = ph_pooling_type

        if global_pooling == "mean":
            self.pooling_fun = global_mean_pool
        elif global_pooling == "sum":
            self.pooling_fun = global_add_pool

        self.num_coord_funs = 3
        self.num_coord_funs1 = self.num_coord_funs

        coord_funs = {"Triangle_transform": self.num_coord_funs,
                          "Gaussian_transform": self.num_coord_funs,
                          "Line_transform": self.num_coord_funs,
                          "RationalHat_transform": self.num_coord_funs
                          }

        coord_funs1 = {"Triangle_transform": self.num_coord_funs1,
                           "Gaussian_transform": self.num_coord_funs1,
                           "Line_transform": self.num_coord_funs1,
                           "RationalHat_transform": self.num_coord_funs1
                           }

        tanh_filtrations=False
        deepset_type='full'
        share_filtration_parameters=True
        fake=False
        deepset=False
        swap_bn_order=False,
        dist_dim1=False
        residual_and_bn=True
        self.topo_vectors = []
        self.topo1 = TopologyLayer(
                hidden_dim, hidden_dim, num_filtrations=num_filtrations,
                num_coord_funs=coord_funs, filtration_hidden=filtration_hidden,
                dim1=True, num_coord_funs1=coord_funs1,
                residual_and_bn=residual_and_bn, swap_bn_order=swap_bn_order,
                share_filtration_parameters=share_filtration_parameters, fake=fake,
                tanh_filtrations=tanh_filtrations,
                dist_dim1=dist_dim1
                )


        self.classif = torch.nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self.node_readout = torch.nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim))

        self.big_data = 0

    def ode(self,t,dx):
        tt = torch.ones_like(dx[:, :1]) * t
        x = torch.cat([tt.float(), dx], 1)
        x = self.layers[0](x, edge_index=self.edge_index)
        x, x_dim1, filtration = self.topo1(x, self.big_data, return_filtration=False)
        x = self.layers[1](x, edge_index=self.edge_index)

        #for idx,layer in enumerate(self.layers):
            #if idx==0:
            #    x = layer(self.ode_embed(x),edge_index=self.edge_index)
            #else: 
        #    x = layer(x, edge_index=self.edge_index)
        #    x, x_dim1, filtration = self.topo1(x, self.big_data, return_filtration=False)

        self.topo_vectors.append(x_dim1)

        return x
    

    def forward(self, data):
        self.big_data = data.clone()
        self.topo_vectors = []
        x, self.edge_index = data.x, data.edge_index
        time_steps = torch.linspace(0,1,steps=self.n_steps)
        ode_rhs  = lambda t,x: self.ode(t,x)
        x = self.embedding(x.float())
        x = odeint(ode_rhs,x,time_steps,method=self.solver,atol=1e-3,rtol=1e-3)

        #for i, layer in enumerate(self.n_st):
        #    x = layer(x, edge_index=edge_index)
        #    ph_vectors += [self.ph_layers[i](x, data,pos)]

        # Pooling GNN embeddings
        x = self.pooling_fun(self.node_readout(x[-1]), data.batch)
        ph_embedding = torch.stack(self.topo_vectors).mean(dim=0)
        # Pooling PH diagrams
        #if self.ph_pooling_type == "mean":
        #    ph_embedding = torch.stack(self.topo_vectors).mean(dim=0)
        #else:
        #    ph_embedding = self.jump(ph_vectors)
        x_pre_class = torch.cat([x, ph_embedding], axis=1)

        # Final classification
        x = self.classif(x_pre_class)
        return x


if __name__=='__main__':
    input_node_feat = torch.randn(10,15)
    input_pos_feat = torch.randn(10,3)
    input_edge_index = [torch.randint(0,10,(20,)),torch.randint(0,10,(20,))]
    G = nx.path_graph(10)
    pyg_graph = from_networkx(G)
    data = [Data(x=input_node_feat,edge_index=pyg_graph.edge_index,pos=input_pos_feat),Data(x=input_node_feat,edge_index=pyg_graph.edge_index,pos=input_pos_feat)]
    loader = DataLoader(data, batch_size=2, shuffle=True)
    model  = TopNN(hidden_dim=64,num_node_features=15,num_filtrations=8,filtration_hidden=16,out_ph_dim=64,n_steps=10,solver='adaptive_heun')
    breakpoint()
    print(model)
    for data in loader:
        final = model(data)
