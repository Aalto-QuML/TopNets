import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GCNConv,GPSConv
#from ..layers import rephine_layer
from models.gnn import EGNN_cont,EGNN_cont_v2,SSP,GNN_cont,GNN
import sys
sys.path.append('../')
from layers.rephine_layer import RephineLayer,RephineLayer_Equiv
#from models.gnn import GNN
import networkx as nx
from torch_geometric.utils.convert import from_networkx



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
        diagram_type="rephine",
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

        #for i, layer in enumerate(self.n_st):
        #    x = layer(x, edge_index=edge_index)
        #    ph_vectors += [self.ph_layers[i](x, data,pos)]

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
    



class TopNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_node_features,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        edge_nf,
        n_steps, #time-steps for ODE-system
        solver, # solver used for solving ODE
        diagram_type="rephine",
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
        global_pooling="mean",
        deg=None,
        batch_norm=False,
    ):
        super().__init__()

        self.ODE_GNN = EGNN_cont_v2(hidden_dim,num_node_features,global_pooling,n_steps,solver=solver,edge_nf=edge_nf)
        topo_layers = []
        self.ph_pooling_type = ph_pooling_type
        for i in range(n_steps):
            topo = RephineLayer_Equiv(
                n_features=hidden_dim,
                n_filtrations=num_filtrations,
                filtration_hidden=filtration_hidden,
                out_dim=out_ph_dim,
                diagram_type=diagram_type,
                dim1=dim1,
                sig_filtrations=sig_filtrations,
            )
            topo_layers.append(topo)

        if global_pooling == "mean":
            self.pooling_fun = global_mean_pool
        elif global_pooling == "sum":
            self.pooling_fun = global_add_pool


        self.ph_layers = nn.ModuleList(topo_layers)
        final_dim = (
            hidden_dim + len(self.ph_layers) * out_ph_dim
            if self.ph_pooling_type == "cat"
            else hidden_dim + out_ph_dim
        )
        self.act_fn = nn.LeakyReLU(0.3)
        self.reggif = torch.nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.node_dec = nn.Linear(hidden_dim,hidden_dim)
        #self.node_dec = torch.nn.Sequential(
        #    nn.Linear(hidden_dim, 2*hidden_dim),
        #    self.act_fn,
        #    nn.Linear(2*hidden_dim, 2*hidden_dim),
        #    self.act_fn,
        #    nn.Linear(2*hidden_dim, hidden_dim),
        #)

        #self.classif = torch.nn.Sequential(
        #    nn.Linear(final_dim, hidden_dim // 2),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim // 2, hidden_dim // 4),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim // 4, num_classes),
        #)

        if self.ph_pooling_type != "mean":
            self.jump = JumpingKnowledge(mode=self.ph_pooling_type)

    def forward(self, data):
        x, edge_index,pos = data.x, data.edge_index,data.pos
        ode_node_embed = self.ODE_GNN(data)
        ph_vectors = []
        for i in range(ode_node_embed.shape[0]):
            temp_embed = self.node_dec(ode_node_embed[i])
            ph_vectors += [self.ph_layers[i](temp_embed, data,pos)]

        #for i, layer in enumerate(self.n_st):
        #    x = layer(x, edge_index=edge_index)
        #    ph_vectors += [self.ph_layers[i](x, data,pos)]

        # Pooling GNN embeddings
        x = self.pooling_fun(self.node_dec(ode_node_embed[-1]), data.batch)

        # Pooling PH diagrams
        if self.ph_pooling_type == "mean":
            ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        else:
            ph_embedding = self.jump(ph_vectors)

        x_pre_class = torch.cat([x, ph_embedding], axis=1)

        # Final regression
        x = self.reggif(x_pre_class)
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
