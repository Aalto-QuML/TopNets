import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GCNConv,GPSConv,TransformerConv
from torchdiffeq import odeint as odeint
from layers.gin_layer import GINLayer
from layers.graph_convolution_layer import GCNLayer
from models.gnn_factory.gcn_layer_factory import GcnCreator
from models.gnn_factory.gin_layer_factory import GinCreator
from models.gnn_factory.pna_layer_factory import PnaCreator
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from models.egnn_clean import EGNN


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
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance

        #self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        self.ode_embed = torch.nn.Linear(hidden_dim+1, hidden_dim)
        #layers = []
        #for i in range(depth):
        layers = [TransformerConv(hidden_dim+1,hidden_dim,head=1)]
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
        x = odeint(ode_rhs,x,time_steps,method=self.solver,atol=1e-4,rtol=1e-4)
        #x = self.pooling_fun(x, data.batch)
        #x = self.classif(x)
        return x


class EGNN_cont(nn.Module):
    def __init__(
        self,
        #gnn,
        hidden_dim,
        #depth,
        num_node_features,
        #num_classes,
        global_pooling,
        n_steps,
        solver,
        #deg=None,
        #batch_norm=True,
    ):
        super().__init__()
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        self.act_fn = nn.LeakyReLU(0.3)
        self.n_steps = n_steps
        self.solver = solver


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 +  1, 2*hidden_dim),
            self.act_fn,
            nn.Linear(2*hidden_dim, 2*hidden_dim ),
            self.act_fn,
            nn.Linear(2*hidden_dim, hidden_dim))
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim + 3 +4, hidden_dim ),
            self.act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.act_fn,
            nn.Linear(hidden_dim//2, hidden_dim // 4))
        

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim //2 ),
            self.act_fn,
            nn.Linear(hidden_dim //2 , hidden_dim // 2),
            self.act_fn,
            nn.Linear(hidden_dim//2, 3))


        self.edge_index = 0
        self.coords_weight=1.0
        self.edge_attr = 0

    def ode(self,t,data):
        x,pos = data[:,:-3],data[:,-3:]
        tt = torch.ones_like(x[:, :1]) * t
        dx_final = torch.cat([tt.float(), x], 1)
        row, col = self.edge_index
        radial, coord_diff = self.coord2radial(self.edge_index, pos)
        edge_feats = self.edge_model(dx_final[row],dx_final[col],radial,edge_attr=self.edge_attr)
        coord = self.coord_model(pos, self.edge_index, coord_diff, edge_feats)
        h, agg = self.node_model(dx_final, self.edge_index, edge_feats, node_attr=None)
        h = torch.cat([h,coord],dim=1)
        return h
    

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out
    

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord
    
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        #out = x + out
        return out, agg
    

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        #if self.norm_diff:
        #    norm = torch.sqrt(radial) + 1
        #    coord_diff = coord_diff/(norm)

        return radial, coord_diff
    

    def forward(self, data):
        x, self.edge_index,pos,self.edge_attr = data.x, data.edge_index,data.pos,data.edge_attr
        time_steps = torch.linspace(0,2,steps=self.n_steps)
        ode_rhs  = lambda t,x: self.ode(t,x)
        x = self.embedding(x.float())
        x = torch.cat([x,pos],dim=1)
        x = odeint(ode_rhs,x,time_steps,method=self.solver,atol=1e-3,rtol=1e-3)
        node_feat,coords  = x[:,:,:-3],x[-1,:,-3:]
        return node_feat
    
class SSP(torch.nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)
    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class EGNN_cont_v2(nn.Module):
    def __init__(
        self,
        #gnn,
        hidden_dim,
        #depth,
        num_node_features,
        #num_classes,
        global_pooling,
        n_steps,
        solver,
        edge_nf
        #deg=None,
        #batch_norm=True,
    ):
        super().__init__()
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        self.n_steps = n_steps
        self.solver = solver

        self.egnn_model = EGNN(in_node_nf=hidden_dim,hidden_nf = hidden_dim,out_node_nf= hidden_dim,in_edge_nf=edge_nf,act_fn=SSP(),n_layers=1,residual=False)
        self.edge_index = 0
        self.coords_weight=1.0
        self.edge_attr = 0

    def ode(self,t,data):
        x,pos = data[:,:-3],data[:,-3:]
        tt = torch.ones_like(x[:, :1]) * t
        dx_final = torch.cat([tt.float(), x], 1)
        h,x = self.egnn_model(h=x,x=pos,edges=self.edge_index, edge_attr=self.edge_attr)
        h = torch.cat([h,x],1)
        return h
    

    def forward(self, data):
        x, self.edge_index,pos,self.edge_attr = data.x, data.edge_index,data.pos,data.edge_attr
        time_steps = torch.linspace(0,1,steps=self.n_steps)
        ode_rhs  = lambda t,x: self.ode(t,x)
        x = self.embedding(x.float())
        x = torch.cat([x,pos],dim=1)
        x = odeint(ode_rhs,x,time_steps,method=self.solver,atol=1e-1,rtol=1e-1)
        node_feat,coords  = x[:,:,:-3],x[-1,:,-3:]
        return node_feat



# Used in the toy example
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 36)
        self.conv2 = GCNConv(36, 16)
        self.out = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 1))
        self.bn = nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        h = x.clone().detach()
        x = self.bn(x)
        x = self.out(x)
        return x, h
    


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


if __name__=='__main__':
    input_node_feat = torch.randn(10,15)
    input_pos_feat = torch.randn(10,3)
    input_edge_index = [torch.randint(0,10,(20,)),torch.randint(0,10,(20,))]
    data = Data(x=input_node_feat,edge_index=input_edge_index,pos=input_pos_feat)
    model  = EGNN_cont(hidden_dim=64,num_node_features=15,global_pooling='mean',n_steps=10,solver='adaptive_heun')
    breakpoint()
    print(model)
    final = model(data)
