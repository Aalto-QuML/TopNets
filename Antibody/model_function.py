import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics.pairwise import rbf_kernel
import warnings
from torch_geometric.nn import GATConv,GATv2Conv,TransformerConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DimeNet
from torch_geometric.nn import global_mean_pool, global_add_pool
import matplotlib.pyplot as plt
from  torch.distributions import multivariate_normal
from torch_scatter import scatter_add
from torch.nn.functional import normalize
from torch_geometric.utils import add_self_loops
from utils import *
from torch_scatter import scatter_add
from torch_geometric.nn import knn_graph,radius_graph

import sys
sys.path.append('../')
from RePHINE.layers.rephine_layer import RephineLayer_Equiv



class TopNets(nn.Module): 
    
    def __init__(self,c_in,c_out,n_layers,params=None):
        super().__init__()
        layers_mlp = []
        activation_fns = []
        hidden_dim=128
        num_inputs = [c_in,128,256,128]
        num_outputs = [128,256,128,64]
        for layer in range(len(num_outputs)):
            activation_fn = nn.ReLU()
            layers_mlp.append(TransformerConv(num_inputs[layer],num_outputs[layer],edge_dim=46,heads=1))
            activation_fns.append(activation_fn)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(len(num_outputs)):
            topo = RephineLayer_Equiv(
                n_features=num_outputs[i],
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type='standard',
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"
        
        self.readout =  nn.Sequential(
            nn.Sequential(nn.Linear(hidden_dim + self.out_ph, hidden_dim), nn.LeakyReLU(0.3), nn.Linear(hidden_dim, c_out))
        )

        self.prepool =  nn.Sequential(
            nn.Sequential(nn.Linear(hidden_dim//2, 2*hidden_dim), nn.LeakyReLU(0.3), nn.Linear(2*hidden_dim, hidden_dim))
        )

        self.prepool_ph =  nn.Sequential(
            nn.Sequential(nn.Linear(self.out_ph, 2*self.out_ph), nn.LeakyReLU(0.3), nn.Linear(2*self.out_ph, self.out_ph))
        )
 
        self.layer_mlp = nn.ModuleList(layers_mlp)
        self.activation_mlp = nn.ModuleList(activation_fns)

        self.edge_index = 0
        self.edge_ab = 0
        self.antigen_coords = 0
        self.antigen_labels = 0
        self.order = 0
        self.amino_index = 0
        
        
    def _get_quaternion(self,R):
        # Taken from https://github.com/wengong-jin/RefineGNN/blob/main/structgen/protein_features.py
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        
        return Q
        
    def _get_pairwise_scores(self,total_scores,edge_index):
        
        starting = torch.index_select(total_scores,0,edge_index[0].long())
        ending = torch.index_select(total_scores,0,edge_index[1].long())
        diff = starting - ending
        
        return diff
    
    
    def _get_pairwise(self,euclid_coords,edge_index):
            
            starting = torch.index_select(euclid_coords,0,edge_index[0].long())
            ending = torch.index_select(euclid_coords,0,edge_index[1].long())
            diff_vector = starting - ending
            diff_norm = torch.norm(diff_vector,dim=2).view(-1,3)
            diff_vector = F.normalize(diff_vector.view(-1,3,3), dim=2)

            return diff_norm,diff_vector.view(-1,9)
    
    def _rbf_weight(self,d_ij):
        
        # Different resolution weight depending on position
        alpha_dist = d_ij.view(len(d_ij),3)
        diff_scaled = torch.div(alpha_dist,1*torch.ones(3*len(alpha_dist)).view(len(d_ij),3))
        RBF = torch.exp(-diff_scaled).view(-1,3)
        return RBF
        
    def _get_orientation_vector(self,O,r_ij,edge_index):
        starting_node_orientation = torch.index_select(O,0,edge_index[0].long()).view(-1,3,3)
        vector_mat = F.normalize(r_ij[:,3:6].view(-1,3,1),dim=-1)
        transposed_starting_mat = torch.transpose(starting_node_orientation,1,2)
        final_orient_vector = torch.matmul(transposed_starting_mat,vector_mat)
        return final_orient_vector.view(-1,3)
        
    def _get_orientations(self,coords,eps=1e-6):
        
        
        #print("Number of nodes",len(coords), " and Edges ",E_idx.shape[1])
        coords = coords.reshape(len(coords),3,3)
        X = coords[:,:3,:].reshape(1,3*coords.shape[0],3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(torch.tensor(dX), dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
    
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        
        O = F.pad(O, (0,0,1,2), 'constant', 0).view(len(coords),3,9)
        O_final = O[:,1,:]
        #print(O_final.shape)
        
        return O_final
    
    
    def _get_neighbour_orientation(self,O,edge_index):
        
        starting_node_orientation = torch.index_select(O,0,edge_index[0].long()).view(-1,3,3)
        ending_node_orientation = torch.index_select(O,0,edge_index[1].long()).view(-1,3,3)
        transposed_mat = torch.transpose(starting_node_orientation,1,2)
        final_combined_orient = torch.matmul(transposed_mat,ending_node_orientation)
        
        return final_combined_orient.view(-1,9)
        
        

    def forward(self,data):
        Node_label, Node_coord  = data.x.float(), data.pos.float()
        Edge_index_ag = data.edge_index
        r_ij,r_ij_vector = self._get_pairwise(Node_coord.view(-1,3,3),Edge_index_ag)
        rbf_weight = self._rbf_weight(r_ij).float().view(-1,3)
        node_label_ag = self._get_pairwise_scores(Node_label,Edge_index_ag)
        spatial_diff = self._get_pairwise_scores(data.a_index.view(-1,1),Edge_index_ag)
        
        Orientations_node = self._get_orientations(Node_coord.view(-1,3,3))
        orient_features = self._get_neighbour_orientation(Orientations_node,Edge_index_ag)
        oriented_vector = self._get_orientation_vector(Orientations_node,r_ij_vector.view(-1,9),Edge_index_ag)
        final_edge_feature = torch.cat([spatial_diff,node_label_ag,rbf_weight,r_ij_vector,orient_features,oriented_vector,data.order.view(-1,1)],dim=1).float()
        h,x = data.x,data.pos
        h = torch.cat([h,x],dim=1)
        ph_vectors = []
        for l,layer in enumerate(self.layer_mlp):
            h = layer(h,edge_index=Edge_index_ag,edge_attr=final_edge_feature)
            ph_vectors += [self.ph_layers[l](h.float(), data,data.pos)]
            h = self.activation_mlp[l](h)
        
        h = self.prepool(h)
        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        ph_embedding = self.prepool_ph(ph_embedding)
        comb_input = torch.cat([h,ph_embedding[data.batch]],dim=1)
        out = self.readout(comb_input)
        return out
