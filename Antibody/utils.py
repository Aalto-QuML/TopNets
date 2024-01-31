import torch
import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch_geometric.nn import knn_graph,radius_graph
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from scipy.special import softmax
import torch.nn as nn
from rmsd import *
from scipy.stats import vonmises
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import *
import sys
sys.path.append('../')

np.random.seed(10)
random.seed(10)
unq_aa = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V','O','U','B','Z','X','J']
ALPH_prot = list(unq_aa)


def get_antibody_entries(pred,batch,antibody_len,anitgen_len):
    size = int(batch.max().item()+1)
    for entry in range(size):
        ab_len = antibody_len[entry]
        ag_len = anitgen_len[entry]
        if entry == 0: 
            start_pos = 0
        ab_end_pos = start_pos + ab_len
        
        ab_pred = pred[start_pos:ab_end_pos]
        start_pos = ab_end_pos + ag_len
        
        if entry == 0:
            final_data = ab_pred
        else:
            final_data = torch.cat([final_data,ab_pred])
            
    return final_data
    



def R_vec_z(theta):
        c, s = torch.cos(theta).view(1,-1), torch.sin(theta).view(1,-1)
        return torch.cat([c,s,torch.zeros(len(theta)).view(1,-1),s,c,torch.zeros(len(theta)).view(1,-1),torch.zeros(len(theta)).view(1,-1),torch.zeros(len(theta)).view(1,-1),torch.ones(len(theta)).view(1,-1)],dim=0).T.view(len(theta),3,3)


def R_vec_y(theta):
        c, s = torch.cos(theta).view(1,-1), torch.sin(theta).view(1,-1)
        return torch.cat([c,torch.zeros(len(theta)).view(1,-1), s, torch.zeros(len(theta)).view(1,-1), torch.ones(len(theta)).view(1,-1),torch.zeros(len(theta)).view(1,-1), -s, torch.zeros(len(theta)).view(1,-1),c],dim=0).T.view(len(theta),3,3)

def _get_rotated_orientation(coords,pred):
        
        coords = torch.tensor(coords).view(-1,9)
        pred = torch.tensor(pred).view(-1,9)
        
        pred_r = pred[:,[0,3,6]]
        pred_theta = pred[:,[1,4,7]]
        pred_phi = pred[:,[2,5,8]]
        
        coords_r =  pred_r
        coords_theta = coords[:,[1,4,7]]
        coords_phi = coords[:,[2,5,8]]
        
        #print("Pred shape", pred.shape,"True shape",coords.shape)
        
        x_coord_n  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.cos(coords_phi[:,0]).view(-1,1)
        y_coord_n  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.sin(coords_phi[:,0]).view(-1,1)
        z_coord_n  = coords_r[:,0].view(-1,1)*torch.cos(coords_theta[:,0]).view(-1,1)
        
        x_coord_ca   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.cos(coords_phi[:,1]).view(-1,1)
        y_coord_ca   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.sin(coords_phi[:,1]).view(-1,1)
        z_coord_ca   = coords_r[:,1].view(-1,1)*torch.cos(coords_theta[:,1]).view(-1,1).view(-1,1)
        
        x_coord_c   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.cos(coords_phi[:,2]).view(-1,1)
        y_coord_c   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.sin(coords_phi[:,2]).view(-1,1)
        z_coord_c   = coords_r[:,2].view(-1,1)*torch.cos(coords_theta[:,2]).view(-1,1)
        
        
        rot_theta_n = R_vec_y(pred_theta[:,0].flatten()).float()
        rot_theta_ca = R_vec_y(pred_theta[:,1].flatten()).float()
        rot_theta_c = R_vec_y(pred_theta[:,2].flatten()).float()
        
        rot_phi_n = R_vec_z(pred_phi[:,0]).float()
        rot_phi_ca = R_vec_z(pred_phi[:,1]).float()
        rot_phi_c = R_vec_z(pred_phi[:,2]).float()
        
        cart_coord_n = torch.cat([x_coord_n,y_coord_n,z_coord_n]).view(-1,3).float()
        cart_coord_ca = torch.cat([x_coord_ca,y_coord_ca,z_coord_ca]).view(-1,3).float()
        cart_coord_c = torch.cat([x_coord_c,y_coord_c,z_coord_c]).view(-1,3).float()
        
        
        cart_coord_n_final = torch.matmul(torch.matmul(rot_phi_n,rot_theta_n),cart_coord_n.view(-1,3,1)).view(-1,3)
        cart_coord_ca_final = torch.matmul(torch.matmul(rot_phi_ca,rot_theta_ca),cart_coord_ca.view(-1,3,1)).view(-1,3)
        cart_coord_c_final = torch.matmul(torch.matmul(rot_phi_c,rot_theta_c),cart_coord_c.view(-1,3,1)).view(-1,3)
        
        #polar_n_r = torch.sqrt(torch.square(cart_coord_n_final).sum(dim=1)).view(-1,1)
        #polar_n_theta = torch.acos(torch.div(cart_coord_n_final[:,2],polar_n_r)).view(-1,1)
        #polar_n_phi = torch.atan(torch.div(cart_coord_n_final[:,1],cart_coord_n_final[:,0]))
        
        #polar_ca_r = torch.sqrt(torch.square(cart_coord_ca_final).sum(dim=1)).view(-1,1)
        #polar_ca_theta = torch.acos(torch.div(cart_coord_ca_final[:,2],polar_ca_r)).view(-1,1)
        #polar_ca_phi = torch.atan(torch.div(cart_coord_ca_final[:,1],cart_coord_ca_final[:,0]))
        
        
        #polar_c_r = torch.sqrt(torch.square(cart_coord_c_final).sum(dim=1)).view(-1,1)
        #polar_c_theta = torch.acos(torch.div(cart_coord_c_final[:,2],polar_c_r)).view(-1,1)
        #polar_c_phi = torch.atan(torch.div(cart_coord_c_final[:,1],cart_coord_c_final[:,0]))
        
        
        final_coords = torch.cat([cart_coord_n_final,cart_coord_ca_final,cart_coord_c_final],dim=1)
        if torch.isnan(final_coords).any() == True: print("Getting Nan here")
        
        return final_coords.numpy()
    
    


def _get_pairwise_dist(euclid_coords,edge_index):
    
    starting = torch.index_select(euclid_coords,0,edge_index[0].long())
    ending = torch.index_select(euclid_coords,0,edge_index[1].long())
    diff = torch.square(starting - ending).sum(dim=1).view(len(starting),3)
        
    return final_dist

def _get_pairwise_angle(angle_coords,edge_index):
        
    starting = torch.index_select(angle_coords,0,edge_index[0].long())
    ending = torch.index_select(angle_coords,0,edge_index[1].long())
    diff = starting - ending
        
    return diff

def _rbf_weight(d_ij):

    alpha_dist = d_ij[:,1].view(len(d_ij),1)
    diff_scaled = torch.div(alpha_dist,10*torch.ones(len(alpha_dist)).view(len(d_ij),1))
    RBF = torch.exp(-diff_scaled)
        
    return RBF 

def clean_edges(edge_list,node_to_remove):
    edge_start = edge_list[0]
    edge_end = edge_list[1]
    edge_start_temp = []
    edge_end_temp = []
    for idx in range(len(edge_start)):
        if edge_start[idx]!= node_to_remove : 
            edge_start_temp.append(edge_start[idx])
            edge_end_temp.append(edge_end[idx])
        
    final_edges = torch.tensor([edge_start_temp,edge_end_temp])
    return final_edges

def _get_dihedrals(coords):
    
    eps=1e-7
    # N,Ca,C coordinates are first three
    coords = np.array(coords).reshape(len(coords),3,3)
    X = coords[:,:3,:].reshape(1,3*coords.shape[0],3)
    
    # Shifted slices of unit vectors
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(torch.tensor(dX), dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]
    
    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
    
    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (3,0), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    phi, psi, omega = torch.unbind(D,-1)

    angle_coord = torch.cat([phi.view(len(coords),1),psi.view(len(coords),1),omega.view(len(coords),1)],dim=1)
    
    return angle_coord


def _get_dihedrals_uncond(coords):
    
    eps=1e-7
    # N,Ca,C coordinates are first three
    coords = np.array(coords).reshape(len(coords),3,3)
    X = coords[:,:3,:].reshape(1,3*coords.shape[0],3)
    
    # Shifted slices of unit vectors
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(torch.tensor(dX), dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]
    
    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
    
    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (3,0), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    phi, psi, omega = torch.unbind(D,-1)

    angle_coord = torch.cat([phi.view(len(coords),1),psi.view(len(coords),1),omega.view(len(coords),1)],dim=1)
    
    return angle_coord



        
def gather_nodes(nodes,neighbor_idx):
    
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    #print(neighbors_flat.shape,nodes.shape)
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    
    return neighbor_features
        
def get_seq_and_coord(cdr_type,file_path):
    
    final_ab_seq = []
    final_ag_seq = []
    final_ab_ang_coord = []
    final_ag_ang_coord = []
    final_ab_euc_coord = []
    final_ag_euc_coord = []
    
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())
        for entry in contents:
          
            total_length_cdr = len(entry['ab_seq'])
            ab_seq = entry['ab_seq']
            

            euclid_coords_ab = np.array(entry['coords_ab']).reshape(total_length_cdr,3,3)
            euclid_coords_ab = euclid_coords_ab[:,:3,:]
            
            ag_seq = entry['ag_seq']
   
            euclid_coords_ag = np.array(entry['coords_ag']).reshape(len(ag_seq),3,3)
            euclid_coords_ag = euclid_coords_ag[:,:3,:]
            
            final_ab_seq.append(ab_seq)

            final_ab_euc_coord.append(euclid_coords_ab)
            final_ag_seq.append(ag_seq)
            final_ag_euc_coord.append(euclid_coords_ag)

   
    return final_ab_seq,final_ab_euc_coord,final_ag_seq,final_ag_euc_coord




def get_seq_and_coord_whole(cdr_type,file_path):
    
    final_ab_seq = []
    final_ag_seq = []
    final_rest_seq  = []
    final_rest_coord = []
    final_ab_ang_coord = []
    final_ag_ang_coord = []
    final_ab_euc_coord = []
    final_ag_euc_coord = []
    
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())
        for entry in contents:
          
            total_length_cdr = len(entry['ab_seq'])
            ab_seq = entry['ab_seq']
            if len(ab_seq)<2: continue
            angle_coords_ab = _get_dihedrals(entry['coords_ab'])
            euclid_coords_ab = np.array(entry['coords_ab']).reshape(total_length_cdr,3,3)
            euclid_coords_ab = euclid_coords_ab[:,:3,:]
            
            ag_seq = entry['ag_seq']
            angle_coords_ag = _get_dihedrals(entry['coords_ag'])
            euclid_coords_ag = np.array(entry['coords_ag']).reshape(len(ag_seq),3,3)
            euclid_coords_ag = euclid_coords_ag[:,:3,:]
            rest_coords = np.array(entry['coords_rest']).reshape(len(entry['rest_seq']),3,3)
            
            final_rest_seq.append(entry['rest_seq'])
            final_rest_coord.append(rest_coords)
            final_ab_seq.append(ab_seq)
            final_ab_ang_coord.append(angle_coords_ab)
            final_ab_euc_coord.append(euclid_coords_ab)
            final_ag_seq.append(ag_seq)
            final_ag_euc_coord.append(euclid_coords_ag)
            final_ag_ang_coord.append(angle_coords_ag)
   
    return final_ab_seq,final_ab_ang_coord,final_ab_euc_coord,final_ag_seq,final_ag_ang_coord,final_ag_euc_coord,final_rest_seq,final_rest_coord



def get_seq_and_coord_protein(file_path):
    
    final_seq = []
    final_coord =[]
    final_ang_coord = []
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())
        for entry in contents:
            seq = entry['seq']
            coords = np.array(entry['coords']).reshape(-1,3,3)
            angle_coords = _get_dihedrals(coords)
            phi = angle_coords[:,0].view(-1,1)
            psi = angle_coords[:,1].view(-1,1)
            omega = angle_coords[:,2].view(-1,1)
            angle_feat = torch.cat((torch.cos(phi),torch.cos(psi),torch.cos(omega),torch.sin(phi),torch.sin(psi),torch.sin(omega)),dim=1).view(-1,6)
            
            final_seq.append(seq)
            final_coord.append(coords)
            final_ang_coord.append(angle_feat)
    return final_seq,final_coord,final_ang_coord


def get_seq_and_coord_uncond(cdr_type,file_path):
    
    final_ab_seq = []
    final_ab_ang_coord = []
    final_ab_euc_coord = []
    final_pdb = []
    with open(file_path) as f:
        all_lines = f.readlines()
        for idx in all_lines:
            entry  = json.loads(idx)
            pdb = entry['pdb']
            if str(cdr_type) not in entry['cdr']: continue
            
            first_location = entry['cdr'].index(str(cdr_type))
            last_location = entry['cdr'].rindex(str(cdr_type))
           
            
            if first_location == last_location: continue
            if abs(first_location - last_location) >50: continue
            
            first_location = first_location -1
            last_location  = last_location +1
            total_length_cdr = len(entry['seq'][first_location:last_location])
            ab_seq = entry['seq'][first_location:last_location]
            if len(ab_seq) ==0: continue
            N_coords = np.array(entry['coords']['N'],dtype=float).reshape(-1,1,3)
            C_coords = np.array(entry['coords']['C'],dtype=float).reshape(-1,1,3)
            Ca_coords = np.array(entry['coords']['CA'],dtype=float).reshape(-1,1,3)
            
            antibody_coords_total = np.concatenate([N_coords,Ca_coords,C_coords],axis=1)
            
            euclid_coords_ab = antibody_coords_total[first_location:last_location]
            angle_coords_ab = _get_dihedrals_uncond(euclid_coords_ab)

            final_ab_seq.append(ab_seq)
            final_ab_ang_coord.append(angle_coords_ab)
            final_ab_euc_coord.append(euclid_coords_ab)
            final_pdb.append(pdb)
   
    return final_ab_seq,final_ab_ang_coord,final_ab_euc_coord,final_pdb



def get_seq_and_coord_uncond_whole(cdr_type,file_path):
    
    final_ab_seq = []
    final_ab_ang_coord = []
    final_ab_euc_coord = []
    final_pdb = []
    final_before_seq = []
    final_before_coord = []
    with open(file_path) as f:
        all_lines = f.readlines()
        for idx in all_lines:
            entry  = json.loads(idx)
            pdb = entry['pdb']
            if str(cdr_type) not in entry['cdr']: continue
            
            first_location = entry['cdr'].index(str(cdr_type))
            last_location = entry['cdr'].rindex(str(cdr_type))
           
            
            if first_location == last_location: continue
            if abs(first_location - last_location) >50: continue
            
            first_location = first_location -1
            last_location  = last_location +1
            total_length_cdr = len(entry['seq'][first_location:last_location])
            ab_seq = entry['seq'][first_location:last_location]
            before_ab_seq = entry['seq'][:first_location+1]
            if len(ab_seq) ==0: continue
            N_coords = np.array(entry['coords']['N'],dtype=float).reshape(-1,1,3)
            C_coords = np.array(entry['coords']['C'],dtype=float).reshape(-1,1,3)
            Ca_coords = np.array(entry['coords']['CA'],dtype=float).reshape(-1,1,3)
            
            antibody_coords_total = np.concatenate([N_coords,Ca_coords,C_coords],axis=1)
            
            if np.isnan(antibody_coords_total.numpy()).any() == True: continue
                
            euclid_coords_ab = antibody_coords_total[first_location:last_location]
            angle_coords_ab = _get_dihedrals_uncond(euclid_coords_ab)
            rest_coords_ab =  antibody_coords_total[:first_location+1]
                
            final_before_seq.append(before_ab_seq)
            final_ab_seq.append(ab_seq)
            final_ab_ang_coord.append(angle_coords_ab)
            final_ab_euc_coord.append(euclid_coords_ab)
            final_pdb.append(pdb)
            final_before_coord.append(rest_coords_ab)
   
    return final_ab_seq,final_ab_ang_coord,final_ab_euc_coord,final_pdb,final_before_seq,final_before_coord




def get_seq_and_coord_whole_protseed(cdr_type,file_path):
    
    final_ab_seq = []
    final_ag_seq = []
    final_before_seq  = []
    final_before_coord = []
    final_after_seq  = []
    final_after_coord = []
    final_ab_euc_coord = []
    final_ag_euc_coord = []
    
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())
        for entry in contents:
          
            total_length_cdr = len(entry['ab_seq'])
            ab_seq = entry['ab_seq']
            if len(ab_seq)<2: continue
            euclid_coords_ab = np.array(entry['coords_ab']).reshape(total_length_cdr,3,3)
            euclid_coords_ab = euclid_coords_ab[:,:3,:]
            
            ag_seq = entry['ag_seq']
            angle_coords_ag = _get_dihedrals(entry['coords_ag'])
            euclid_coords_ag = np.array(entry['coords_ag']).reshape(len(ag_seq),3,3)
            euclid_coords_ag = euclid_coords_ag[:,:3,:]
            if len(entry['coords_after'])!=len(entry['after_seq']): continue
            before_coords = np.array(entry['coords_before']).reshape(len(entry['before_seq']),3,3)
            after_coords = np.array(entry['coords_after']).reshape(len(entry['after_seq']),3,3)
            
            final_before_seq.append(entry['before_seq'])
            final_after_seq.append(entry['after_seq'])
            final_before_coord.append(before_coords)
            final_after_coord.append(after_coords)
            
            final_ab_seq.append(ab_seq)
            final_ab_euc_coord.append(euclid_coords_ab)
            final_ag_seq.append(ag_seq)
            final_ag_euc_coord.append(euclid_coords_ag)
   
    return final_ab_seq,final_ab_euc_coord,final_ag_seq,final_ag_euc_coord,final_before_seq,final_before_coord,final_after_seq,final_after_coord




def get_graph_data(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Ag_seq,Ag_ang_coord,Ag_euc_coord = get_seq_and_coord(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_ang_coord = Ag_ang_coord[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)
        antigen_len = len(antigen_seq)
        
        for residue in list(antibody_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
         
        #print(antigen_seq)
        for residue in list(antigen_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ag_hot_encoding.append(hot_encoder)
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(len(antibody_seq),20)
        ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)
        
        pos_array = np.array(antibody_euc_coord[:,2,:]).reshape(antibody_cdr_len,3)
        
        edges_ab = radius_graph(torch.tensor(pos_array),r=10,loop=False)
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        
        #Fully connect the Antigen graph
        
        for i in range(antibody_cdr_len,antibody_cdr_len+antigen_len,1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
        
        
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        #Appending position features
        
        alpha_carbon_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
        alpha_carbon_antibody = np.array(antibody_euc_coord).reshape(len(antibody_euc_coord),3,3)
        
        local_atom = alpha_carbon_antibody[0]
        
        alpha_carbon_antibody = alpha_carbon_antibody - local_atom*np.ones_like(alpha_carbon_antibody)
        
        antigen_pos_features = torch.cat([torch.tensor(alpha_carbon_antigen[:,1,:]).view(-1,1,3),torch.tensor(antigen_ang_coord).view(-1,1,3)],dim=1)
        antibody_pos_features = torch.cat([torch.tensor(alpha_carbon_antibody[:,1,:]).view(-1,1,3),torch.tensor(antibody_ang_coord).view(-1,1,3)],dim=1)
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features.view(antibody_cdr_len,6)],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        Input_ab_angle_coords = np.random.vonmises(0, 1, (antibody_cdr_len,3)).reshape(-1,1,3)
        Input_ab_euclid_coords = np.random.standard_normal((antibody_cdr_len,3)).reshape(-1,1,3)
        
        Input_ab_coords = torch.tensor(np.concatenate([Input_ab_euclid_coords,Input_ab_angle_coords],axis=1))
        
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords.view(antibody_cdr_len,6)],dim=1)
        

        data = Data(x=Final_input_anitbody_features, edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), y=Final_target_antibody_features,antigen_labels=ag_label_features,antigen_pos=antigen_pos_features)
        final_data.append(data)
        
    return final_data
   


def loss_function_polar(y_pred,y_true):
    
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_true[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    pred_r = y_pred[:,20].view(-1,1)
    true_r = y_true[:,20].view(-1,1)
    
    r_loss = nn.SmoothL1Loss(reduction='mean')
    loss_val = r_loss(pred_r,true_r)
    
    
    pred_angle = y_pred[:,21:23].view(-1,2)
    true_angle = y_true[:,21:23].view(-1,2)
    
    diff_angle = pred_angle - true_angle
    loss_per_angle = torch.mean(1- torch.square(torch.cos(diff_angle)),0).view(-1,2)
    total_angle_loss  = loss_per_angle.sum(dim=1)
    
    #print("CE loss",loss_ce)
    #print("Dihedral Angle loss",total_angle_loss)
    #print("dr loss",loss_val)
    
    total_loss = loss_ce + 0.8*(loss_val + total_angle_loss)
    
    return total_loss


def loss_function_vm_with_side_chains(y_pred,y_true):
    
    kappa = 10
    pred_labels = y_pred[:,:20].view(-1,20).to(y_pred.device)
    truth_labels = y_true[:,:20].view(-1,20).to(y_pred.device)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    
    pred_coords = y_pred[:,20:29].view(-1,3,3).to(y_pred.device)
    true_coords = y_true[:,20:29].view(-1,3,3).to(y_pred.device)
    
    pred_r = pred_coords[:,:,:1].reshape(-1,1).to(y_pred.device)
    true_r = true_coords[:,:,:1].reshape(-1,1).to(y_pred.device)
    
    #r_loss = nn.MSELoss(reduction='mean')
    #loss_val = r_loss(pred_r,true_r)
    
    pred_angle = pred_coords[:,:,1:3].reshape(-1,2).to(y_pred.device)
    true_angle = true_coords[:,:,1:3].reshape(-1,2).to(y_pred.device)
    
    diff_angle = pred_angle - true_angle
    m = torch.distributions.von_mises.VonMises(torch.tensor([0]).to(y_pred.device), torch.tensor([kappa]).to(y_pred.device))
    nll = - m.log_prob(diff_angle).view(-1,2)
    total_angle_loss = torch.mean(nll,dim=0).sum()
    
    normal_lkl = torch.distributions.normal.Normal(torch.tensor([0]).to(y_pred.device), torch.tensor([0.314]).to(y_pred.device))
    r_diff = pred_r - true_r
    nll_r = - normal_lkl.log_prob(r_diff).view(-1,1)
    loss_val = torch.mean(nll_r,dim=0)
    
    pred_polar_coord = y_pred[:,20:29].view(-1,9)
    truth_polar_coord = y_true[:,20:29].view(-1,9)
    
    Cart_pred,Cart_truth = _get_cartesian(pred_polar_coord,truth_polar_coord)
    cart_lkl = torch.distributions.normal.Normal(torch.tensor([0]).to(y_pred.device), torch.tensor([0.5]).to(y_pred.device)) #1,0.4
    
    cart_loss = nn.MSELoss(reduction='mean')
    loss_cart = cart_loss(Cart_pred,Cart_truth).sum()
    
    diff_cart = Cart_pred -  Cart_truth
    nll_cart = - cart_lkl.log_prob(diff_cart).view(-1,9)
    loss_cart = torch.mean(nll_cart,dim=0).sum()
    
    
    
    
    #print("predicted",pred_polar_coord)
    #print("truth",truth_polar_coord)
    #print("radius loss",loss_val)
    #rint("angle nll loss",total_angle_loss)
    #print("cart loss",loss_cart)
    total_loss = loss_ce + 0.8*(loss_val + total_angle_loss) #+ 0.8*loss_cart
    
    return total_loss



def loss_function_vm_with_side_chains_angle(y_pred,y_true,batch_size):
    
    kappa = 10
    pred_labels = y_pred[:,:20].view(-1,20).to(y_pred.device)
    truth_labels = y_true[:,:20].view(-1,20).to(y_pred.device)
    
    celoss = nn.CrossEntropyLoss(reduction='sum')
    loss_ce = celoss(pred_labels,truth_labels)/batch_size

    pred_coords = y_pred[:,20:29].view(-1,3,3).to(y_pred.device)
    true_coords = y_true[:,20:29].view(-1,3,3).to(y_pred.device)
    
    pred_r = pred_coords[:,:,:1].reshape(-1,1).to(y_pred.device)
    true_r = true_coords[:,:,:1].reshape(-1,1).to(y_pred.device)
    
    #print(pred_r,true_r)
    #r_loss = nn.MSELoss(reduction='mean')
    #loss_val = r_loss(pred_r,true_r)
    
    pred_angle = pred_coords[:,:,1:3].reshape(-1,2).to(y_pred.device)
    true_angle = true_coords[:,:,1:3].reshape(-1,2).to(y_pred.device)
    
    diff_angle = pred_angle - true_angle
    m = torch.distributions.von_mises.VonMises(torch.tensor([0]).to(y_pred.device), torch.tensor([kappa]).to(y_pred.device))
    nll = - m.log_prob(diff_angle).view(-1,2)
    total_angle_loss = torch.mean(nll,dim=0).sum()
    
    
    #r_loss = nn.SmoothL1Loss(reduction='mean')
    #loss_val = r_loss(pred_r,true_r)
    
    normal_lkl = torch.distributions.normal.Normal(torch.tensor([0]).to(y_pred.device), torch.tensor([0.314]).to(y_pred.device))
    r_diff = pred_r - true_r
    nll_r = - normal_lkl.log_prob(r_diff).view(-1,1)
    loss_val = torch.mean(nll_r,dim=0)

    total_loss = loss_ce + 0.8*(loss_val + total_angle_loss)
    
    return total_loss

def loss_function_vm_with_side_chains_rolled(y_pred,y_true,batch):
    
    kappa = 10
    pred_labels = y_pred[:,:20].view(-1,20).to(y_pred.device)
    truth_labels = y_true[:,:20].view(-1,20).to(y_pred.device)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    batch_size = batch.batch[-1].item() + 1
    Cart_pred = y_pred[:,20:29].to(y_pred.device).view(-1,3,3)
    Cart_truth = y_true[:,20:29].to(y_pred.device).view(-1,3,3)
    
    for entry in range(len(Cart_pred)):
        if entry == 0: 
            pred_polar_coord[entry] = pred_polar_coord[entry] + first_residue_coord
            truth_polar_coord[entry] = truth_polar_coord[entry] + first_residue_coord
        else:
            pred_polar_coord[entry] = pred_polar_coord[entry] + pred_polar_coord[entry-1]
            truth_polar_coord[entry] = truth_polar_coord[entry] + truth_polar_coord[entry-1]
    #cart_lkl = torch.distributions.normal.Normal(torch.tensor([0]).to(y_pred.device), torch.tensor([0.5]).to(y_pred.device)) #1,0.4
    
    #diff_cart = Cart_pred -  Cart_truth
    #nll_cart = - cart_lkl.log_prob(diff_cart).view(-1,9)
    #loss_cart = torch.mean(nll_cart,dim=0).sum()
    
    mseloss = nn.MSELoss()
    loss_mse = mseloss(Cart_pred,Cart_truth)
    

    total_loss = loss_ce + loss_mse
    
    return total_loss


def loss_function_protein(y_pred,y_true,batch_size):
    
    kappa = 10
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_true[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss(reduction='sum',label_smoothing=0.0)
    loss_ce = celoss(pred_labels,truth_labels)
    
    total_loss = torch.div(loss_ce,torch.tensor([batch_size]).float().view(-1,1))
    
    return total_loss




def _get_cartesian(pred_polar_coord,truth_polar_coord):
    
    pred_r = pred_polar_coord[:,[0,3,6]]
    pred_theta = pred_polar_coord[:,[1,4,7]]
    pred_phi = 3.14/2 - pred_polar_coord[:,[2,5,8]]
        
    coords_r =  truth_polar_coord[:,[0,3,6]]
    coords_theta = truth_polar_coord[:,[1,4,7]]
    coords_phi = 3.14/2 -truth_polar_coord[:,[2,5,8]]
    
    Cart_true = _transform_to_cart(coords_r,coords_theta,coords_phi)
    Cart_pred = _transform_to_cart(pred_r,pred_theta,pred_phi)
    
    return Cart_true,Cart_pred
        

def _transform_to_cart(coords_r,coords_theta,coords_phi):
    
    x_coord_n_true  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.cos(coords_phi[:,0]).view(-1,1)
    y_coord_n_true  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.sin(coords_phi[:,0]).view(-1,1)
    z_coord_n_true  = coords_r[:,0].view(-1,1)*torch.cos(coords_theta[:,0]).view(-1,1)
        
    x_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.cos(coords_phi[:,1]).view(-1,1)
    y_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.sin(coords_phi[:,1]).view(-1,1)
    z_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.cos(coords_theta[:,1]).view(-1,1).view(-1,1)
        
    x_coord_c_true   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.cos(coords_phi[:,2]).view(-1,1)
    y_coord_c_true   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.sin(coords_phi[:,2]).view(-1,1)
    z_coord_c_true   = coords_r[:,2].view(-1,1)*torch.cos(coords_theta[:,2]).view(-1,1)
    
    Cart = torch.cat([x_coord_n_true,y_coord_n_true,z_coord_n_true,x_coord_ca_true,y_coord_ca_true,z_coord_ca_true,x_coord_c_true,y_coord_c_true,z_coord_c_true],dim=1).view(-1,9)
    
    return Cart
    
    

def loss_function_vm_with_side_chains_v2(y_pred,y_true):
    
    kappa = 10
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_true[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    pred_coords = y_pred[:,20:29].view(-1,3,3)
    true_coords = y_true[:,20:29].view(-1,3,3)
    
    pred_r = pred_coords[:,:,:1].reshape(-1,1)
    true_r = true_coords[:,:,:1].reshape(-1,1)
    
    r_loss = nn.SmoothL1Loss(reduction='mean')
    loss_val = r_loss(pred_r,true_r)
    
    pred_angle_phi = pred_coords[:,:,1].reshape(-1,1)
    true_angle_phi = true_coords[:,:,1].reshape(-1,1)
    
    pred_angle_psi = pred_coords[:,:,2].reshape(-1,1)
    true_angle_psi = true_coords[:,:,2].reshape(-1,1)
    
    loss_phi = torch.square(torch.cos(pred_angle_phi) - torch.cos(true_angle_phi)) + torch.square(torch.sin(pred_angle_phi) - torch.sin(true_angle_phi))
    
    loss_psi = torch.square(torch.cos(pred_angle_psi) - torch.cos(true_angle_psi)) + torch.square(torch.sin(pred_angle_psi) - torch.sin(true_angle_psi))
    
    total_angle_loss = torch.mean(loss_phi) + torch.mean(loss_psi)
    
    #Distance between residues loss
    
    #pred_roll = torch.roll(pred_coords,1,0)
    #truth_roll = torch.roll(true_coords,1,0)
    
    #truth_inter_distance = true_coords[:,1,:] - truth_roll[:,1,:]
    #pred_inter_distance =pred_coords[:,1,:] - pred_roll[:,1,:]
    
    #inter_dist_loss = nn.SmoothL1Loss(reduction='mean')
    #loss_inter = inter_dist_loss(pred_inter_distance,truth_inter_distance)
    
    #print("CE loss",loss_ce)
    #print("Dihedral Angle loss",total_angle_loss)
    #print("dr loss",loss_val)
    
    total_loss = loss_ce + loss_val + total_angle_loss
    
    return total_loss


def loss_function_polar_v2(y_pred,y_true):
    
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_true[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    pred_r = y_pred[:,20].view(-1,1)
    true_r = y_true[:,20].view(-1,1)
    
    r_loss = nn.SmoothL1Loss(reduction='mean')
    loss_val = r_loss(pred_r,true_r)
    
    
    pred_angle_phi = y_pred[:,21].view(-1,1)
    true_angle_phi = y_true[:,21].view(-1,1)
    
    pred_angle_psi = y_pred[:,22].view(-1,1)
    true_angle_psi = y_true[:,22].view(-1,1)
    
    loss_phi = torch.square(torch.cos(pred_angle_phi) - torch.cos(true_angle_phi)) + torch.square(torch.sin(pred_angle_phi) - torch.sin(true_angle_phi))
    
    loss_psi = torch.square(torch.cos(pred_angle_psi) - torch.cos(true_angle_psi)) + torch.square(torch.sin(pred_angle_psi) - torch.sin(true_angle_psi))
    
    total_angle_loss = torch.mean(loss_phi) + torch.mean(loss_psi)
    #print("CE loss",loss_ce)
    #print("Dihedral Angle loss",total_angle_loss)
    #print("dr loss",loss_val)
    
    total_loss = loss_ce + 0.8*(loss_val + total_angle_loss)
    
    return total_loss
    
    
    
    
#get_graph_data(1,train_json_file)


def get_graph_data_polar_with_sidechains(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Ag_seq,Ag_ang_coord,Ag_euc_coord = get_seq_and_coord(cdr_type,file_path)
    
    #Comb = list(zip(Ab_seq,Ab_ang_coord,Ab_euc_coord,Ag_seq,Ag_ang_coord,Ag_euc_coord))
    #visited = set()
    #Output= []
    
    #for idx,a, b,c,d,e,f in enumerate(Comb):
    #    if not a in visited:
    #        visited.add(a)
    #        Output.append(idx)
        

    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_ang_coord = Ag_ang_coord[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)
        antigen_len = len(antigen_seq)
        
        for residue in list(antibody_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
         
        #print(antigen_seq)
        for residue in list(antigen_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ag_hot_encoding.append(hot_encoder)
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(len(antibody_seq),20)
        ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)
        
        pos_antibody = np.array(antibody_euc_coord).reshape(len(antibody_euc_coord),3,3)
        pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
    
        
        local_atom = pos_antibody[0][1][:]*np.ones_like(pos_antibody[0][1][:])
        all_coords_trans = pos_antibody - local_atom*np.ones_like(pos_antibody)
        all_coords_ab = all_coords_trans.reshape(-1,3)
        
        local_atom = pos_antibody[0][1][:]*np.ones_like(pos_antigen[0][1][:])
        all_coords_trans = pos_antigen - local_atom*np.ones_like(pos_antigen)
        all_coords_ag = all_coords_trans.reshape(-1,3)

        
        r_ab,t_ab,z_ab = cartesian_to_spherical(all_coords_ab[:,0].reshape(-1,1),all_coords_ab[:,1].reshape(-1,1),all_coords_ab[:,2].reshape(-1,1))
        r_ag,t_ag,z_ag = cartesian_to_spherical(all_coords_ag[:,0].reshape(-1,1),all_coords_ag[:,1].reshape(-1,1),all_coords_ag[:,2].reshape(-1,1))
        antibody_pos_features = torch.cat([torch.tensor(r_ab).view(-1,1),torch.tensor(t_ab).view(-1,1),torch.tensor(z_ab).view(-1,1)],dim=1).view(-1,9)
        antigen_pos_features = torch.cat([torch.tensor(r_ag).view(-1,1),torch.tensor(t_ag).view(-1,1),torch.tensor(z_ag).view(-1,1)],dim=1).view(-1,9)
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
           
        edge_s = []
        edge_f = []
        order_ab = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=1000,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
                order_ab.append(1)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        order_ag = []
        
        #Fully connect the Antigen graph
        for i in range(len(ab_label_features),len(ab_label_features)+len(ag_label_features),1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
                order_ag.append(2)
                
        order_final = order_ab + order_ag
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),len(antibody_seq))).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)
        
        Final_input_antigen_features = torch.cat([ag_label_features,antigen_pos_features],dim=1)
       
        Final_input_features = torch.cat([Final_input_anitbody_features,Final_input_antigen_features],dim=0)
        
        amino_index = torch.tensor([i for i in range(len(Final_input_features))]).view(-1,1).float()
        #Final_target_features = 

        data = Data(x=Final_input_features, edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), order = torch.tensor(order_final).view(-1,1),y=Final_target_antibody_features,antigen_labels=ag_label_features,antigen_pos=antigen_pos_features, ag_len= torch.tensor([len(antigen_seq)]).view(-1,1),ab_len= torch.tensor([len(antibody_seq)]).view(-1,1),a_index = amino_index.view(1,-1))
                    
        final_data.append(data)
        
    return final_data








def get_graph_data_polar_uncond(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord = get_seq_and_coord_uncond(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)
        
        for residue in list(antibody_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(len(antibody_seq),20)
        C_alpha_ab = antibody_euc_coord[:,1,:].reshape(antibody_cdr_len,3)
        
        local_atom = C_alpha_ab[0]
        C_alpha_ab = C_alpha_ab - local_atom*np.ones_like(C_alpha_ab)
        
        r_ab,t_ab,z_ab = cartesian_to_spherical(C_alpha_ab[:,0].reshape(-1,1),C_alpha_ab[:,1].reshape(-1,1),C_alpha_ab[:,2].reshape(-1,1))

        antibody_pos_features = torch.cat([torch.tensor(r_ab).view(-1,1),torch.tensor(t_ab).view(-1,1),torch.tensor(z_ab).view(-1,1)],dim=1)
        
        
        #antibody_pos_features[:,0] = abs(antibody_pos_features[:,0])
        
        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(C_alpha_ab)):
            for idx_end in range(len(C_alpha_ab)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        Input_r_coords = torch.linspace(antibody_pos_features[0][0],antibody_pos_features[-1][0],len(antibody_seq)).view(-1,1)
        Input_psi_coords = torch.linspace(antibody_pos_features[0][1],antibody_pos_features[-1][1],len(antibody_seq)).view(-1,1)
        Input_phi_coords = torch.linspace(antibody_pos_features[0][2],antibody_pos_features[-1][2],len(antibody_seq)).view(-1,1)
        
        #Input_angle_coords = torch.tensor(np.random.standard_normal((antibody_cdr_len,2)).reshape(-1,2))
        Input_ab_coords = torch.cat([Input_r_coords,Input_psi_coords,Input_phi_coords],dim=1)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features)
        final_data.append(data)
        
    return final_data


def evaluate_rmsd(y_pred,y_truth):
    pred_polar_coord = y_pred[:,20:23].detach().numpy()
    truth_polar_coord = y_truth[:,20:23].detach().numpy()
    
    pred_polar_coord[0] = np.array([0,0,0]).reshape(-1,3)
    
    return kabsch_rmsd(pred_polar_coord,truth_polar_coord)


def evaluate_rmsd_with_sidechains_rolled(y_initial,y_pred,y_truth,first_residue):
    
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_truth[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    ppl = torch.exp(loss_ce)
    
    initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].detach().numpy().reshape(-1,3,3)
    first_residue_coord = first_residue.detach().numpy().reshape(-1,3,3)
    
    
    for entry in range(len(pred_labels)):
        if entry == 0: 
            pred_polar_coord[entry] = pred_polar_coord[entry] + first_residue_coord
            truth_polar_coord[entry] = truth_polar_coord[entry] + first_residue_coord
        else:
            pred_polar_coord[entry] = pred_polar_coord[entry] + pred_polar_coord[entry-1]
            truth_polar_coord[entry] = truth_polar_coord[entry] + truth_polar_coord[entry-1]
    
    #for first_entry in range(len(pred_labels)):
    #    if first_entry == len(pred_labels)-1: continue
    #    for second_entry in range(first_entry+1,len(pred_labels)):
    #        pred_polar_coord[first_entry] = pred_polar_coord[first_entry].reshape(-1,3,3) + pred_polar_coord[second_entry].reshape(-1,3,3)
     #       truth_polar_coord[first_entry] = truth_polar_coord[first_entry].reshape(-1,3,3) + truth_polar_coord[second_entry].reshape(-1,3,3)
    
    pred_polar_coord = pred_polar_coord[:len(pred_labels)-1].reshape(-1,3,3)
    truth_polar_coord = truth_polar_coord[:len(pred_labels)-1].reshape(-1,3,3)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    Cart_pred[0] = Cart_truth[0]
    Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred,C_alpha_truth)
    
    return rmsd_N,rmsd_Ca,rmsd_C,ppl.item(),rmsd_cart_Ca

def evaluate_rmsd_with_sidechains_angle(y_initial,y_pred,y_truth,first_residue):
    
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_truth[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    ppl = torch.exp(loss_ce)
    
    initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].detach().numpy().reshape(-1,3,3)
    first_residue_coord = first_residue[:,1,:].detach().numpy().reshape(-1,3)
    #pred_polar_coord[0] = truth_polar_coord[0]
    #pred_polar_coord[-1] = truth_polar_coord[-1]
    
    #print(pred_polar_coord)
    #print(truth_polar_coord)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    Cart_pred[0] = Cart_truth[0]
    Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    
    for entry in range(len(C_alpha_pred)):
        if entry == 0: 
            C_alpha_pred[entry] = C_alpha_pred[entry] + first_residue_coord
            C_alpha_truth[entry] = C_alpha_truth[entry] + first_residue_coord
        else:
            C_alpha_pred[entry] =C_alpha_pred[entry] + C_alpha_pred[entry-1]
            C_alpha_truth[entry] = C_alpha_truth[entry] + C_alpha_truth[entry-1]
    
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred,C_alpha_truth)
    
    return rmsd_N,rmsd_Ca,rmsd_C,ppl.item(),rmsd_cart_Ca


def evaluate_rmsd_with_sidechains(y_initial,y_pred,y_truth):
    
    pred_labels = y_pred[:,:20].view(-1,20)
    truth_labels = y_truth[:,:20].view(-1,20)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    ppl = torch.exp(loss_ce)
    
    initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].detach().numpy().reshape(-1,3,3)
    
    #pred_polar_coord[0] = truth_polar_coord[0]
    #pred_polar_coord[-1] = truth_polar_coord[-1]
    
    #print(pred_polar_coord)
    #print(truth_polar_coord)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    Cart_pred[0] = Cart_truth[0]
    Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred,C_alpha_truth)
    
    return rmsd_N,rmsd_Ca,rmsd_C,ppl.item(),rmsd_cart_Ca


def evaluate_rmsd_with_sidechains_cond(y_pred,y_truth):
    
    pred_labels = y_pred[:,:20].view(-1,20).detach().cpu()
    truth_labels = y_truth[:,:20].view(-1,20).detach().cpu()
    
    y_pred_softmax = torch.log_softmax(pred_labels, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    _,y_true = torch.max(truth_labels,dim=1)
    
    #print("prediction",pred_labels)
    #print("truth",truth_labels)
    #print("y_pred_softmax",y_pred_softmax)
    #print("y_pred_tags",y_pred_tags)
    #print("y_true",y_true)
    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    #initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    
    #pred_polar_coord[0] = truth_polar_coord[0]
    #pred_polar_coord[-1] = truth_polar_coord[-1]
    
    #print(pred_polar_coord)
    #print(truth_polar_coord)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    Cart_pred[0] = Cart_truth[0]
    Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred,C_alpha_truth)
    
    return rmsd_N,rmsd_Ca,rmsd_C,acc,rmsd_cart_Ca




def evaluate_rmsd_with_sidechains_cond_angle(y_pred,y_truth,first_residue):
    
    pred_labels = y_pred[:,:20].view(-1,20).detach().cpu()
    truth_labels = y_truth[:,:20].view(-1,20).detach().cpu()
    
    y_pred_softmax = torch.log_softmax(pred_labels, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    _,y_true = torch.max(truth_labels,dim=1)
    
    #print("prediction",pred_labels)
    #print("truth",truth_labels)
    #print("y_pred_softmax",y_pred_softmax)
    #print("y_pred_tags",y_pred_tags)
    #print("y_true",y_true)
    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    #initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    first_residue_coord = first_residue[:,1,:].detach().numpy().reshape(-1,3)
    
    #pred_polar_coord[0] = truth_polar_coord[0]
    #pred_polar_coord[-1] = truth_polar_coord[-1]
    
    #print(pred_polar_coord)
    #print(truth_polar_coord)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    #Cart_pred[0] = Cart_truth[0]
    #Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    
    for entry in range(len(C_alpha_pred)):
        if entry == 0: 
            C_alpha_pred[entry] = C_alpha_pred[entry] + first_residue_coord
            C_alpha_truth[entry] = C_alpha_truth[entry] + first_residue_coord
        else:
            C_alpha_pred[entry] =C_alpha_pred[entry] + C_alpha_pred[entry-1]
            C_alpha_truth[entry] = C_alpha_truth[entry] + C_alpha_truth[entry-1]
    
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred[:-1],C_alpha_truth[:-1])
    
    return rmsd_N,rmsd_Ca,rmsd_C,acc,rmsd_cart_Ca


def evaluate_rmsd_with_sidechains_cond_angle_protseed(y_pred,y_truth,first_residue):
    
    pred_labels = y_pred[:,:20].view(-1,20).detach().cpu()
    truth_labels = y_truth[:,:20].view(-1,20).detach().cpu()
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    ppl = torch.exp(loss_ce)
    
    y_pred_softmax = torch.log_softmax(pred_labels, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    _,y_true = torch.max(truth_labels,dim=1)
    
    #print("prediction",pred_labels)
    #print("truth",truth_labels)
    #print("y_pred_softmax",y_pred_softmax)
    #print("y_pred_tags",y_pred_tags)
    #print("y_true",y_true)
    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    #initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].cpu().detach().numpy().reshape(-1,3,3)
    first_residue_coord = first_residue[:,1,:].detach().numpy().reshape(-1,3)
    
    #pred_polar_coord[0] = truth_polar_coord[0]
    #pred_polar_coord[-1] = truth_polar_coord[-1]
    
    #print(pred_polar_coord)
    #print(truth_polar_coord)
    rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:],truth_polar_coord[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:],truth_polar_coord[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:],truth_polar_coord[:][:,2][:])
    
    
    Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1,9),torch.tensor(truth_polar_coord).view(-1,9))
    #Cart_pred[0] = Cart_truth[0]
    #Cart_pred[-1] = Cart_truth[-1]
    
    C_alpha_pred = Cart_pred[:,3:6].numpy()
    C_alpha_truth = Cart_truth[:,3:6].numpy()
    
    for entry in range(len(C_alpha_pred)):
        if entry == 0: 
            C_alpha_pred[entry] = C_alpha_pred[entry] + first_residue_coord
            C_alpha_truth[entry] = C_alpha_truth[entry] + first_residue_coord
        else:
            C_alpha_pred[entry] =C_alpha_pred[entry] + C_alpha_pred[entry-1]
            C_alpha_truth[entry] = C_alpha_truth[entry] + C_alpha_truth[entry-1]
    
    rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred[:-1],C_alpha_truth[:-1])
    
    return rmsd_N,rmsd_Ca,rmsd_C,ppl.item(),rmsd_cart_Ca,acc



def save_seq_to_file(pred_list,path):
    f = open(path,"w+")
    for pred in pred_list:
        gen = [] 
        gen_seq = ""
        pred_labels = pred[:,:20].view(-1,20).detach().cpu()
        pred = softmax(pred_labels, axis=1)
        final_pred =  np.argmax(pred,axis=1)
        for k in final_pred:
            gen_seq = gen_seq + str(ALPHABET[k])
        
        f.write(str(gen_seq))
        f.write("\n")
        
    f.close()
    
    
    
def evaluate_protein(y_pred,y_truth):
    
    pred_labels = y_pred.view(-1,20).cpu().detach()
    truth_labels = y_truth.view(-1,20).cpu().detach()

    y_pred_softmax = torch.log_softmax(pred_labels, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    _,y_true = torch.max(truth_labels,dim=1)
    
    #pred_labels = pred_labels + 0.5*truth_labels
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
   
    celoss = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.0)
    loss_ce = celoss(pred_labels,truth_labels)
    #print("CEloss shape",loss_ce.shape)
    #print("CEloss values",loss_ce)
    #loss_ce = celoss(pred_labels,truth_labels)
    ppl = torch.exp(loss_ce)
    
    
    return acc,ppl.item()


    
def evaluate_rmsd_with_sidechains_v2(y_initial,y_pred,y_truth):
    
    initial_polar_coord = y_initial[:,20:29].detach().numpy().reshape(-1,3,3)
    pred_polar_coord = y_pred[:,20:29].detach().numpy().reshape(-1,3,3)
    truth_polar_coord = y_truth[:,20:29].detach().numpy().reshape(-1,3,3)
    
    
    pred_cart = _get_rotated_orientation(initial_polar_coord,pred_polar_coord)
    
    x_cart,y_cart,z_cart = spherical_to_cartesian(truth_polar_coord[:,:,:1].reshape(-1,1),truth_polar_coord[:,:,1:2].reshape(-1,1),truth_polar_coord[:,:,2:3].reshape(-1,1))
    
    true_cart = np.concatenate([x_cart,y_cart,z_cart],axis=1).reshape(-1,3,3)
    pred_cart = pred_cart.reshape(-1,3,3)
    pred_cart[0] = true_cart[0]
   

    rmsd_N = kabsch_rmsd(pred_cart[:][:,0][:],true_cart[:][:,0][:])
    rmsd_Ca = kabsch_rmsd(pred_cart[:][:,1][:],true_cart[:][:,1][:])
    rmsd_C = kabsch_rmsd(pred_cart[:][:,2][:],true_cart[:][:,2][:])
    
    return rmsd_N,rmsd_Ca,rmsd_C
    
    
def get_graph_data_polar_uncond_with_side_chains(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Pdb = get_seq_and_coord_uncond(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        pdb_ab = Pdb[entry_number]
        
        ab_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)
        
        for residue in list(antibody_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(len(antibody_seq),20)
        all_coords = antibody_euc_coord.reshape(antibody_cdr_len,3,3)
        
        local_atom = all_coords[0][1][:]*np.ones_like(all_coords[0][1][:])
        all_coords_trans = all_coords - local_atom*np.ones_like(all_coords)
        all_coords_flat = all_coords_trans.reshape(-1,3)
        
        
        r_ab,t_ab,z_ab = cartesian_to_spherical(all_coords_flat[:,0].reshape(-1,1),all_coords_flat[:,1].reshape(-1,1),all_coords_flat[:,2].reshape(-1,1))

        antibody_pos_features = torch.cat([torch.tensor(r_ab).view(-1,1),torch.tensor(t_ab).view(-1,1),torch.tensor(z_ab).view(-1,1)],dim=1).view(-1,9)
        
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
            
    
        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        amino_index = torch.tensor([i for i in range(len(antibody_seq))]).view(-1,1).float()
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),len(antibody_seq))).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features,a_index = amino_index.view(1,-1))
        #print(data)
        final_data.append(data)
        
    return final_data








def get_graph_data_polar_uncond_with_side_chains_rolled(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Pdb = get_seq_and_coord_uncond(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        pdb_ab = Pdb[entry_number]
        
        ab_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)-1
        if len(antibody_seq)<=2: continue
        for residue in list(antibody_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        

        ab_label_features = torch.tensor(ab_hot_encoding[1:]).view(antibody_cdr_len,20)
        
        all_coords = antibody_euc_coord.reshape(antibody_cdr_len+1,3,3)
        
        rolled_coords = all_coords[:len(all_coords)-1].reshape(antibody_cdr_len,3,3)
        
        first_coord = torch.from_numpy(all_coords[0]).view(-1,3,3)

        diff_coords = all_coords[1:] - rolled_coords
    
        antibody_pos_features = torch.from_numpy(diff_coords).view(-1,9)
        
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue

        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        amino_index = torch.tensor([i for i in range(antibody_cdr_len)]).view(-1,1).float()
        temp_coords = antibody_pos_features.view(-1,3,3)
        #print(temp_coords)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features,a_index = amino_index.view(1,-1),first_res=first_coord)
        #print(data)
        final_data.append(data)
        
    return final_data




def get_graph_data_polar_uncond_with_side_chains_angle(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Pdb = get_seq_and_coord_uncond(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        pdb_ab = Pdb[entry_number]
        
        ab_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        #print(len(antibody_seq))
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len<=1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        

        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        
        ab_coords_forward_rolled = torch.roll(all_coords,1,0)
        ab_diff_forward = all_coords - ab_coords_forward_rolled
        
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        
        first_coord = all_coords[0].view(-1,3,3)
        
        ab_diff_forward = ab_diff_forward[1:-1]
        ab_diff_backward = ab_diff_backward[1:-1]
        r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue

        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        amino_index = torch.tensor([i for i in range(antibody_cdr_len)]).view(-1,1).float()
        temp_coords = antibody_pos_features.view(-1,3,3)
        #print(temp_coords)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features,a_index = amino_index.view(1,-1),first_res=first_coord)
        #print(data)
        final_data.append(data)
        
    return final_data




def get_graph_data_polar_uncond_with_side_chains_angle_whole(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Pdb,Rest_seq,Rest_euc_coord = get_seq_and_coord_uncond_whole(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    final_data_rest = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        pdb_ab = Pdb[entry_number]
        
        ab_hot_encoding = []
        ab_rest_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_rest_seq = Rest_seq[entry_number]
        antibody_rest_coord = Rest_euc_coord[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        #print(len(antibody_seq))
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len<=1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        
  
        for residue in list(antibody_rest_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_rest_encoding.append(hot_encoder)
            
            
        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        ab_rest_label_features = torch.tensor(ab_rest_encoding).view(len(antibody_rest_seq[1:-1]),20)
        
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        all_rest_coords = torch.from_numpy(antibody_rest_coord.reshape(len(antibody_rest_coord),9))

        ab_coords_forward_rolled = torch.roll(all_coords,1,0)
        ab_diff_forward = all_coords - ab_coords_forward_rolled
        
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        
        first_coord = all_coords[0].view(-1,3,3)
        
        ab_diff_forward = ab_diff_forward[1:-1]
        ab_diff_backward = ab_diff_backward[1:-1]
        r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue

        rest_ab_coords_forward_rolled = torch.roll(all_rest_coords,1,0)
        rest_ab_diff_forward = all_rest_coords - rest_ab_coords_forward_rolled 
        
        rest_ab_coords_backward_rolled = torch.roll(all_rest_coords,-1,0)
        rest_ab_diff_backward = rest_ab_coords_backward_rolled - all_rest_coords
        
        
        rest_ab_diff_forward = rest_ab_diff_forward[1:-1]
        rest_ab_diff_backward = rest_ab_diff_backward[1:-1]
        r_norm = torch.norm(rest_ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(rest_ab_diff_forward.view(-1,3,3),rest_ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(rest_ab_diff_forward.view(-1,3,3),rest_ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,rest_ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        rest_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        if np.isnan(rest_pos_features.numpy()).any() == True: continue

        edge_index_res = knn_graph(rest_pos_features[:,3:6].view(-1,3),k=5,loop=True)
        
        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        amino_index = torch.tensor([i for i in range(antibody_cdr_len)]).view(-1,1).float()
        temp_coords = antibody_pos_features.view(-1,3,3)
        #print(temp_coords)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features,a_index = amino_index.view(1,-1),first_res=first_coord)
        #print(data)
        data_rest = Data(x=rest_pos_features,label=ab_rest_label_features,edge_index=edge_index_res)
        final_data.append((data,data_rest))
        
    return final_data





def get_graph_data_polar_with_sidechains_angle(cdr_type,file_path,mask):
    
    Ab_seq,Ab_euc_coord,Ag_seq,Ag_euc_coord = get_seq_and_coord(cdr_type,file_path)
    
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        # Converting sequence into labels
        
        antigen_len = len(antigen_seq)
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len<=1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
         
        if '*' in antigen_seq: continue
        for residue in list(antigen_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ag_hot_encoding.append(hot_encoder)
        
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        ab_coords_forward_rolled = torch.roll(all_coords,1,0)
        ab_diff_forward = all_coords - ab_coords_forward_rolled
        
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        
        first_coord = all_coords[0].view(-1,3,3)
        
        ab_diff_forward = ab_diff_forward[1:-1]
        ab_diff_backward = ab_diff_backward[1:-1]
        r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        
        if mask == 0:
            ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)
            pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
            all_coords_ag = torch.from_numpy(antigen_euc_coord.reshape(len(antigen_euc_coord),9))
            

        if mask>0:
            ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)
            pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
            all_coords_ag = torch.from_numpy(antigen_euc_coord.reshape(len(antigen_euc_coord),9))
            number_masked = int(np.ceil(mask*len(ag_label_features)/100))
            entry_masked = random.sample(range(len(ag_label_features)), number_masked)
            for entry in entry_masked:
                ag_label_features[entry] = torch.tensor(float(1/20)*np.ones((1,20))).view(-1,20)
                all_coords_ag[entry] = torch.from_numpy(np.mean(antigen_euc_coord.reshape(len(antigen_euc_coord),9),axis=0))
            
            
        ag_coords_forward_rolled = torch.roll(all_coords_ag,1,0)
        ag_diff_forward = all_coords_ag - ag_coords_forward_rolled
        
        ag_coords_backward_rolled = torch.roll(all_coords_ag,-1,0)
        ag_diff_backward = ag_coords_backward_rolled - all_coords_ag
        
        r_norm_ag = torch.norm(ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
         
            
        mid_angle_ag = torch.acos(F.cosine_similarity(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)    
        cross_vector_ag = torch.cross(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle_ag = torch.acos(F.cosine_similarity(cross_vector_ag,ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        antigen_pos_features = torch.cat((r_norm_ag,mid_angle_ag,normal_angle_ag),dim=2).view(-1,9)

        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
            
        
           
        edge_s = []
        edge_f = []
        order_ab = []

        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
                order_ab.append(1)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        order_ag = []
        
        #Fully connect the Antigen graph
        for i in range(len(ab_label_features),len(ab_label_features)+len(ag_label_features),1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
                order_ag.append(2)
                
        order_final = order_ab + order_ag
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)
        
        Final_input_antigen_features = torch.cat([ag_label_features,antigen_pos_features],dim=1)
       
        Final_input_features = torch.cat([Final_input_anitbody_features,Final_input_antigen_features],dim=0)
        
        amino_index = torch.tensor([i for i in range(len(Final_input_features))]).view(-1,1).float()
        #Final_target_features = 

        
        
        data = Data(x=Final_input_features, edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), order = torch.tensor(order_final).view(-1,1),y=Final_target_antibody_features,antigen_labels=ag_label_features,antigen_pos=antigen_pos_features, ag_len= torch.tensor(len(antigen_seq)).view(-1,1),ab_len= torch.tensor(len(antibody_seq)-2).view(-1,1),a_index = amino_index.view(1,-1),first_res=first_coord)
                    
        final_data.append(data)
        
    return final_data


def get_graph_data_polar_with_sidechains_angle_impsn(cdr_type,file_path):
    
    Ab_seq,Ab_euc_coord,Ag_seq,Ag_euc_coord = get_seq_and_coord(cdr_type,file_path)
    
    final_data = []

    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        # Converting sequence into labels
        
        antigen_len = len(antigen_seq)
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len<=1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
         
        if '*' in antigen_seq: continue
        for residue in list(antigen_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ag_hot_encoding.append(hot_encoder)
        
        
        
        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        ab_coords_forward_rolled = torch.roll(all_coords,1,0)
        ab_diff_forward = all_coords - ab_coords_forward_rolled
        
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        
        first_coord = all_coords[0].view(-1,3,3)
        
        ab_diff_forward = ab_diff_forward[1:-1]
        ab_diff_backward = ab_diff_backward[1:-1]
        r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        
        ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)
        pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
        all_coords_ag = torch.from_numpy(antigen_euc_coord.reshape(len(antigen_euc_coord),9))
            
        ag_coords_forward_rolled = torch.roll(all_coords_ag,1,0)
        ag_diff_forward = all_coords_ag - ag_coords_forward_rolled
        
        ag_coords_backward_rolled = torch.roll(all_coords_ag,-1,0)
        ag_diff_backward = ag_coords_backward_rolled - all_coords_ag
        
        r_norm_ag = torch.norm(ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
         
            
        mid_angle_ag = torch.acos(F.cosine_similarity(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)    
        cross_vector_ag = torch.cross(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle_ag = torch.acos(F.cosine_similarity(cross_vector_ag,ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        antigen_pos_features = torch.cat((r_norm_ag,mid_angle_ag,normal_angle_ag),dim=2).view(-1,9)

        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
        if np.isnan(antigen_pos_features.numpy()).any() == True: continue

           
        edge_s = []
        edge_f = []
        order_ab = []

        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
                order_ab.append(1)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        order_ag = []
        
        #Fully connect the Antigen graph
        for i in range(len(ab_label_features),len(ab_label_features)+len(ag_label_features),1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
                order_ag.append(2)
                
        order_final = order_ab + order_ag
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)
        
        Final_input_antigen_features = torch.cat([ag_label_features,antigen_pos_features],dim=1)
       
        Final_input_features = torch.cat([Input_ab_labels,ag_label_features],dim=0)
        
        amino_index = torch.tensor([i for i in range(len(Final_input_features))]).view(-1,1).float()
        #Final_target_features = 

        pos = torch.cat([Input_ab_coords,antigen_pos_features],dim=0)

        data = Data(x=Final_input_features.float(),pos=pos.float(), edge_attr=torch.tensor(order_final).view(-1,1), edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), order = torch.tensor(order_final).view(-1,1),y=Final_target_antibody_features.float(),antigen_labels=ag_label_features,antigen_pos=antigen_pos_features.float(), ag_len= torch.tensor(len(antigen_seq)).view(-1,1),ab_len= torch.tensor(len(antibody_seq)-2).view(-1,1),a_index = amino_index.view(1,-1),first_res=first_coord)
        
        final_data.append(data)
        
    return final_data

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
def get_graph_data_polar_with_sidechains_angle_whole(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Ag_seq,Ag_ang_coord,Ag_euc_coord,Rest_seq,Rest_coord = get_seq_and_coord_whole(cdr_type,file_path)
    
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        rest_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_ang_coord = Ag_ang_coord[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        rest_seq = Rest_seq[entry_number]
        rest_coord = Rest_coord[entry_number]
        
        # Converting sequence into labels
        
        antigen_len = len(antigen_seq)
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len==1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
         
        #print(antigen_seq)
        for residue in list(antigen_seq):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ag_hot_encoding.append(hot_encoder)
        
        
        for residue in list(rest_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            rest_hot_encoding.append(hot_encoder)
            
        ab_rest_label_features = torch.tensor(rest_hot_encoding).view(len(rest_seq[1:-1]),20)  
            
        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        ab_coords_forward_rolled = torch.roll(all_coords,1,0)
        ab_diff_forward = all_coords - ab_coords_forward_rolled
        
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        
        first_coord = all_coords[0].view(-1,3,3)
        
        ab_diff_forward = ab_diff_forward[1:-1]
        ab_diff_backward = ab_diff_backward[1:-1]
        r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        
        ag_label_features = torch.tensor(ag_hot_encoding).view(len(antigen_seq),20)

        pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
        all_coords_ag = torch.from_numpy(antigen_euc_coord.reshape(len(antigen_euc_coord),9))
        ag_coords_forward_rolled = torch.roll(all_coords_ag,1,0)
        ag_diff_forward = all_coords_ag - ag_coords_forward_rolled
        
        ag_coords_backward_rolled = torch.roll(all_coords_ag,-1,0)
        ag_diff_backward = ag_coords_backward_rolled - all_coords_ag
        
        r_norm_ag = torch.norm(ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
         
            
        mid_angle_ag = torch.acos(F.cosine_similarity(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)    
        cross_vector_ag = torch.cross(ag_diff_forward.view(-1,3,3),ag_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle_ag = torch.acos(F.cosine_similarity(cross_vector_ag,ag_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        antigen_pos_features = torch.cat((r_norm_ag,mid_angle_ag,normal_angle_ag),dim=2).view(-1,9)

        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
        
        
        
        all_rest_coords =  torch.from_numpy(rest_coord.reshape(len(rest_coord),9)) 
        rest_ab_coords_forward_rolled = torch.roll(all_rest_coords,1,0)
        rest_ab_diff_forward = all_rest_coords - rest_ab_coords_forward_rolled 
        
        rest_ab_coords_backward_rolled = torch.roll(all_rest_coords,-1,0)
        rest_ab_diff_backward = rest_ab_coords_backward_rolled - all_rest_coords
        
        
        rest_ab_diff_forward = rest_ab_diff_forward[1:-1]
        rest_ab_diff_backward = rest_ab_diff_backward[1:-1]
        r_norm = torch.norm(rest_ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
        mid_angle = torch.acos(F.cosine_similarity(rest_ab_diff_forward.view(-1,3,3),rest_ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        cross_vector = torch.cross(rest_ab_diff_forward.view(-1,3,3),rest_ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
        normal_angle = torch.acos(F.cosine_similarity(cross_vector,rest_ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
        
        rest_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
        if np.isnan(rest_pos_features.numpy()).any() == True: continue
            
        edge_index_res = knn_graph(rest_pos_features[:,3:6].view(-1,3),k=5,loop=True)
        edge_s = []
        edge_f = []
        order_ab = []

        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
                order_ab.append(1)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        order_ag = []
        
        #Fully connect the Antigen graph
        for i in range(len(ab_label_features),len(ab_label_features)+len(ag_label_features),1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
                order_ag.append(2)
                
        order_final = order_ab + order_ag
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)
        
        Final_input_antigen_features = torch.cat([ag_label_features,antigen_pos_features],dim=1)
       
        Final_input_features = torch.cat([Final_input_anitbody_features,Final_input_antigen_features],dim=0)
        
        amino_index = torch.tensor([i for i in range(len(Final_input_features))]).view(-1,1).float()
        #Final_target_features = 
        
        data = Data(x=Final_input_features, edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), order = torch.tensor(order_final).view(-1,1),y=Final_target_antibody_features,antigen_labels=ag_label_features,antigen_pos=antigen_pos_features, ag_len= torch.tensor(len(antigen_seq)).view(-1,1),ab_len= torch.tensor(len(antibody_seq)-2).view(-1,1),a_index = amino_index.view(1,-1),first_res=first_coord)
        data_rest = Data(x=rest_pos_features,label=ab_rest_label_features,edge_index=edge_index_res)
                    
        final_data.append((data,data_rest))
        
    return final_data

def get_one_hot(seq):
    hot_encoding = []
    for residue in list(seq):
        if str(residue) == '*': continue
        hot_encoder = np.zeros(20)
        res_idx = ALPHABET.index(residue)
        hot_encoder[res_idx] = 1
        hot_encoding.append(hot_encoder)
    
    return torch.tensor(hot_encoding).view(-1,20)

def get_pos_features(all_coords):
    
    ab_coords_forward_rolled = torch.roll(all_coords,1,0)
    ab_diff_forward = all_coords - ab_coords_forward_rolled
        
    ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
    ab_diff_backward = ab_coords_backward_rolled - all_coords
    
    ab_diff_forward = ab_diff_forward[1:-1]
    ab_diff_backward = ab_diff_backward[1:-1]
    r_norm = torch.norm(ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,1)
        
    mid_angle = torch.acos(F.cosine_similarity(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
        
    cross_vector = torch.cross(ab_diff_forward.view(-1,3,3),ab_diff_backward.view(-1,3,3),dim=2).view(-1,3,3)
        
    normal_angle = torch.acos(F.cosine_similarity(cross_vector,ab_diff_backward.view(-1,3,3),dim=2)).view(-1,3,1)
    
    antibody_pos_features = torch.cat((r_norm,mid_angle,normal_angle),dim=2).view(-1,9)
    
    return antibody_pos_features
    
    
def get_graph_data_polar_with_sidechains_angle_whole_protseed(cdr_type,file_path):
    
    Ab_seq,Ab_euc_coord,Ag_seq,Ag_euc_coord,Before_seq,Before_coord,After_seq,After_coord = get_seq_and_coord_whole_protseed(cdr_type,file_path)
    
    final_data = []
    for entry_number in range(len(Ab_seq)):
        #print(entry_number)
        if cdr_type==3 and entry_number in [1402,1447,1841,1892,2131,2261]: continue
        
        #print(entry_number,Ab_seq[entry_number])
        
        ab_hot_encoding = []
        ag_hot_encoding = []
        rest_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]
        
        antigen_seq = Ag_seq[entry_number]
        antigen_euc_coord = Ag_euc_coord[entry_number]
        
        before_seq = Before_seq[entry_number]
        before_coord = Before_coord[entry_number]
        
        after_seq = After_seq[entry_number]
        after_coord = After_coord[entry_number]
        
        # Converting sequence into labels
        
        antigen_len = len(antigen_seq)
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len==1: continue
            
        ab_label_features = get_one_hot(antibody_seq[1:-1])
        ag_label_features = get_one_hot(antigen_seq[1:-1])
        if len(ag_label_features) == 0: continue
        if len(ab_label_features) ==0: continue
        before_label_features = get_one_hot(before_seq[1:-1])
        after_label_features = get_one_hot(after_seq[1:-1])
        
        
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        first_coord = all_coords[0].view(-1,3,3)
        antibody_pos_features = get_pos_features(all_coords)
        

        pos_antigen = np.array(antigen_euc_coord).reshape(len(antigen_euc_coord),3,3)
        all_coords_ag = torch.from_numpy(antigen_euc_coord.reshape(len(antigen_euc_coord),9))
        
        antigen_pos_features = get_pos_features(all_coords_ag)

        if np.isnan(antibody_pos_features.numpy()).any() == True: continue
        
        
        all_before_coords =  torch.from_numpy(before_coord.reshape(len(before_coord),9))
        before_pos_features = get_pos_features(all_before_coords)
        
        all_after_coords =  torch.from_numpy(after_coord.reshape(len(after_coord),9))
        after_pos_features = get_pos_features(all_after_coords)
        
        if np.isnan(before_pos_features.numpy()).any() == True: continue
        if np.isnan(after_pos_features.numpy()).any() == True: continue
            
        edge_index_before = knn_graph(before_pos_features[:,3:6].view(-1,3),k=5,loop=True)
        edge_index_after = knn_graph(after_pos_features[:,3:6].view(-1,3),k=5,loop=True)
        
        edge_s = []
        edge_f = []
        order_ab = []

        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
                order_ab.append(1)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        edge_start = edges_ab[0].view(len(edges_ab[0])).numpy().tolist()
        edge_end = edges_ab[1].view(len(edges_ab[0])).numpy().tolist()
        order_ag = []
        
        #Fully connect the Antigen graph
        for i in range(len(ab_label_features),len(ab_label_features)+len(ag_label_features),1):
            for j in range(antibody_cdr_len):
                edge_start.append(i)
                edge_end.append(j)
                order_ag.append(2)
                
        order_final = order_ab + order_ag
        final_edge_index = torch.tensor([edge_start,edge_end])
        
        
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        
        temp_coords = antibody_pos_features.view(-1,3,3)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)
        
        Final_input_antigen_features = torch.cat([ag_label_features,antigen_pos_features],dim=1)
       
        Final_input_features = torch.cat([Final_input_anitbody_features,Final_input_antigen_features],dim=0)
        
        amino_index = torch.tensor([i for i in range(len(Final_input_features))]).view(-1,1).float()
        #Final_target_features = 
        
        data = Data(x=Final_input_features, edge_index=final_edge_index,edge_ab = edges_ab.view(-1,2), order = torch.tensor(order_final).view(-1,1),y=Final_target_antibody_features,antigen_labels=ag_label_features,antigen_pos=antigen_pos_features, ag_len= torch.tensor(len(antigen_seq)-2).view(-1,1),ab_len= torch.tensor(len(antibody_seq)-2).view(-1,1),a_index = amino_index.view(1,-1),first_res=first_coord)
        data_before = Data(x=before_pos_features,label=before_label_features,edge_index=edge_index_before)
        data_after = Data(x=after_pos_features,label=after_label_features,edge_index=edge_index_after)
                    
        final_data.append((data,data_before,data_after))
        
    return final_data


def get_graph_data_polar_uncond_with_side_chains_angle_v2(cdr_type,file_path):
    
    Ab_seq,Ab_ang_coord,Ab_euc_coord,Pdb = get_seq_and_coord_uncond(cdr_type,file_path)
    #print(Ab_seq)
    final_data = []
    for entry_number in range(len(Ab_seq)):
        
        #print(entry_number,Ab_seq[entry_number])
        pdb_ab = Pdb[entry_number]
        
        ab_hot_encoding = []
        
        antibody_seq = Ab_seq[entry_number]
        antibody_ang_coord = Ab_ang_coord[entry_number]
        antibody_euc_coord = Ab_euc_coord[entry_number]

        # Converting sequence into labels
        antibody_cdr_len = len(antibody_seq)-2
        if antibody_cdr_len==1: continue
        for residue in list(antibody_seq[1:-1]):
            hot_encoder = np.zeros(20)
            res_idx = ALPHABET.index(residue)
            hot_encoder[res_idx] = 1
            ab_hot_encoding.append(hot_encoder)
        

        ab_label_features = torch.tensor(ab_hot_encoding).view(antibody_cdr_len,20)
        
        all_coords = torch.from_numpy(antibody_euc_coord.reshape(len(antibody_seq),9))
        first_coord = all_coords[0].view(-1,3,3)
        ab_coords_backward_rolled = torch.roll(all_coords,-1,0)
        ab_diff_backward = ab_coords_backward_rolled - all_coords
        ab_diff_backward = ab_diff_backward[1:-1].numpy().reshape(-1,3)
        
        
        r_ab,t_ab,z_ab = cartesian_to_spherical(ab_diff_backward[:,0].reshape(-1,1),ab_diff_backward[:,1].reshape(-1,1),ab_diff_backward[:,2].reshape(-1,1))

        antibody_pos_features = torch.cat([torch.tensor(r_ab).view(-1,1),torch.tensor(t_ab).view(-1,1),torch.tensor(z_ab).view(-1,1)],dim=1).view(-1,9)
        
        if np.isnan(antibody_pos_features.numpy()).any() == True: continue

        edge_s = []
        edge_f = []
        #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        for idx_start in range(len(antibody_pos_features)):
            for idx_end in range(len(antibody_pos_features)):
                edge_s.append(idx_start)
                edge_f.append(idx_end)
        
        edges_ab = torch.tensor([edge_s,edge_f])
        
        Final_target_antibody_features = torch.cat([ab_label_features,antibody_pos_features],dim=1)
        
        Input_ab_labels = torch.tensor(float(1/20)*np.ones((antibody_cdr_len,20))).view(-1,20)
        amino_index = torch.tensor([i for i in range(antibody_cdr_len)]).view(-1,1).float()
        temp_coords = antibody_pos_features.view(-1,3,3)
        #print(temp_coords)
        Input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(),temp_coords[-1].numpy(),antibody_cdr_len)).view(-1,9)
        Final_input_anitbody_features = torch.cat([Input_ab_labels,Input_ab_coords],dim=1)

        data = Data(x=Final_input_anitbody_features, edge_index=edges_ab,y=Final_target_antibody_features,a_index = amino_index.view(1,-1),first_res=first_coord)
        #print(data)
        final_data.append(data)
        
    return final_data



    
    

if __name__ == "__main__":
    #Test RMSD calculation
    io = PDBIO()
    p = PDBParser()
