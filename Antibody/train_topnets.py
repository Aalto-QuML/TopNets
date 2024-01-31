import warnings
import os
from model_function import *
from utils import *
from torch_geometric.loader import DataLoader
import torch.nn.functional as Fin
import timeit
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
import matplotlib 
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import time
import torch
import torch.optim as optim
from rmsd import *

set_seed(42)
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('TopNets')

parser.add_argument('--solver', type=str, default="adaptive_heun", choices=SOLVERS)
parser.add_argument('--atol', type=float, default=5e-1)
parser.add_argument('--rtol', type=float, default=5e-1)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-9)
parser.add_argument('--cdr', type=int, default=1)
args = parser.parse_args()

cwd = os.getcwd() 
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

test_path = str(cwd) + "/data/Cond_cdr_cluster/cdrh3/test.json"
train_path = str(cwd) +  "/data/Cond_cdr_cluster/cdrh3/train.json"
val_path = str(cwd) + "/data/Cond_cdr_cluster/cdrh3/val.json"

model = TopNets(29, 29,7)

t_begin=0.
t_end=1
t_nsamples=200
t_space = np.linspace(t_begin, t_end, t_nsamples)

print("############################ Data is loading ###########################")
Train_data = get_graph_data_polar_with_sidechains_angle_impsn(args.cdr,train_path)
Test_data = get_graph_data_polar_with_sidechains_angle_impsn(args.cdr,test_path)
Val_data = get_graph_data_polar_with_sidechains_angle_impsn(args.cdr,val_path)

#Train_data = Train_data
Train_loader = DataLoader(Train_data, batch_size=args.batch_size, shuffle=True,num_workers=0)
Test_loader = DataLoader(Test_data, batch_size=1, shuffle=False,num_workers=0)
Val_loader = DataLoader(Val_data, batch_size=args.batch_size, shuffle=False,num_workers=0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.niters)
best_loss = float('inf')
print("############################ Data is loaded, Model starts to train ###########################")
print(model)
PPL_final = []
RMSD_final = []
for epoch in range(args.niters):
    total_train_loss =0
    total_val_loss =0 
    RMSD_test_n = []
    RMSD_test_ca = []
    RMSD_test_ca_cart = []
    RMSD_test_c = []
    ACC = []
    model.train()
    for batch in Train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        y_pd = model(batch)
        antigen_len = []
        for entry in batch.ag_len.numpy().tolist():
            antigen_len.append(entry[0])
        
        antibody_len = []
        for entry in batch.ab_len.numpy().tolist():
            antibody_len.append(entry[0])
        
        y_gt = batch.y.to(device)
        final_pred = get_antibody_entries(y_pd.to(device),batch.batch,antibody_len,antigen_len)
        loss = loss_function_vm_with_side_chains_angle(final_pred,y_gt.to(device),args.batch_size)  
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()

    
    lr_scheduler.step()

    model.eval()
    for idx,batch in enumerate(Val_loader):
        batch = batch.to(device)
        y_pd = model(batch)
        antigen_len = []
        for entry in batch.ag_len.numpy().tolist():
            antigen_len.append(entry[0])
        
        antibody_len = []
        for entry in batch.ab_len.numpy().tolist():
            antibody_len.append(entry[0])
        
        y_gt = batch.y.to(device)
        final_pred = get_antibody_entries(y_pd.to(device),batch.batch,antibody_len,antigen_len)
        loss = loss_function_vm_with_side_chains_angle(final_pred,y_gt.to(device),args.batch_size)
        total_val_loss = total_val_loss + loss.item()

    total_train_loss = float(total_train_loss/len(Train_loader))
    total_val_loss = float(total_val_loss/len(Val_loader))

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, str(cwd)+"/Models/TopNNs_rabd_"+str(epoch) + ".pth")
    
    for batch in Test_loader:
        batch = batch.to(device)
        y_pd = model(batch)
        y_gt = batch.y.to(device)
        antigen_len = []
        for entry in batch.ag_len.numpy().tolist():
            antigen_len.append(entry[0])
        
        antibody_len = []
        for entry in batch.ab_len.numpy().tolist():
            antibody_len.append(entry[0])
        
        
        final_pred = get_antibody_entries(y_pd.to(device),batch.batch,antibody_len,antigen_len)
        rmsd_n,rmsd_ca,rmsd_c,acc,rmsd_cart_ca = evaluate_rmsd_with_sidechains_cond_angle(final_pred,y_gt,batch.first_res)

        RMSD_test_n.append(rmsd_n)
        RMSD_test_ca.append(rmsd_ca)
        RMSD_test_ca_cart.append(rmsd_cart_ca)
        RMSD_test_c.append(rmsd_c)
        ACC.append(acc)

    RMSD_test_arr_n = np.array(RMSD_test_n).reshape(-1,1) 
    RMSD_test_arr_ca = np.array(RMSD_test_ca).reshape(-1,1)
    RMSD_test_arr_ca_cart = np.array(RMSD_test_ca_cart).reshape(-1,1)
    RMSD_test_arr_c = np.array(RMSD_test_c).reshape(-1,1)
    ACC_arr = np.array(ACC).reshape(-1,1)

    print(
                f"{epoch:3d}: Train Loss: {float(total_train_loss):.3f},"
                f" Val Loss: {float(total_val_loss):.3f}, "
                f"Test Acc: {np.mean(ACC_arr,axis=0)[0]:.3f}, RMSD Ca: {np.mean(RMSD_test_arr_ca_cart,axis=0)[0]:.3f}"
            )
    


    
        
    


