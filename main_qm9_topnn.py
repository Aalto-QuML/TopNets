import torch
from torch import nn, optim
import argparse
import utils
import json
from torch_geometric.datasets import QM9
from torch_geometric.nn.models.schnet import qm9_target_dict
from models.topo_gnn import TopNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os

utils.utils.set_seed(42)
cwd = os.getcwd()
label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_filtrations', type=int, default=8, metavar='nf',
                    help='Number of filtration functions')
parser.add_argument('--nsteps', type=int, default=10, metavar='nf',
                    help='Steps for the ODE solver')
parser.add_argument('--out_ph', type=int, default=64, metavar='nf',
                    help='Out PH embedding dim')
parser.add_argument('--fil_hid', type=int, default=16, metavar='nf',
                    help='Filtration hidden dim')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--property', type=str, default='homo', metavar='N',choices=label2idx.keys())
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
                    

parser.add_argument('--wandb', type=str, default="disabled")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

wandb.init(mode=args.wandb,project="TopNNs")

#utils.makedir(args.outf)
#utils.makedir(args.outf + "/" + args.exp_name)


prop_idx = label2idx[args.property]
print("Training model for ", args.property)
path = './data/QM9'
dataset = QM9(path)
#dataset.process()
Dloader,meann,mad = utils.utils.train_test_split(dataset,0.8,0.1,args.batch_size,prop_idx)
#dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
# compute mean and mean absolute deviation
#meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)

model = TopNN(hidden_dim=128,num_node_features=dataset[0].x.shape[1],num_filtrations=args.num_filtrations,filtration_hidden=args.fil_hid,out_ph_dim=args.out_ph,n_steps=args.nsteps,solver='adaptive_heun').to(device)

config = wandb.config        
config.batch_size = args.batch_size    
config.epochs = args.epochs            
config.lr = args.lr   
config.property = args.property 
config.num_fil = args.num_filtrations
config.nsteps = args.nsteps
config.out_ph_dim = args.out_ph
config.fil_hid_dim = args.fil_hid


#model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers, coords_weight=1.0,attention=args.attention, node_attr=args.node_attr)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss(reduction='mean')
best_loss = float('inf')

for epoch in range(args.epochs):
    total_train_loss = 0
    total_val_loss = 0
    model.train()
    for batch in Dloader['train']:
        optimizer.zero_grad()
        props = batch.y[:,prop_idx].to(device)
        pred = model(batch.to(device))
        loss = loss_l1(pred.squeeze(dim=1), (props - meann) / mad)
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()
        print("Loss for batch is ",loss.item())

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()
    print("|Iter ",epoch," | Total Train Loss ", total_train_loss,"|")

    model.eval()
    for batch in Dloader['valid']:
        props = batch.y[:,prop_idx].to(device)
        pred = model(batch.to(device))
        loss = loss_l1(pred.squeeze(dim=1), (props - meann) / mad)
        total_val_loss = total_val_loss + loss.item()
        print("Val Loss for batch is ",loss.item())


    print("|Iter ",epoch," | Total Val Loss ", total_val_loss,"|")
    wandb.log({
        "Train_loss": float(total_train_loss/len(Dloader['train'])),
        "Val_loss": float(total_val_loss/len(Dloader['valid'])),
        "lr": lr_val})

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, str(cwd)+"/Models/"+"TopNNs_vanilla_"+args.property+"_"+str(epoch) + ".pth")
        #torch.save(model,str(cwd)+"/Models/"+"TopNNs_"+args.property+"_"+str(epoch) + ".pt")

    #test_mae = 0
    #for batch in Dloader['test']:
    #    props = batch.y[:,prop_idx].to(device)
    #    pred = model(batch.to(device))
    #    loss = loss_l1(meann + mad*pred.squeeze(dim=1), props)
    #    test_mae = test_mae + loss.item()

    #print("Test MAE:", float(test_mae/len(Dloader['test'])))

