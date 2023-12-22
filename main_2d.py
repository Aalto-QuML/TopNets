import torch
from torch import nn, optim
import argparse
import utils
import json
from torch_geometric.datasets import QM9
from torch_geometric.nn.models.schnet import qm9_target_dict
from models.topo_gnn import TopNN,TopNN_2D
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os
from datasets.datasets import get_data
from torch_geometric.data import DataLoader

torch.set_printoptions(precision=2)
utils.utils.set_seed(42)
cwd = os.getcwd()

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_filtrations', type=int, default=8, metavar='nf',
                    help='Number of filtration functions')
parser.add_argument('--nsteps', type=int, default=20, metavar='nf',
                    help='Steps for the ODE solver')
parser.add_argument('--out_ph', type=int, default=64, metavar='nf',
                    help='Out PH embedding dim')
parser.add_argument('--fil_hid', type=int, default=16, metavar='nf',
                    help='Filtration hidden dim')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument("--dataset",type=str,default="NCI109",choices=["MUTAG","ogbg-molhiv","ZINC","DD","PROTEINS_full","PROTEINS","NCI109","NCI1","IMDB-BINARY",],)
parser.add_argument('--weight_decay', type=float, default=1e-8, metavar='N',
                    help='weight decay')
                    
parser.add_argument('--wandb', type=str, default="disabled")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

wandb.init(mode=args.wandb,project="TopNNs")
config = wandb.config        
config.batch_size = args.batch_size    
config.epochs = args.epochs            
config.lr = args.lr   
config.dataset = args.dataset
config.num_fil = args.num_filtrations
config.nsteps = args.nsteps
config.out_ph_dim = args.out_ph
config.fil_hid_dim = args.fil_hid
config.hidden_dim = 128
config.solver = 'adaptive_heun'
#utils.makedir(args.outf)
#utils.makedir(args.outf + "/" + args.exp_name)
train_data, val_data, test_data, stats = get_data(args.dataset, False)

num_node_features = stats["num_features"]
num_classes = stats["num_classes"]

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
Dloader = {"train": train_loader,"valid": val_loader, "test":test_loader}

model = TopNN_2D(hidden_dim=128,depth=1,gnn='gin',num_node_features=num_node_features,num_classes=num_classes,num_filtrations=args.num_filtrations,filtration_hidden=args.fil_hid,out_ph_dim=args.out_ph,n_steps=args.nsteps,solver='adaptive_heun').to(device)


print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_fn = torch.nn.CrossEntropyLoss()
best_loss = float('inf')

for epoch in range(args.epochs):
    total_train_loss = 0
    total_val_loss = 0
    model.train()
    for batch in Dloader['train']:
        optimizer.zero_grad()
        pred = model(batch.to(device))
        loss = loss_fn(pred.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()
    
    model.eval()
    for batch in Dloader['valid']:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        total_val_loss = total_val_loss + loss.item()
        val_acc = -loss
        if not isinstance(loss_fn, torch.nn.L1Loss):
            val_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()


    scheduler.step()

    test_acc = 0
    for batch in Dloader['test']:
        batch = batch.to(device)
        out = model(batch)
        if not isinstance(loss_fn, torch.nn.L1Loss):
            test_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()


    print(
                f"{epoch:3d}: Train Loss: {float(total_train_loss/len(Dloader['train'])):.3f},"
                f" Val Loss: {float(total_val_loss/len(Dloader['valid'])):.3f}, Val Acc: {val_acc.item():.3f}, "
                f"Test Acc: {test_acc.item():.3f}"
            )


    wandb.log({
        "Train_loss": float(total_train_loss/len(Dloader['train'])),
        "Val_loss": float(total_val_loss/len(Dloader['valid'])),
        "Val_acc": val_acc,
        "Test_acc":test_acc })

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, str(cwd)+"/Models/"+"TopNNs_2d_"+str(args.nsteps)+"_"+args.dataset+"_"+str(epoch) + ".pth")





