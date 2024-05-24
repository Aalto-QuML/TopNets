import torch
from torch import nn, optim
import argparse
import utils
import json
from models.topo_gnn import TopNN,TopNN_2D,TopoGNN_fixed_PH
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datasets.datasets import get_data
from torch_geometric.data import DataLoader
from ogb.graphproppred import Evaluator
from utils.utils import set_seed, get_cin_tudata
from models.models_cin import CIN0_PH,SparseCIN_PH
import time


torch.set_printoptions(precision=2)
set_seed(42)
cwd = os.getcwd()

parser = argparse.ArgumentParser(description='TopNets')
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
parser.add_argument('--cont', type=bool, default=True, metavar='N',
                    help='Continuous Type or Not')
parser.add_argument("--dataset",type=str,default="NCI1",choices=["PROTEINS_full","NCI109","NCI1","IMDB-BINARY",],)
parser.add_argument('--weight_decay', type=float, default=1e-8, metavar='N',
                    help='weight decay')
parser.add_argument("--diagram_type",type=str,default="rephine",choices=["rephine", "standard", "none"],) 
parser.add_argument("--gnn",type=str,default="gcn",choices=["gcn", "gin"],)              

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

train_data, val_data, test_data, stats = get_data(args.dataset, False)

num_node_features = stats["num_features"]
num_classes = stats["num_classes"]


train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)


model = TopNN_2D(hidden_dim=128,depth=1,gnn=args.gnn,num_node_features=num_node_features,num_classes=num_classes,num_filtrations=args.num_filtrations,filtration_hidden=args.fil_hid,out_ph_dim=args.out_ph,n_steps=args.nsteps,solver='adaptive_heun',diagram_type=args.diagram_type).to(device)

evaluator = None
if args.dataset == "ogbg-molhiv": evaluator = Evaluator(args.dataset)

print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_fn = torch.nn.CrossEntropyLoss()
best_loss = float('inf')


for epoch in range(args.epochs):
    total_train_loss = 0
    total_val_loss = 0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch.to(device))
        if args.dataset == "ogbg-molhiv":
            labels = torch.nn.functional.one_hot(batch.y.squeeze(), num_classes=num_classes).to(device)
        else:
            labels = batch.y

        loss = loss_fn(pred.squeeze(), labels)

        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()

    model.eval()
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch)
        if args.dataset == "ogbg-molhiv":
            labels = torch.nn.functional.one_hot(batch.y.squeeze(), num_classes=num_classes).to(device)
        else:
            labels = batch.y
        loss = loss_fn(out.squeeze(), labels)
        total_val_loss = total_val_loss + loss.item()
        val_acc = -loss

        if evaluator is None:
            if not isinstance(loss_fn, torch.nn.L1Loss):
                val_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()
        else:
            val_acc =  evaluator.eval({'y_pred': out.argmax(dim=-1).unsqueeze(dim=-1), 'y_true': batch.y})[evaluator.eval_metric]


    scheduler.step()

    test_acc = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        if evaluator is None:
            if not isinstance(loss_fn, torch.nn.L1Loss):
                test_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()
        else:
            test_acc = evaluator.eval({'y_pred': out.argmax(dim=-1).unsqueeze(dim=-1), 'y_true': batch.y})[evaluator.eval_metric]


    print(
                f"{epoch:3d}: Train Loss: {float(total_train_loss/len(train_loader)):.3f},"
                f" Val Loss: {float(total_val_loss/len(val_loader)):.3f}, Val Acc: {val_acc.item():.3f}, "
                f"Test Acc: {test_acc.item():.3f}"
            )


    if total_val_loss < best_loss:
        best_loss = total_val_loss
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, str(cwd)+"/Models/"+"TopNets_"+str(args.nsteps)+"_"+args.dataset+"_"+str(epoch) + ".pth")
