import torch
from torch import nn, optim
import argparse
from utils.utils import set_seed
import json
from models.topo_gnn import TopNN_TOGL
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datasets.datasets import get_data
from torch_geometric.data import DataLoader
from ogb.graphproppred import Evaluator
import topognn.data_utils as topo_data
from topognn.cli_utils import str2bool
import sys


torch.set_printoptions(precision=2)
set_seed(42)
cwd = os.getcwd()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dataset', type=str, choices=topo_data.dataset_map_dict().keys())
parser.add_argument('--training_seed', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument("--paired", type = str2bool, default=False)
parser.add_argument("--merged", type = str2bool, default=False)
parser.add_argument("--gnn",type=str,default="gcn",choices=["gcn", "gin"],)  
partial_args, _ = parser.parse_known_args()
#model_cls = MODEL_MAP[partial_args.model]
    #dataset_cls = DATASET_MAP[partial_args.dataset]
dataset_cls = topo_data.get_dataset_class(**vars(partial_args))
#parser = model_cls.add_model_specific_args(parser)
parser = dataset_cls.add_dataset_specific_args(parser)
args = parser.parse_args()
dataset = dataset_cls(**vars(args))
dataset.prepare_data()

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.epochs = 300
args.num_filtrations = 8
args.nsteps = 20
args.out_ph = 64
args.fil_hid = 16
args.lr = 1e-3
dtype = torch.float32
print(args)
Dloader = {"train": dataset.train_dataloader(),"valid": dataset.val_dataloader(), "test":dataset.test_dataloader()}

model = TopNN_TOGL(hidden_dim=128,depth=1,gnn=args.gnn,num_node_features=dataset.node_attributes,num_classes=dataset.num_classes,num_filtrations=args.num_filtrations,filtration_hidden=args.fil_hid,out_ph_dim=args.out_ph,n_steps=args.nsteps,solver='adaptive_heun').to(device)

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
    for batch in Dloader['train']:
        optimizer.zero_grad()
        labels = torch.nn.functional.one_hot(batch.y, num_classes=dataset.num_classes).to(device)
        pred = model(batch.to(device))
        loss = loss_fn(pred.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()
    
    model.eval()
    for batch in Dloader['valid']:
        batch = batch.to(device)
        out = model(batch)
        labels = torch.nn.functional.one_hot(batch.y, num_classes=dataset.num_classes).to(device)
        loss = loss_fn(out.squeeze(), labels.float())
        total_val_loss = total_val_loss + loss.item()
        val_acc = -loss

        if evaluator is None:
            if not isinstance(loss_fn, torch.nn.L1Loss):
                val_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()
        else:
            val_acc =  evaluator.eval({'y_pred': out[:, 1].unsqueeze(dim=1), 'y_true': batch.y})[evaluator.eval_metric]


    scheduler.step()

    test_acc = 0
    for batch in Dloader['test']:
        batch = batch.to(device)
        out = model(batch)
        if evaluator is None:
            if not isinstance(loss_fn, torch.nn.L1Loss):
                test_acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()
        else:
            test_acc = acc = evaluator.eval({'y_pred': out[:, 1].unsqueeze(dim=1), 'y_true': batch.y})[evaluator.eval_metric]


    print(
                f"{epoch:3d}: Train Loss: {float(total_train_loss/len(Dloader['train'])):.3f},"
                f" Val Loss: {float(total_val_loss/len(Dloader['valid'])):.3f}, Val Acc: {val_acc.item():.3f}, "
                f"Test Acc: {test_acc.item():.3f}"
            )


    if total_val_loss < best_loss:
        best_loss = total_val_loss
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, str(cwd)+"/Models/"+"TopNets_togl_"+str(args.nsteps)+"_"+args.dataset+"_"+str(epoch) + ".pth")
