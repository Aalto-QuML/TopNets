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

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',choices=label2idx.keys())
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')

parser.add_argument('--wandb', type=str, default="disabled")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

wandb.init(mode=args.wandb,project="TopNNs")
config = wandb.config        
config.batch_size = args.batch_size    
config.epochs = args.epochs            
config.lr = args.lr   
config.property = args.property 

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

model = TopNN(hidden_dim=128,num_node_features=dataset[0].x.shape[1],num_filtrations=8,filtration_hidden=16,out_ph_dim=64,n_steps=10,solver='adaptive_heun').to(device)

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
        torch.save(checkpoint, str(cwd)+"/Models/"+"TopNNs_v2_"+args.property+"_"+str(epoch) + ".pth")
        #torch.save(model,str(cwd)+"/Models/"+"TopNNs_"+args.property+"_"+str(epoch) + ".pt")

    #test_mae = 0
    #for batch in Dloader['test']:
    #    props = batch.y[:,prop_idx].to(device)
    #    pred = model(batch.to(device))
    #    loss = loss_l1(meann + mad*pred.squeeze(dim=1), props)
    #    test_mae = test_mae + loss.item()

    #print("Test MAE:", float(test_mae/len(Dloader['test'])))









'''
def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}

    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        #batch_size, n_nodes, _ = data['positions'].size()
        #atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
        #atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        #edge_mask = data['edge_mask'].to(device, dtype)
        #one_hot = data['one_hot'].to(device, dtype)
        #charges = data['charges'].to(device, dtype)
        #nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        #nodes = nodes.view(batch_size, n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        #edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        breakpoint()
        label = data[args.property].to(device, dtype)

        breakpoint()
        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        if partition == 'train':
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + meann, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    for epoch in range(0, args.epochs):
        train(epoch, Dloader['train'], partition='train')
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, Dloader['valid'], partition='valid')
            test_loss = train(epoch, Dloader['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)
'''
