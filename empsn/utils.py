import torch.nn as nn
import random
import numpy as np
import os
import torch
from torch import Tensor
from argparse import Namespace
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from typing import Tuple
from gudhi.dtm_rips_complex import DTMRipsComplex
from torch_cluster import knn_graph
import random
random.seed(42)
from simplicial_data.simplicial_data import SimplicialTransform


pwd = os.getcwd()

DATA_PATHS = {'CAD': str(pwd) + '/data_pointnet/ModelNetNoisy01_C=10,N=100,T=1,K=2000_data'}
LABEL_PATHS = {'CAD': str(pwd) + '/data_pointnet/ModelNetNoisy01_C=10,N=100,T=1,K=2000_label'}

def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == 'qm9':
        num_input = 15
        num_out = 1
    else:
        raise ValueError(f'Do not recognize dataset {args.dataset}.')

    if args.model_name == 'egnn':
        from models.egnn import EGNN
        model = EGNN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers
        )
    elif args.model_name == 'empsn':
        from models.empsn import EMPSN, EMPSN_Rephine
        model = EMPSN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
            max_com=args.max_com
        )

    elif args.model_name == 'empsn_rephine':
        from models.empsn import EMPSN, EMPSN_Rephine
        model = EMPSN_Rephine(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
            max_com=args.max_com
        )

    elif args.model_name == 'empsn_rephine_cont':
        from models.empsn import EMPSN, EMPSN_Rephine_Cont
        model = EMPSN_Rephine_Cont(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=1,
            max_com=args.max_com,
            solver = 'adaptive_heun',
            nsteps = 5
        )

    else:
        raise ValueError(f'Model type {args.model_name} not recognized.')

    print(model)
    return model


def pointcloud_normalize(X):
    pc_list = []
    for i in range(X.shape[0]):
        avg = torch.mean(X[i, :, :], dim=0)
        std = torch.std(X[i, :, :], dim=0)
        pointcloud = (X[i, :, :] - avg.reshape(1, X.shape[2])) / std.reshape(1, X.shape[2])
        pc_list.append(pointcloud)
    data = torch.stack(pc_list, axis=0)
    
    return data

def get_dtm_points(all_X,device):
    dtm_points_list = []
    dist = torch.cdist(all_X, all_X)
    for i in range(all_X.shape[0]):
            filt = DTMRipsComplex(distance_matrix=dist[i, :, :].cpu().detach().numpy(), k=1)
            simplex_tree = filt.create_simplex_tree(max_dimension=2)
            barcode = simplex_tree.persistence()
            points = []
            for x in barcode:
                if x[0] == 1:
                    points.append((torch.tensor(x[1][0]).to(device,torch.float32), torch.tensor(x[1][1] - x[1][0]).to(device,torch.float32)))
            dtm_points_list.append(points)

    return dtm_points_list


def load_train_test_data(batch_size):

    data_path = DATA_PATHS['CAD']
    label_path = LABEL_PATHS['CAD']

    data = torch.load(data_path)
    label = torch.load(label_path)

    all_X = pointcloud_normalize(data[:, :128, :])
    data_num = all_X.shape[0]
    data_idx = list(range(data_num))
    # random.shuffle(data_idx,)
    data_idx.sort(key=lambda i: (label[i], i))

    train_idx = [data_idx[i] for i in range(data_num) if i % 5 != 1]
    valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == 1]

    final_data_train = []
    final_data_test = []

    for i in train_idx:
        data = torch.load(str(pwd) + '/data/train_'+str(i)+'.pt')
        final_data_train.append(data)


    for i in valid_idx:
        data = torch.load(str(pwd) + '/data/test_'+str(i)+'.pt')
        final_data_test.append(data)


    follow = [f"x_{i}" for i in range(2+1)] + ['x']
    train_loader = DataLoader(final_data_train,batch_size=batch_size,shuffle=True,follow_batch=follow)
    #val_loader = DataLoader(dataset[idx_val],batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(final_data_test,batch_size=batch_size,shuffle=True,follow_batch=follow)
    dataloader = {"train": train_loader, "test":test_loader}
    return dataloader,63



def train_test_split_pointdata(data_type,batch_size):
    data_path = DATA_PATHS[data_type]
    label_path = LABEL_PATHS[data_type]

    data = torch.load(data_path)
    label = torch.load(label_path)

    all_X = pointcloud_normalize(data[:, :128, :])
    data_num = all_X.shape[0]
    data_idx = list(range(data_num))
    # random.shuffle(data_idx,)
    data_idx.sort(key=lambda i: (label[i], i))

    train_idx = [data_idx[i] for i in range(data_num) if i % 5 != 1]
    valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == 1]

    final_data_train = []
    final_data_test = []

    max_deg = 0
    num_neigh = 2

    transform = SimplicialTransform(dim=2, dis=2.0)
    for idx,i in enumerate(train_idx):
        print(idx)
        edges = knn_graph(all_X[i],num_neigh,loop=False)
        node_features = torch.zeros((all_X[i].shape[0],64))
        edge_weights = []
        for edge in edges:
            sq_dist = (all_X[i][edge[0]] -  all_X[i][edge[1]])**2
            weight = torch.exp(- sq_dist.sum()/2*2)
            edge_weights.append(weight)

        data = Data(x=node_features,pos=all_X[i], edge_index=edges,y=label[i],edge_attr=torch.tensor(edge_weights).view(-1,1))
        data_trans = transform(data)
        torch.save(data_trans,str(pwd)+ '/data/train_'+str(i)+'.pt')
        final_data_train.append(data_trans)


    for i in valid_idx:
        edges = knn_graph(all_X[i],num_neigh,loop=False)
        node_features = torch.zeros((all_X[i].shape[0],64))
        edge_weights = []
        for edge in edges:
            sq_dist = (all_X[i][edge[0]] -  all_X[i][edge[1]])**2
            weight = torch.exp(- sq_dist.sum()/2*2)
            edge_weights.append(weight)

        data = Data(x=node_features,pos=all_X[i], edge_index=edges,y=label[i],edge_attr=torch.tensor(edge_weights).view(-1,1))
        data_trans = transform(data)
        torch.save(data_trans,str(pwd)+ '/data/test_'+str(i)+'.pt')
        final_data_test.append(data_trans)

    follow = [f"x_{i}" for i in range(2+1)] + ['x']
    train_loader = DataLoader(final_data_train,batch_size=batch_size,shuffle=True,follow_batch=follow)
    #val_loader = DataLoader(dataset[idx_val],batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(final_data_test,batch_size=batch_size,shuffle=True,follow_batch=follow)
    breakpoint()
    dataloader = {"train": train_loader, "test":test_loader}
    return dataloader,63

def train_test_split_pointdata_dtm(data_type,batch_size,device):
    data_path = DATA_PATHS[data_type]
    label_path = LABEL_PATHS[data_type]

    data = torch.load(data_path)
    label = torch.load(label_path)

    all_X = pointcloud_normalize(data[:, :128, :])
    dtm_points_list = get_dtm_points(all_X,device)

    data_num = all_X.shape[0]
    data_idx = list(range(data_num))
    # random.shuffle(data_idx,)
    data_idx.sort(key=lambda i: (label[i], i))

    train_idx = [data_idx[i] for i in range(data_num) if i % 5 != 1]
    valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == 1]

    random.shuffle(train_idx)
    random.shuffle(valid_idx)

    final_data_train = []
    final_data_test = []

    final_train_dtm = []
    final_test_dtm = []

    num_neigh = 40

    transform = SimplicialTransform(dim=2, dis=2.0)
    for i in train_idx:
        edges = knn_graph(all_X[i],num_neigh,loop=False)
        node_features = torch.zeros((all_X[i].shape[0],64))
        edge_weights = []
        for edge in range(edges.shape[1]):
            sq_dist = (all_X[i][edges[0][edge]] -  all_X[i][edges[1][edge]])**2
            weight = torch.exp(- sq_dist.sum()/2*2)
            edge_weights.append(torch.tensor([weight,torch.sqrt(sq_dist.sum())]))

        #node_features = torch.cdist(all_X[i],all_X[i])
        #node_features = torch.from_numpy(distance.cdist(all_X[i].numpy(),all_X[i].numpy(),'euclidean'))
        #degrees = degree(edges[0],num_nodes=all_X[i].shape[0]).view(-1,1)
        #node_features = torch.cat([degrees,node_features],dim=1)

        data = Data(x=node_features,pos=all_X[i], edge_index=edges,y=label[i],edge_attr=torch.stack(edge_weights))
        #transform = OneHotDegree(max_degree=max_deg,cat=True)  # Assuming a maximum degree of 3 for simplicity
        #data = transform(data)
        final_train_dtm.append(dtm_points_list[i])     
        final_data_train.append(transform(data))


    for i in valid_idx:
        edges = knn_graph(all_X[i],num_neigh,loop=False)
        #node_features = torch.from_numpy(distance.cdist(all_X[i].numpy(),all_X[i].numpy(),'euclidean'))
        node_features = torch.zeros((all_X[i].shape[0],64))
        edge_weights = []
        for edge in range(edges.shape[1]):
            sq_dist = (all_X[i][edges[0][edge]] -  all_X[i][edges[1][edge]])**2
            weight = torch.exp(- sq_dist.sum()/2*2)
            edge_weights.append(torch.tensor([weight,torch.sqrt(sq_dist.sum())]))

        #node_features = torch.cdist(all_X[i],all_X[i])
        #node_features = torch.from_numpy(distance.cdist(all_X[i].numpy(),all_X[i].numpy(),'euclidean'))
        #degrees = degree(edges[0],num_nodes=all_X[i].shape[0]).view(-1,1)
        #node_features = torch.cat([degrees,node_features],dim=1)
        data = Data(x=node_features,pos=all_X[i], edge_index=edges,y=label[i],edge_attr=torch.stack(edge_weights))
        #transform = OneHotDegree(max_degree=max_deg,cat=True)  # Assuming a maximum degree of 3 for simplicity
        #data = transform(data)
        final_test_dtm.append(dtm_points_list[i])   
        final_data_test.append(transform(data))


    train_loader = DataLoader(final_data_train,batch_size=batch_size,shuffle=False)
    #val_loader = DataLoader(dataset[idx_val],batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(final_data_test,batch_size=batch_size,shuffle=False)
    dataloader = {"train": train_loader, "test":test_loader}
    dtm_dataloader = {"train": final_train_dtm, "test":final_test_dtm}
    return dataloader,63,dtm_dataloader


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == 'qm9':
        from qm9.utils import generate_loaders_qm9
        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized.')

    return train_loader, val_loader, test_loader


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


if __name__ == "__main__":
    train_test_split_pointdata('CAD',32)