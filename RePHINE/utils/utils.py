import torch
from torch_scatter import scatter
import os
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataloader
import pandas as pd
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from joblib import delayed
from sklearn.model_selection import StratifiedShuffleSplit
from torch_cluster import knn_graph
import os.path as osp
import pickle
import sys
from ogb.graphproppred import PygGraphPropPredDataset
sys.path.append('../')
import gudhi as gd
import itertools
from RePHINE.datasets.datasets import get_tudataset
from RePHINE.utils.parallel import ProgressParallel
from RePHINE.utils.dataset import InMemoryComplexDataset
from RePHINE.utils.data_loading import DataLoader
import tqdm


#import deepchem as dc
#from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer


ATOM_REF_ENRGY = {1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}
EV_TO_KCAL_MOL = 23.06052





def remove_duplicate_edges(batch):
    with torch.no_grad():
        batch = batch.clone().detach()

        device = batch.x.device
        # Computing the equivalent of batch over edges.
        #            edge_slices = torch.tensor(batch._slice_dict["edge_index"], device=device)
        edge_slices = batch._slice_dict["edge_index"].clone().detach()
        edge_slices = edge_slices.to(device)

        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(
            torch.arange(n_batch, device=device), edge_diff_slices
        )

        correct_idx = batch.edge_index[0] <= batch.edge_index[1]
        # batch_e_idx = batch_e[correct_idx]

        n_edges = scatter(correct_idx.long(), batch_e, reduce="sum")

        #           batch.edge_index = batch.edge_index[:,correct_idx]

        new_slices = torch.cumsum(
            torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_edges)), 0
        )

        vertex_slice = batch._slice_dict["x"].clone()
        #           batch._slice_dict['edge_index'] = new_slices
        new_edge_index = batch.edge_index[:, correct_idx]

        return new_edge_index, vertex_slice, new_slices, batch.batch

#from tu import TUData
def load_ogb_graph_dataset(root, name):
    raw_dir = osp.join(root, 'raw')
    dataset = PygGraphPropPredDataset(name, raw_dir)
    idx = dataset.get_idx_split()

    return dataset, idx['train'], idx['valid'], idx['test']


def get_cin_tudata(data,batch_size):

    if data=='ogbg-molhiv':
        path = osp.dirname(osp.realpath(__file__))
        original_data = PygGraphPropPredDataset(name='ogbg-molhiv', root=path)
        split_idx = original_data.get_idx_split()
        datasets = OGBDataset("", data, max_ring_size=2)
    
    else:
        datasets = TUData("",data, max_dim=2, num_classes=2,
                fold=1, degree_as_tag=False, include_down_adj='store_true',
                init_method='sum', max_ring_size=None)

        original_data =  get_tudataset(data, rwr=False, cleaned=False)
        split_idx = datasets.get_idx_split()

    org_train = [dats for dats in original_data[split_idx['train']]]
    org_val = [dats for dats in original_data[split_idx['valid']]]
    org_test = [dats for dats in original_data[split_idx['test']]]

    torch.manual_seed(42)
    train_loader = DataLoader(datasets[split_idx['train']], batch_size=batch_size, shuffle=False,max_dim=datasets.max_dim)
    val_loader = DataLoader(datasets[split_idx['valid']], batch_size=len(split_idx['valid']), shuffle=False,max_dim=datasets.max_dim)
    test_loader = DataLoader(datasets[split_idx['test']], batch_size=len(split_idx['test']), shuffle=False,max_dim=datasets.max_dim)

    org_train_loader = PyGDataloader(org_train,batch_size=batch_size,shuffle=False)
    org_val_loader = PyGDataloader(org_val,batch_size=len(split_idx['valid']),shuffle=False)
    org_test_loader = PyGDataloader(org_test,batch_size=len(split_idx['test']),shuffle=False)

    Dloader = {"train": train_loader,"valid": val_loader, "test":test_loader}
    Dloader_org = {"train":org_train_loader,"valid": org_val_loader, "test":org_test_loader}

    return Dloader,Dloader_org


def train_test_split(dataset,train_ratio,val_ratio,batch_size,prop_idx):
    dataset_len = len(dataset)
    y = torch.cat([dataset[i].y[:,prop_idx] for i in range(dataset_len) ])

    global_mean = y.mean()
    global_std = y.std()
    idxs = np.arange(dataset_len,dtype=int)
    idxs = np.random.default_rng(42).permutation(idxs)
    
    train_size = 110000#110000
    val_size = 10000
    #test_size = 1000

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : ]

    train_loader = DataLoader(dataset[idx_train],batch_size=batch_size,shuffle=False)
    val_loader = DataLoader(dataset[idx_val],batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(dataset[idx_test],batch_size=batch_size,shuffle=False)

    dataloader = {"train": train_loader,"valid": val_loader, "test":test_loader}
    return dataloader,global_mean,global_std



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


def compute_ref_energy(atomic_id):
    e0 = 0
    for idx in atomic_id:
        e0 = e0 + ATOM_REF_ENRGY[idx]

    return e0


def get_batched_data(n_nodes,x,edge_mask,batch_size):
        batches = []
        num_neigh = 3
        for num in range(batch_size):
            mask = x[num].abs().sum(dim=1).bool()
            edges = knn_graph(x[num][mask].cpu(),num_neigh,loop=False).to(x.device)
            batches.append(Data(x=x[num],edge_index=edges))

        return DataLoader(batches,shuffle=False,batch_size=len(batches))


def get_data_rmd17(name_mol,file_path,train_idx,test_idx,batch_size=8):
    Train_data = []
    Test_data = []
    data = np.load(file_path) 
    Energy = data['energies']
    Forces = data['forces']
    Positions = data['coords']
    Atom_id = data['nuclear_charges']

    train_traj  = pd.read_csv(train_idx,header=None).to_numpy().flatten().tolist()
    test_traj  = pd.read_csv(test_idx,header=None).to_numpy().flatten().tolist()

    mol = pubchem_atoms_search(name=name_mol)
    e0 = compute_ref_energy(Atom_id)
    Energy_list = []
    Force_list = []
    for entry in train_traj:
        energy = Energy[entry]/EV_TO_KCAL_MOL - e0
        energy = energy.reshape(-1,1)
        forces = Forces[entry]/EV_TO_KCAL_MOL
        forces = forces.reshape(-1,3)

        Energy_list.append(energy)
        Force_list.append(forces)

        positions = Positions[entry].reshape(-1,3)
        atomic_id = Atom_id.reshape(-1,1)

        graph_data_object = construct_graph(mol,energy,forces,positions,atomic_id)
        Train_data.append(graph_data_object)

    for entry in test_traj:
        energy = Energy[entry]/EV_TO_KCAL_MOL -e0
        energy = energy.reshape(-1,1)

        forces = Forces[entry]/EV_TO_KCAL_MOL
        forces = forces.reshape(-1,3)

        Energy_list.append(energy)
        Force_list.append(forces)


        positions = Positions[entry].reshape(-1,3)
        atomic_id = Atom_id.reshape(-1,1)

        graph_data_object = construct_graph(mol,energy,forces,positions,atomic_id)
        Test_data.append(graph_data_object)



    Train_Data = Train_data[0:int(0.95*len(train_traj))]
    Val_data = Train_data[int(0.95*len(train_traj)):]
    #E_arr = np.array(Energy_list)
    #F_arr = np.array(Force_list).reshape(-1,10,3)
    #energy_mean,energy_std = np.mean(E_arr),np.std(E_arr)
    #force_mean,force_std = np.mean(F_arr),np.std(F_arr)

    train_loader = DataLoader(Train_Data,batch_size=10,shuffle=False)
    val_loader = DataLoader(Val_data,batch_size=10,shuffle=False)
    test_loader = DataLoader(Test_data,batch_size=10,shuffle=False)


    dataloader = {"train": train_loader,"valid": val_loader, "test":test_loader}

    return dataloader


def compute_force(energy,data):
    return torch.autograd.grad(outputs=energy, inputs=data.pos, grad_outputs=torch.ones_like(energy),retain_graph=True, create_graph=True, only_inputs=True,allow_unused=True)



def construct_graph(mol,energy,force,position,atom_id):

    cutOff = neighborlist.natural_cutoffs(mol)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    atom_feature = []
    for idx,atom in enumerate(mol):
        atom_feature.append([atom.mass,atom.magmom,atom.charge,atom_id[idx][0]])
        atom.position = position[idx]
        
    neighborList.update(mol)
    matrix = neighborList.get_connectivity_matrix(sparse=False)
    Atom_arr = np.array([atom_feature]).reshape(-1,4)
    edge_start = []
    edge_end = []
    
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row][col] == 1:
                edge_start.append(row)
                edge_end.append(col)
                
    Edge_index = torch.tensor([edge_start,edge_end])
    pos_features = torch.from_numpy(position).view(-1,3)
    energy_feature = torch.tensor(energy).view(-1,1)
    force_feature = torch.from_numpy(force).view(-1,3)
    Node_features = torch.from_numpy(Atom_arr)
    
    return Data(x=Node_features.float(),force=force_feature.float(), edge_index=Edge_index,y=energy_feature.float(),pos=pos_features.float(),edge_attr=None)



class TUData(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0,
                 init_method='sum', seed=0, include_down_adj=False, max_ring_size=None):
        self.name = name
        self.degree_as_tag = degree_as_tag
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        self.graph_list = 0
        if cellular:
            assert max_dim == 2

        super(TUData, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)

        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.fold = fold
        self.seed = seed
        self.graph_list = 0
        dataset = get_tudataset(name)
        seed=42
        skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
        skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
        self.train_ids = train_idx
        self.val_ids = val_test_idx[val_idx]
        self.test_ids = val_test_idx[test_idx]
        # TODO: Add this later to our zip
        # tune_train_filename = os.path.join(self.raw_dir, 'tests_train_split.txt'.format(fold + 1))
        # self.tune_train_ids = np.loadtxt(tune_train_filename, dtype=int).tolist()
        # tune_test_filename = os.path.join(self.raw_dir, 'tests_val_split.txt'.format(fold + 1))
        # self.tune_val_ids = np.loadtxt(tune_test_filename, dtype=int).tolist()
        # self.tune_test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUData, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix
            
    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]
    
    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return ['{}_graph_list_degree_as_tag_{}.pkl'.format(self.name, self.degree_as_tag)]
    
    def download(self):
        # This will process the raw data into a list of PyG Data objs.

        dataset = get_tudataset(self.name, rwr=False, cleaned=False)
        self._num_classes = 2
        #print('Converting graph data into PyG format...')
        self.graph_list = dataset
        
    def process(self):
        dataset = get_tudataset(self.name, rwr=False, cleaned=False)
        if self._cellular:
            print("Converting the dataset accounting for rings...")
            complexes, _, _ = convert_graph_dataset_with_rings(dataset, max_ring_size=self._max_ring_size,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method,
                                                               init_edges=True, init_rings=True)
        else:
            print("Converting the dataset with gudhi...")
            # TODO: eventually remove the following comment
            # What about the init_method here? Adding now, although I remember we had handled this
            complexes, _, _ = convert_graph_dataset_with_gudhi(dataset, expansion_dim=self.max_dim,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method)


        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])


class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, max_ring_size, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, init_method='sum', 
                 include_down_adj=False, simple=False, n_jobs=2):
        self.name = name
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._simple = simple
        self._n_jobs = n_jobs
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, init_method=init_method, 
                                         include_down_adj=include_down_adj, cellular=True)
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']
        
    @property
    def raw_file_names(self):
        name = self.name.replace('-', '_')  # Replacing is to follow OGB folder naming convention
        # The processed graph files are our raw files.
        return [f'{name}/processed/geometric_data_processed.pt']

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']
    
    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and simple feats"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        suffix3 = "-S" if self._simple else ""
        return directory + suffix1 + suffix2 + suffix3

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        tasks = torch.load(self.processed_paths[2])
        return data, slices, idx, tasks

    def process(self):
        
        # At this stage, the graph dataset is already downloaded and processed
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)
        split_idx = dataset.get_idx_split()
        if self._simple:  # Only retain the top two node/edge features
            print('Using simple features')
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
        
        # NB: the init method would basically have no effect if 
        # we use edge features and do not initialize rings. 
        print(f"Converting the {self.name} dataset to a cell complex...")

        complexes, _, _ = convert_graph_dataset_with_gudhi(dataset, expansion_dim=self.max_dim,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method)
        
        #complexes, _, _ = convert_graph_dataset_with_rings(
        #    dataset,
        #    max_ring_size=self._max_ring_size,
        #    include_down_adj=self.include_down_adj,
        #    init_method=self._init_method,
        #    init_edges=self._use_edge_features,
        #    init_rings=False,
        #    n_jobs=self._n_jobs)
        
        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(split_idx, self.processed_paths[1])
        
        print(f'Saving num_tasks in {self.processed_paths[2]}...')
        torch.save(dataset.num_tasks, self.processed_paths[2])


###### Data_utils for CIN stuff #########



def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim+2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim+2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex), so that is very efficient for us.
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
    """
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim+1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def generate_cochain(dim, x, all_upper_index, all_lower_index,
                   all_shared_boundaries, all_shared_coboundaries, cell_tables, boundaries_tables,
                   complex_dim, y=None):
    """Builds a Cochain given all the adjacency data extracted from the complex."""
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_cells_down = len(cell_tables[dim-1]) if dim > 0 else None
    num_cells_up = len(cell_tables[dim+1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                      if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                    if len(all_shared_boundaries[dim]) > 0 else None)
    
    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, cell in enumerate(boundaries_tables[dim]):
            for boundary in cell:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)
        
    if num_cells_down is None:
        assert shared_boundaries is None
    if num_cells_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_cells_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_cells_down >= shared_boundaries.max() + 1

    return Cochain(dim=dim, x=x, upper_index=up_index,
                 lower_index=down_index, shared_coboundaries=shared_coboundaries,
                 shared_boundaries=shared_boundaries, y=y, num_cells_down=num_cells_down,
                 num_cells_up=num_cells_up, boundary_index=boundary_index)


def compute_clique_complex_with_gudhi(x: Tensor, edge_index: Adj, size: int,
                                      expansion_dim: int = 2, y: Tensor = None,
                                      include_down_adj=True,
                                      init_method: str = 'sum') -> Complex:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(expansion_dim)  # Computes the clique complex up to the desired dim.
    complex_dim = simplex_tree.dimension()  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundaries_tables, boundaries, co_boundaries = (
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim))

    # Computes the adjacencies between all the simplexes in the complex
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    # TODO: Make this handle edge features as well and add alternative options to compute this.
    xs = construct_features(x, simplex_tables, init_method)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim+1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum'):
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim+1)]
    for data in dataset:
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,
            expansion_dim=expansion_dim, y=data.y, include_down_adj=include_down_adj,
            init_method=init_method)
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        complexes.append(complex)

    return complexes, dimension, num_features[:dimension+1]


# ---- support for rings as cells

def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    cell_tables, id_maps = build_tables(simplex_tree, size)
    
    # Find rings in the graph
    rings = get_rings(edge_index, max_k=max_k)
    
    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                            simplex_tree, id_maps, simplex_tree.dimension())
    
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
                boundaries_tables[2][-1].append(id_maps[1][boundary])
    
    return boundaries_tables, boundaries, coboundaries


def compute_ring_2complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None, max_k: int = 7,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges=True, init_rings=False) -> Complex:
    """Generates a ring 2-complex of a pyG graph via graph-tool.

    Args:
        x: The feature matrix for the nodes of the graph (shape [num_vertices, num_v_feats])
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        edge_attr: The feature matrix for the edges of the graph (shape [num_edges, num_e_feats])
        size: The number of nodes in the graph
        y: Labels for the graph nodes or a label for the whole graph.
        max_k: maximum length of rings to look for.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # For parallel processing with joblib we need to pass numpy arrays as inputs
    # Therefore, we convert here everything back to a tensor.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Creates the gudhi-based simplicial complex up to edges
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0

    # Builds tables of the cellular complexes at each level and their IDs
    cell_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    assert len(id_maps) <= 3
    complex_dim = len(id_maps)-1

    # Extracts the boundaries and coboundaries of each cell in the complex
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)

    # Computes the adjacencies between all the cells in the complex;
    # here we force complex dimension to be 2
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)
    
    # Construct features for the higher dimensions
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
    
    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        # If we have edge-features we simply use them for 1-cells
        else:
            # If edge_attr is a list of scalar features, make it a matrix
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # Retrieve feats and check edge features are undirected
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            # Build edge feature matrix
            max_id = max(ex.keys())
            edge_feats = []
            assert len(cell_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               cell_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    # Process the dataset in parallel
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    complexes = parallel(delayed(compute_ring_2complex)(
        maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k=max_ring_size,
        include_down_adj=include_down_adj, init_method=init_method,
        init_edges=init_edges, init_rings=init_rings) for data in dataset)

    # NB: here we perform additional checks to verify the order of complexes
    # corresponds to that of input graphs after _parallel_ conversion
    for c, complex in enumerate(complexes):

        # Handle dimension and number of features
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features

        # Validate against graph
        graph = dataset[c]
        if complex.y is None:
            assert graph.y is None
        else:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return complexes, dimension, num_features[:dimension+1]


def load_data(path, dataset, degree_as_tag):
    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (path, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())  # <- this might not be used...!?
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)
    
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
            # ^^^ !? it should probably be replaced by the following one:
            # g.node_tags = [g.g.degree[node] for node in range(len(g.g))]

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        
    # ==================
    # Here we recompute degree encodings with external code,
    # as we observed some unexpected behaviors likely due to
    # incompatibilities w.r.t.  python versions
    # ==================
    def get_node_degrees(graph):
        edge_index = graph.edge_mat
        if edge_index.shape[1] == 0:  # just isolated nodes
            degrees = torch.zeros((graph.node_features.shape[0],1))
        else:
            degrees = torch_geometric.utils.degree(edge_index[0]).unsqueeze(1)

        return degrees
    if degree_as_tag:
        # 1. cumulate node degrees
        degs = torch.cat([get_node_degrees(graph) for graph in g_list], 0)
        # 2. compute unique values
        uniques, corrs = np.unique(degs, return_inverse=True, axis=0)
        # 3. encode
        pointer = 0
        for graph in g_list:
            n = graph.node_features.shape[0]
            hots = torch.LongTensor(corrs[pointer:pointer+n])
            graph.node_features = torch.nn.functional.one_hot(hots, len(uniques)).float()
            pointer += n
    # ====================

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def S2V_to_PyG(data):
    
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())
    
    return new_data
if __name__=='__main__':
    get_cin_tudata('NCI1')
