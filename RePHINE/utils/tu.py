import os
import torch
import pickle
import numpy as np
import os.path as osp
from utils.data_utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings
from utils.dataset import InMemoryComplexDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.utils import degree
#from torch_geometric.datasets import TUDataset


class FilterConstant(object):
  def __init__(self, dim):
    self.dim = dim

  def __call__(self, data):
    data.x = torch.ones(data.num_nodes, self.dim)
    return data


def get_tudataset(name, rwr=False, cleaned=False):
  transform = None
  if rwr:
    transform = None
  path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''))
  dataset = TUDataset(path, name, pre_transform=transform, use_edge_attr=rwr, cleaned=cleaned)

  if not hasattr(dataset, 'x'):
    max_degree = 0
    degs = []
    for data in dataset:
      degs += [degree(data.edge_index[0], dtype=torch.long)]
      max_degree = max(max_degree, degs[-1].max().item())
    dataset.transform = FilterConstant(10)  # T.OneHotDegree(max_degree)
  return dataset



class TUData(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False, fold=0,
                 init_method='sum', seed=0, include_down_adj=False, max_ring_size=None):
        self.name = name
        self.degree_as_tag = degree_as_tag
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2

        super(TUData, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)

        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.fold = fold
        self.seed = seed
        dataset = get_tudataset(name)
        seed=42
        skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
        skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
        self.train_ids = train_idx
        self.val_ids = val_idx
        self.test_ids = test_idx
        # TODO: Add this later to our zip
        # tune_train_filename = os.path.join(self.raw_dir, 'tests_train_split.txt'.format(fold + 1))
        # self.tune_train_ids = np.loadtxt(tune_train_filename, dtype=int).tolist()
        # tune_test_filename = os.path.join(self.raw_dir, 'tests_val_split.txt'.format(fold + 1))
        # self.tune_val_ids = np.loadtxt(tune_test_filename, dtype=int).tolist()
        # self.tune_test_ids = None

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUDataset, self).processed_dir
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
        data, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], 'wb') as handle:
            pickle.dump(graph_list, handle)
        
    def process(self):
        with open(self.raw_paths[0], 'rb') as handle:
            graph_list = pickle.load(handle)        
        
        if self._cellular:
            print("Converting the dataset accounting for rings...")
            complexes, _, _ = convert_graph_dataset_with_rings(graph_list, max_ring_size=self._max_ring_size,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method,
                                                               init_edges=True, init_rings=True)
        else:
            print("Converting the dataset with gudhi...")
            # TODO: eventually remove the following comment
            # What about the init_method here? Adding now, although I remember we had handled this
            complexes, _, _ = convert_graph_dataset_with_gudhi(graph_list, expansion_dim=self.max_dim,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method)

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])