import argparse

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from reproducibility.utils import set_seeds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["cub08", "cub10", "cub12"], default="cub06"
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_alien_nodes", type=int, default=1)

args = parser.parse_args()

set_seeds(args.seed)
graphs = nx.read_graph6(f"./data_toy/{args.dataset}.g6")
dataset = []
cut = args.num_alien_nodes
for i, g in enumerate(graphs):
    pyg_graph = from_networkx(g)
    x = torch.ones(len(g.nodes))
    indices = torch.randperm(len(g.nodes))[:cut]
    x[indices] = -1.0
    label = 1
    if i >= len(graphs) / 2:
        label = 0
    y = torch.tensor(label, dtype=torch.long)
    new_g = Data(x=x.unsqueeze(1), edge_index=pyg_graph.edge_index, y=y)
    dataset.append(new_g)
# print('Number of graphs:', len(graphs))
# print('Number of nodes:', len(graphs[0].nodes))
torch.save(dataset, f"./data_toy/{args.dataset}-{cut}_seed-{args.seed}.dat")
