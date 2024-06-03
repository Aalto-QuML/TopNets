import os.path
import itertools
import h5py
import numpy as np
import pandas as pd
import gudhi as gd

from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.linalg import eigh
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from models.perslay_model import PerslayModel


def get_parameters(dataset):
    if dataset == "MUTAG" or dataset == "PROTEINS":
        dataset_parameters = {"data_type": "graph", "filt_names": ["Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "COX2" or dataset == "DHFR" or dataset == "NCI1" or dataset == "NCI109" or dataset == "IMDB-BINARY" or dataset == "IMDB-MULTI":
        dataset_parameters = {"data_type": "graph", "filt_names": ["Ord0_0.1-hks", "Rel1_0.1-hks", "Ext0_0.1-hks", "Ext1_0.1-hks", "Ord0_10.0-hks", "Rel1_10.0-hks", "Ext0_10.0-hks", "Ext1_10.0-hks"]}
    elif dataset == "ORBIT5K" or dataset == "ORBIT100K":
        dataset_parameters = {"data_type": "orbit", "filt_names": ["Alpha0", "Alpha1"]}
    return dataset_parameters

def get_model(dataset, nf):
    if dataset == "MUTAG":
        plp = {}
        plp["pweight"] = "grid"
        plp["pweight_init"] = torch.rand
        plp["pweight_size"] = (10, 10)
        plp["pweight_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"] = True
        plp["layer"] = "Image"
        plp["image_size"] = (20, 20)
        plp["image_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = torch.rand
        plp["layer_train"] = True
        plp["perm_op"] = "sum"
        perslay_parameters = [plp.copy() for _ in range(4)]

        for i in range(4):
            fmodel = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2),
                nn.Flatten()
            )
            perslay_parameters[i]["final_model"] = fmodel

        rho = nn.Sequential(
            nn.Linear(in_features=nf + 14440, out_features=2),
        )
        model = PerslayModel(diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
        optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-4)
        loss_fn = nn.CrossEntropyLoss()
    elif dataset == "PROTEINS":
            plp = {}
            plp["pweight"] = "grid"
            plp["pweight_init"] = torch.rand
            plp["pweight_size"] = (10, 10)
            plp["pweight_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
            plp["pweight_train"] = True
            plp["layer"] = "Image"
            plp["image_size"] = (15, 15)
            plp["image_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
            plp["lvariance_init"] = torch.rand
            plp["layer_train"] = True
            plp["perm_op"] = "sum"
            perslay_parameters = [plp.copy() for _ in range(4)]

            for i in range(4):
                fmodel = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2),
                    nn.Flatten()
                )
                perslay_parameters[i]["final_model"] = fmodel

            rho = nn.Sequential(
                nn.Linear(in_features=nf + 7840, out_features=2),
            )
            model = PerslayModel(diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
            optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-4)
            loss_fn = nn.CrossEntropyLoss()
    elif dataset == 'IMDB-BINARY':
        plp = {}
        plp["pweight"] = "grid"
        plp["pweight_init"] = torch.rand
        plp["pweight_size"] = (20, 20)
        plp["pweight_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"] = True
        plp["layer"] = "Image"
        plp["image_size"] = (20, 20)
        plp["image_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
        plp["lvariance_init"] = torch.rand
        plp["layer_train"] = True
        plp["perm_op"] = "sum"
        perslay_parameters = [plp.copy() for _ in range(4)]

        for i in range(4):
            fmodel = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2),
                nn.Flatten()
            )
            perslay_parameters[i]["final_model"] = fmodel

        rho = nn.Sequential(
            nn.Linear(in_features=nf + 14440, out_features=2),
        )
        model = PerslayModel(diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
        optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-4)
        loss_fn = nn.CrossEntropyLoss()
    elif dataset == 'NCI1' or dataset == 'NCI109':
        plp = {}
        plp["pweight"] = "grid"
        plp["pweight_init"] = torch.rand
        plp["pweight_size"] = (10, 10)
        plp["pweight_bnds"] = ((-0.001, 1.001), (-0.001, 1.001))
        plp["pweight_train"] = True
        plp["layer"] = "PermutationEquivariant"
        plp["lpeq"] = [(25, None), (25, 'max')]
        plp["lweight_init"] = torch.rand
        plp["lbias_init"] = torch.rand
        plp["lgamma_init"] = torch.rand
        plp["layer_train"] = True
        plp["perm_op"] = "sum"
        perslay_parameters = [plp.copy() for _ in range(4)]

        for i in range(4):
            perslay_parameters[i]["final_model"] = torch.nn.Identity()

        rho = nn.Sequential(
            nn.Linear(in_features=nf + 100, out_features=2),
        )
        model = PerslayModel(diagdim=2, perslay_parameters=perslay_parameters, rho=rho)
        optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-4)
        loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn

def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack([np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0, 2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1


def generate_diagrams_and_features(dataset, path_dataset=""):
    dataset_parameters = get_parameters(dataset)
    dataset_type = dataset_parameters["data_type"]

    if "REDDIT" in dataset:
        print("Unfortunately, REDDIT data are not available yet for memory issues.\n")
        print("Moreover, the link we used to download the data,")
        print("http://www.mit.edu/~pinary/kdd/datasets.tar.gz")
        print("is down at the commit time (May 23rd).")
        print("We will update this repository when we figure out a workaround.")
        return

    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5", "w")
    list_filtrations = dataset_parameters["filt_names"]
    [diag_file.create_group(str(filtration)) for filtration in dataset_parameters["filt_names"]]

    if dataset_type == "graph":
        list_hks_times = np.unique([filtration.split("_")[1] for filtration in list_filtrations])
        pad_size = 1
        for graph_name in os.listdir(path_dataset + "mat/"):
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            pad_size = np.max((A.shape[0], pad_size))

        feature_names = ["eval"+str(i) for i in range(pad_size)] + [name+"-percent"+str(i) for name, i in itertools.product([f for f in list_hks_times if "hks" in f], 10*np.arange(11))]
        features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))), columns=["label"] + feature_names)

        for idx, graph_name in enumerate(os.listdir(path_dataset + "mat/")):
            name = graph_name.split("_")
            gid = int(name[name.index("gid") + 1]) - 1
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            num_vertices = A.shape[0]
            label = int(name[name.index("lb") + 1])

            L = csgraph.laplacian(A, normed=True)
            egvals, egvectors = eigh(L)
            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for fhks in list_hks_times:
                hks_time = float(fhks.split("-")[0])
                filtration_val = hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val)
                diag_file.create_dataset("Ord0_" + str(hks_time) + "-hks/" + str(gid), data=dgmOrd0)
                diag_file.create_dataset("Ext0_" + str(hks_time) + "-hks/" + str(gid), data=dgmExt0)
                diag_file.create_dataset("Rel1_" + str(hks_time) + "-hks/" + str(gid), data=dgmRel1)
                diag_file.create_dataset("Ext1_" + str(hks_time) + "-hks/" + str(gid), data=dgmExt1)
                graph_features.append(
                    np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time), 10 * np.arange(11)))
            features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)

        features["label"] = features["label"].astype(int)

    features.to_csv(path_dataset + dataset + ".csv")
    return diag_file.close()


def load_data(dataset, path_dataset="", filtrations=[], verbose=False):
    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys()) if len(filtrations) == 0 else filtrations

    diags_dict = dict()
    if len(filts) == 0:
        filts = diagfile.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diagfile[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diagfile[filtration][str(diag)]))
        diags_dict[filtration] = list_dgm

    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    F = np.array(feat)[:, 1:]
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diags_dict, F, L
