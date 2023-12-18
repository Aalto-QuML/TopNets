import torch
import numpy as np
import torch.nn as nn

from layers.permutation_equivariant_layer import (
    permutation_equivariant_layer,
    image_layer,
)


class PerslayModel(nn.Module):
    def __init__(self, diagdim, perslay_parameters, rho):
        super(PerslayModel, self).__init__()
        self.diagdim = diagdim
        self.perslay_parameters = perslay_parameters
        self.rho = rho
        self.final_models = torch.nn.ParameterList()

        self.vars = [
            torch.nn.ParameterList() for _ in range(len(self.perslay_parameters))
        ]
        for nf, plp in enumerate(self.perslay_parameters):
            self.final_models.append(plp["final_model"])
            weight = plp["pweight"]
            if weight is not None:
                Winit, Wtrain = plp["pweight_init"], plp["pweight_train"]
                W = Winit if not callable(Winit) else Winit(plp["pweight_size"])
                W = nn.Parameter(W, requires_grad=Wtrain)
                self.vars[nf].append(W)

            layer, Ltrain = plp["layer"], plp["layer_train"]

            if layer == "PermutationEquivariant":
                Lpeq, LWinit, LBinit, LGinit = (
                    plp["lpeq"],
                    plp["lweight_init"],
                    plp["lbias_init"],
                    plp["lgamma_init"],
                )
                LW, LB, LG = [], [], []
                for idx, (dim, pop) in enumerate(Lpeq):
                    dim_before = self.diagdim if idx == 0 else Lpeq[idx - 1][0]
                    LWiv = LWinit([dim_before, dim]) if callable(LWinit) else LWinit
                    LBiv = LBinit([dim]) if callable(LBinit) else LBinit
                    LW.append(nn.Parameter(LWiv, requires_grad=Ltrain))
                    LB.append(nn.Parameter(LBiv, requires_grad=Ltrain))
                    if pop is not None:
                        LGiv = LGinit([dim_before, dim]) if callable(LGinit) else LGinit
                        LG.append(nn.Parameter(LGiv, requires_grad=Ltrain))
                    else:
                        LG.append(None)
                self.vars[nf].append(nn.ParameterList([LW, LB, LG]))
            elif layer == "Image":
                LVinit = plp["lvariance_init"]
                LViv = LVinit if not callable(LVinit) else LVinit([1])
                LV = torch.nn.Parameter(LViv, requires_grad=Ltrain)
                self.vars[nf].append(LV)
        self.vars = nn.ParameterList(self.vars)

    def compute_representations(self, diags):
        list_v = []

        for nf, plp in enumerate(self.perslay_parameters):
            diag = diags[nf]

            N, dimension_diag = diag.shape[1], diag.shape[2]
            tensor_mask = diag[:, :, dimension_diag - 1]
            tensor_diag = diag[:, :, : dimension_diag - 1]

            W = self.vars[nf][0]

            if plp["pweight"] == "power":
                p = plp["pweight_power"]
                weight = W * torch.abs(
                    tensor_diag[:, :, 1:2] - tensor_diag[:, :, 0:1]
                ).pow(p)

            elif plp["pweight"] == "grid":
                grid_shape = W.shape
                indices = []
                for dim in range(dimension_diag - 1):
                    m, M = plp["pweight_bnds"][dim]
                    coords = tensor_diag[:, :, dim : dim + 1]
                    ids = grid_shape[dim] * (coords - m) / (M - m)
                    indices.append(ids.to(torch.int32))
                indices = torch.cat(indices, dim=2).long()
                weight = W[indices[:, :, 0], indices[:, :, 1]].unsqueeze(-1)

            elif plp["pweight"] == "gmix":
                M, V = W[:2, :].unsqueeze(0).unsqueeze(0), W[2:, :].unsqueeze(
                    0
                ).unsqueeze(0)
                bc_inp = tensor_diag.unsqueeze(-1)
                weight = torch.sum(
                    torch.exp(
                        torch.sum(
                            -torch.mul(torch.square(bc_inp - M), torch.square(V)), dim=2
                        )
                    ),
                    dim=2,
                ).unsqueeze(-1)

            #            weight = torch.abs(tensor_diag[:, :, 1:2] - tensor_diag[:, :, 0:1]).pow(W)

            lvars = self.vars[nf][1]

            if plp["layer"] == "PermutationEquivariant":
                for idx, (dim, pop) in enumerate(plp["lpeq"]):
                    tensor_diag = permutation_equivariant_layer(
                        tensor_diag,
                        dim,
                        pop,
                        lvars[0][idx],
                        lvars[1][idx],
                        lvars[2][idx],
                    )
            elif plp["layer"] == "Image":
                tensor_diag = image_layer(
                    tensor_diag, plp["image_size"], plp["image_bnds"], lvars
                )

            output_dim = tensor_diag.dim() - 2
            if plp["pweight"] is not None:
                for _ in range(output_dim - 1):
                    weight = weight.unsqueeze(-1)
                tiled_weight = weight.expand(
                    tensor_diag.shape[0], tensor_diag.shape[1], *tensor_diag.shape[2:]
                )
                tensor_diag = tensor_diag * tiled_weight

            for _ in range(output_dim):
                tensor_mask = tensor_mask.unsqueeze(-1)
            tiled_mask = tensor_mask.expand(-1, -1, *tensor_diag.shape[2:])
            masked_layer = torch.mul(tensor_diag, tiled_mask)

            if plp["perm_op"] == "sum":
                vector = masked_layer.sum(dim=1)
            elif plp["perm_op"] == "max":
                vector = masked_layer.max(dim=1).values
            elif plp["perm_op"] == "mean":
                vector = masked_layer.mean(dim=1)

            if not isinstance(plp["final_model"], torch.nn.Identity):
                vector = self.final_models[nf](vector.permute(0, 3, 1, 2))
            list_v.append(vector)

        representations = torch.cat(list_v, dim=1)
        return representations

    def forward(self, inputs):
        diags, feats = inputs[0], inputs[1]
        representations = self.compute_representations(diags)
        concat_representations = torch.cat([representations, feats], dim=1)
        final_representations = (
            self.rho(concat_representations)
            if self.rho != "identity"
            else concat_representations
        )
        return final_representations
