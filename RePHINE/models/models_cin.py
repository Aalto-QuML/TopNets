import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import JumpingKnowledge
from models.layers_cin import (
    CINConv, EdgeCINConv, SparseCINConv, CINppConv,DummyCellularMessagePassing, OrientedConv,InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce)
from models.nn import get_nonlinearity, get_pooling_fn, pool_complex, get_graph_norm
from models.data_complex import ComplexBatch, CochainBatch
from layers.rephine_layer import RephineLayer,RephineLayer_Equiv
from torchdiffeq import odeint as odeint
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class CIN0(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum'):
        super(CIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)

        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            self.convs.append(
                CINConv(layer_dim, layer_dim,
                    conv_up, conv_down, conv_update, train_eps=False, max_dim=self.max_dim))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        if jump_mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim)
            xs = conv(*params)
            data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)
        pooled_xs = self.pool_complex(xs, data)
        x = pooled_xs.sum(dim=0)

        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class CIN0_PH(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,diagram,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum'):
        super(CIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(num_layers):
            topo = RephineLayer_Equiv(
                n_features=hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type=diagram,
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"
        final_dim = (
            hidden + len(self.ph_layers) * self.out_ph
            if self.ph_pooling_type == "cat"
            else hidden + self.out_ph
        )



        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            self.convs.append(
                CINConv(layer_dim, layer_dim,
                    conv_up, conv_down, conv_update, train_eps=False, max_dim=self.max_dim))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        if jump_mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        
        self.lin2 = Linear(final_dim, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        ph_vectors = []
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim)
            xs = conv(*params)
            ph_vectors += [self.ph_layers[i](xs,data)]
            data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)
        pooled_xs = self.pool_complex(xs, data)
        x = pooled_xs.sum(dim=0)

        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.cat([x,ph_embedding],dim=1)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__



class SparseCIN(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum',
                 train_eps=False, final_hidden_multiplier: int = 2, use_coboundaries=False,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 graph_norm='bn'):
        super(SparseCIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))
        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        self.graph_norm = get_graph_norm(graph_norm)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None, passed_msg_up_nn=None,
                    passed_update_up_nn=None, passed_update_boundaries_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # print(batch_size)
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)

        new_xs = []
        for i in range(self.max_dim + 1):
            new_xs.append(pooled_xs[i])
        return new_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        xs, jump_xs = None, None
        res = {}
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
            # if i == len(self.convs) - 2:
            #     start_to_process = 1
            # if i == len(self.convs) - 1:
            #     start_to_process = 2
            xs = conv(*params, start_to_process=start_to_process)
            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = self.pool_complex(xs, data)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__


class SparseCIN_PH(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,diagram,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum',
                 train_eps=False, final_hidden_multiplier: int = 2, use_coboundaries=False,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 graph_norm='bn'):
        super(SparseCIN_PH, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))
        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        self.graph_norm = get_graph_norm(graph_norm)
        act_module = get_nonlinearity(nonlinearity, return_module=True)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(num_layers):
            topo = RephineLayer(
                n_features=hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type=diagram,
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"
        final_dim = (
            hidden + len(self.ph_layers) * self.out_ph
            if self.ph_pooling_type == "cat"
            else hidden + self.out_ph
        )


        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None, passed_msg_up_nn=None,
                    passed_update_up_nn=None, passed_update_boundaries_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden + self.out_ph, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # print(batch_size)
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)

        new_xs = []
        for i in range(self.max_dim + 1):
            new_xs.append(pooled_xs[i])
        return new_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, data_org,include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        xs, jump_xs = None, None
        res = {}
        ph_vectors = []
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
            #params = [param.to(data.device) for param in params]
            start_to_process = 0
            # if i == len(self.convs) - 2:
            #     start_to_process = 1
            # if i == len(self.convs) - 1:
            #     start_to_process = 2
            xs = conv(*params, start_to_process=start_to_process)

            ph_vectors += [self.ph_layers[c](xs[0],data_org)]


            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = self.pool_complex(xs, data)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = torch.cat([x,ph_embedding],dim=1)
        x = self.lin2(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__



class SparseCIN_PH_Cont(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,diagram,solver,nsteps,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2, jump_mode=None, nonlinearity='relu', readout='sum',
                 train_eps=False, final_hidden_multiplier: int = 2, use_coboundaries=False,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 graph_norm='bn'):
        super(SparseCIN_PH_Cont, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))
        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        self.graph_norm = get_graph_norm(graph_norm)
        act_module = get_nonlinearity(nonlinearity, return_module=True)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []
        self.solver = solver
        self.n_steps = nsteps

        for i in range(nsteps):
            topo = RephineLayer(
                n_features=hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type=diagram,
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"
        final_dim = (
            hidden + len(self.ph_layers) * self.out_ph
            if self.ph_pooling_type == "cat"
            else hidden + self.out_ph
        )
        self.data_org = 0
        embed_mlp = []
        for i in range(max_dim+1):
            embed_mlp.append(torch.nn.Linear(num_input_features, hidden))

        self.embed = nn.ModuleList(embed_mlp)

        for i in range(num_layers):
            layer_dim = hidden + 1 if i==0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None, passed_msg_up_nn=None,
                    passed_update_up_nn=None, passed_update_boundaries_nn=None,
                    train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden + self.out_ph, num_classes)
        self.data_cin = 0
        self.ph_vectors = []

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()


    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # print(batch_size)
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)

        new_xs = []
        for i in range(self.max_dim + 1):
            new_xs.append(pooled_xs[i])
        return new_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs


    def ode(self,t,xs):
        xs = list(xs)
        for idx in range(len(xs)):
            tt = torch.ones_like(xs[idx][:, :1]) * t
            xs[idx] = torch.cat([tt.float(), xs[idx]], 1)
        self.data_cin.set_xs(xs)
        for c, conv in enumerate(self.convs):
                params = self.data_cin.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
                start_to_process = 0
                xs = conv(*params, start_to_process=start_to_process)

                self.data_cin.set_xs(xs)
                if self.jump_mode is not None:
                    if jump_xs is None:
                        jump_xs = [[] for _ in xs]
                    for i, x in enumerate(xs):
                        jump_xs[i] += [x]
        return tuple(xs)


    def forward(self, data: ComplexBatch, data_org,include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        self.data_org = data_org
        self.data_cin = data
        time_steps = torch.linspace(0,1,steps=self.n_steps)
        ph_vectors = []


        ### Initial embedding ###
        params = self.data_cin.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        xs_embed = []
        for idx in range(len(params)):
            xs_embed.append(self.embed[idx](params[idx].x))
            
        self.data_cin.set_xs(xs_embed)


        ode_rhs  = lambda t,x: self.ode(t,x)
        final_xs = odeint(ode_rhs,tuple(xs_embed),time_steps,method=self.solver,atol=1e-3,rtol=1e-3)
        for i in range(self.n_steps):
            ph_vectors += [self.ph_layers[i](final_xs[0][i],data_org)]

        final_xs = [final_xs[dim][-1] for dim in range(len(final_xs))]

        xs = self.pool_complex(final_xs, data)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = torch.cat([x,ph_embedding],dim=1)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

class CINpp(SparseCIN):
    """CINpp
    
    """
    def __init__(self, num_input_features, num_classes, num_layers, hidden, 
                 dropout_rate: float = 0.5, max_dim: int = 2, jump_mode=None, 
                 nonlinearity='relu', readout='sum', train_eps=False, 
                 final_hidden_multiplier: int = 2, use_coboundaries=False, 
                 readout_dims=(0, 1, 2), final_readout='sum', 
                 apply_dropout_before='lin2', graph_norm='bn'):
        super(CINpp, self).__init__(num_input_features, num_classes, num_layers, hidden,
                                    dropout_rate, max_dim, jump_mode, nonlinearity,
                                    readout, train_eps, final_hidden_multiplier,
                                    use_coboundaries, readout_dims, final_readout,
                                    apply_dropout_before, graph_norm)
        self.convs = torch.nn.ModuleList()
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                CINppConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None, passed_msg_up_nn=None,
                    passed_msg_down_nn=None, passed_update_up_nn=None, passed_update_down_nn=None,
                    passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))




class EdgeCIN0(torch.nn.Module):
    """
    A variant of CIN0 operating up to edge level. It may optionally ignore two_cell features.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.5,
                 jump_mode=None, nonlinearity='relu', include_top_features=True,
                 update_top_features=True,
                 readout='sum'):
        super(EdgeCIN0, self).__init__()

        self.max_dim = 1
        self.include_top_features = include_top_features
        # If the top features are included, then they can be updated by a network.
        self.update_top_features = include_top_features and update_top_features
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.update_top_nns = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            v_conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            e_conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden))
            v_conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            e_conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            e_conv_inp_dim = layer_dim * 2 if include_top_features else layer_dim
            e_conv_up = Sequential(
                Linear(e_conv_inp_dim, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim))
            self.convs.append(
                EdgeCINConv(layer_dim, layer_dim, v_conv_up, e_conv_down, e_conv_up,
                    v_conv_update, e_conv_update, train_eps=False))
            if self.update_top_features and i < num_layers - 1:
                self.update_top_nns.append(Sequential(
                    Linear(layer_dim, hidden),
                    conv_nonlinearity(),
                    Linear(hidden, hidden),
                    conv_nonlinearity(),
                    BN(hidden))
                )

        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        if jump_mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for net in self.update_top_nns:
            net.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim,
                include_top_features=self.include_top_features)
            xs = conv(*params)
            # If we are at the last convolutional layer, we do not need to update after
            # We also check two_cell features do indeed exist in this batch before doing this.
            if self.update_top_features and c < len(self.convs) - 1 and 2 in data.cochains:
                top_x = self.update_top_nns[c](data.cochains[2].x)
                data.set_xs(xs + [top_x])
            else:
                data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        pooled_xs = self.pool_complex(xs, data)
        x = pooled_xs.sum(dim=0)

        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class Dummy(torch.nn.Module):
    """
    A dummy cellular network model.
    No parameters in the convolutional layers.
    Readout at each layer is by summation.
    Outputs are computed by one single linear layer.
    """

    def __init__(self, num_input_features, num_classes, num_layers, max_dim: int = 2,
                 readout='sum'):
        super(Dummy, self).__init__()

        self.max_dim = max_dim
        self.convs = torch.nn.ModuleList()
        self.pooling_fn = get_pooling_fn(readout)
        for i in range(num_layers):
            self.convs.append(DummyCellularMessagePassing(max_dim=self.max_dim))
        self.lin = Linear(num_input_features, num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data: ComplexBatch):
        xs = None
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params()
            xs = conv(*params)
            data.set_xs(xs)

        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        # We assume that all layers have the same feature size.
        # Note that levels where we do MP at but where there was no data are set to 0.
        # TODO: shall we retain the device as an attribute of self? then `device=batch_size.device`
        # would become `device=self.device`
        pooled_xs = torch.zeros(self.max_dim + 1, batch_size, xs[0].size(-1),
            device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            # Otherwise, if we have complexes with no cells at certain levels, the wrong
            # shape could be inferred automatically from data.cochains[i].batch.
            # This makes sure the output tensor will have the right dimensions.
            pooled_xs[i, :, :] = self.pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        # Reduce across the levels of the complexes
        x = pooled_xs.sum(dim=0)

        x = self.lin(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class EdgeOrient(torch.nn.Module):
    """
    A model for edge-defined signals taking edge orientation into account.
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.0, jump_mode=None, nonlinearity='id', readout='sum',
                 fully_invar=False):
        super(EdgeOrient, self).__init__()

        self.max_dim = 1
        self.fully_invar = fully_invar
        orient = not self.fully_invar
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            # !!!!! Biases must be set to false. Otherwise, the model is not equivariant !!!!
            update_up = Linear(layer_dim, hidden, bias=False)
            update_down = Linear(layer_dim, hidden, bias=False)
            update = Linear(layer_dim, hidden, bias=False)

            self.convs.append(
                OrientedConv(dim=1, up_msg_size=layer_dim, down_msg_size=layer_dim,
                    update_up_nn=update_up, update_down_nn=update_down, update_nn=update,
                    act_fn=get_nonlinearity(nonlinearity, return_module=False), orient=orient))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: CochainBatch, include_partial=False):
        if self.fully_invar:
            data.x = torch.abs(data.x)
        for c, conv in enumerate(self.convs):
            x = conv(data)
            data.x = x

        cell_pred = x

        # To obtain orientation invariance, we take the absolute value of the features.
        # Unless we did that already before the first layer.
        batch_size = data.batch.max() + 1
        if not self.fully_invar:
            x = torch.abs(x)
        x = self.pooling_fn(x, data.batch, size=batch_size)

        # At this point we have invariance: we can use any non-linearity we like.
        # Here, independently from previous non-linearities, we choose ReLU.
        # Note that this makes the model non-linear even when employing identity
        # in previous layers.
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        if include_partial:
            return x, cell_pred
        return x

    def __repr__(self):
        return self.__class__.__name__


class EdgeMPNN(torch.nn.Module):
    """
    An MPNN operating in the line graph.
    """

    def __init__(self, num_input_features, num_classes, num_layers, hidden,
                 dropout_rate: float = 0.0, jump_mode=None, nonlinearity='relu', readout='sum',
                 fully_invar=True):
        super(EdgeMPNN, self).__init__()

        self.max_dim = 1
        self.dropout_rate = dropout_rate
        self.fully_invar = fully_invar
        orient = not self.fully_invar
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            # We pass this lambda function to discard upper adjacencies
            update_up = lambda x: 0
            update_down = Linear(layer_dim, hidden, bias=False)
            update = Linear(layer_dim, hidden, bias=False)
            self.convs.append(
                OrientedConv(dim=1, up_msg_size=layer_dim, down_msg_size=layer_dim,
                    update_up_nn=update_up, update_down_nn=update_down, update_nn=update,
                    act_fn=get_nonlinearity(nonlinearity, return_module=False), orient=orient))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: CochainBatch, include_partial=False):
        if self.fully_invar:
            data.x = torch.abs(data.x)
        for c, conv in enumerate(self.convs):
            x = conv(data)
            data.x = x
        cell_pred = x

        batch_size = data.batch.max() + 1
        if not self.fully_invar:
            x = torch.abs(x)
        x = self.pooling_fn(x, data.batch, size=batch_size)

        # At this point we have invariance: we can use any non-linearity we like.
        # Here, independently from previous non-linearities, we choose ReLU.
        # Note that this makes the model non-linear even when employing identity 
        # in previous layers.
        x = torch.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        if include_partial:
            return x, cell_pred
        return x

    def __repr__(self):
        return self.__class__.__name__


class MessagePassingAgnostic(torch.nn.Module):
    """
    A model which does not perform any message passing.
    Initial simplicial/cell representations are obtained by applying a dense layer, instead.
    Sort of resembles a 'DeepSets'-likes architecture but on Simplicial/Cell Complexes.
    """
    def __init__(self, num_input_features, num_classes, hidden, dropout_rate: float = 0.5,
                 max_dim: int = 2, nonlinearity='relu', readout='sum'):
        super(MessagePassingAgnostic, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.readout_type = readout
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        self.lin0s = torch.nn.ModuleList()
        for dim in range(max_dim + 1):
            self.lin0s.append(Linear(num_input_features, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for lin0 in self.lin0s:
            lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ComplexBatch):
        
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list()
        for dim in range(len(params)):
            x_dim = params[dim].x
            x_dim = self.lin0s[dim](x_dim)
            xs.append(self.act(x_dim))
        pooled_xs = pool_complex(xs, data, self.max_dim, self.readout_type)
        pooled_xs = self.act(self.lin1(pooled_xs))
        x = pooled_xs.sum(dim=0)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class OGBEmbedSparseCIN(torch.nn.Module):
    """
    A cellular version of GIN with some tailoring to nimbly work on molecules from the ogbg-mol* dataset.
    It uses OGB atom and bond encoders.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, out_size, num_layers, hidden, dropout_rate: float = 0.5, 
                 indropout_rate: float = 0.0, max_dim: int = 2, jump_mode=None,
                 nonlinearity='relu', readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
                 graph_norm='bn'):
        super(OGBEmbedSparseCIN, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))

        if embed_dim is None:
            embed_dim = hidden
        self.v_embed_init = AtomEncoder(embed_dim)

        self.e_embed_init = None
        if embed_edge:
            self.e_embed_init = BondEncoder(embed_dim)
        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.in_dropout_rate = indropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.readout = readout
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_update_up_nn=None,
                    passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        res = {}

        # Embed and populate higher-levels
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list(self.init_conv(*params))

        # Apply dropout on the input features
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)

        data.set_xs(xs)

        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = pool_complex(xs, data, self.max_dim, self.readout)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__


class OGBEmbedSparseCIN_PH(torch.nn.Module):
    """
    A cellular version of GIN with some tailoring to nimbly work on molecules from the ogbg-mol* dataset.
    It uses OGB atom and bond encoders.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, out_size, num_layers, hidden,diagram, dropout_rate: float = 0.5, 
                 indropout_rate: float = 0.0, max_dim: int = 2, jump_mode=None,
                 nonlinearity='relu', readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
                 graph_norm='bn'):
        super(OGBEmbedSparseCIN_PH, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))

        if embed_dim is None:
            embed_dim = hidden
        self.v_embed_init = AtomEncoder(embed_dim)

        self.e_embed_init = None
        if embed_edge:
            self.e_embed_init = BondEncoder(embed_dim)
        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.in_dropout_rate = indropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.readout = readout
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)

        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(num_layers):
            topo = RephineLayer(
                n_features=hidden,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type=diagram,
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"
        final_dim = (
            hidden + len(self.ph_layers) * self.out_ph
            if self.ph_pooling_type == "cat"
            else hidden + self.out_ph
        )

        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_update_up_nn=None,
                    passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden,
                    bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden + self.out_ph, out_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, data_org,include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        xs, jump_xs = None, None
        res = {}

        # Embed and populate higher-levels
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        xs = list(self.init_conv(*params))

        # Apply dropout on the input features
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)

        data.set_xs(xs)
        ph_vectors = []
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
            data.set_xs(xs)
            ph_vectors += [self.ph_layers[c](xs[0],data_org)]

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = pool_complex(xs, data, self.max_dim, self.readout)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        x = self.lin2(torch.cat([x,ph_embedding],dim=1))

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__



