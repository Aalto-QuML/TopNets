import torch
import torch.nn as nn

from layers.rephine_layer import RephineLayer


class RephineLinearClassifier(RephineLayer):
    def __init__(
        self,
        n_classes,
        num_node_features,
        n_graph_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type="rephine",
        dim1=True,
        sig_filtrations=True,
    ):
        super().__init__(
            n_features=num_node_features,
            n_filtrations=n_filtrations,
            filtration_hidden=filtration_hidden,
            out_dim=out_dim,
            diagram_type=diagram_type,
            dim1=dim1,
            sig_filtrations=sig_filtrations,
        )
        self.normalizer = nn.BatchNorm1d(out_dim + n_graph_features)
        self.classifier = nn.Linear(out_dim + n_graph_features, n_classes)

    def forward(self, data):
        x_rephine = super(RephineLinearClassifier, self).forward(data.x, data)
        x = torch.cat([x_rephine, data.graph_features], dim=-1)
        x = self.normalizer(x)
        x = self.classifier(x)
        return x
