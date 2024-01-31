import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.gnn import GCN
from layers.rephine_layer import RephineLayerToy


def run_toy(args, dataset, device):

    if args.model == "gcn":
        model = GCN(1).to(device)
    else:
        model = RephineLayerToy(
            n_features=1,
            n_filtrations=args.n_filtrations,
            filtration_hidden=8,
            out_dim=16,
            diagram_type=args.model,
            dim1=args.dim1,
            sig_filtrations=True,
            reduce_tuples=args.reduce_tuples
        ).to(device)

    print(f"Model: {args.model}")
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # train and test
    def test(model):
        model.eval()
        with torch.no_grad():
            correct = 0
            loader = DataLoader(dataset, batch_size=len(dataset))
            for data in loader:
                out, h = model(data)
                out = out.squeeze()
                unique_embedding = torch.unique(h.round(decimals=8), dim=0).shape[0]
                out[out > 0.0] = 1.0
                out[out <= 0.0] = 0.0
                correct = (out == data.y.float()).sum()
        acc = int(correct) / len(dataset)
        return acc, unique_embedding / len(dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr / 5, gamma=0.5
    )

    results = dict()
    losses = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()
    accuracies, expressivities = [], []
    for epoch in range(args.n_epochs):
        cumulative = 0
        model.train()
        for data in loader:
            optimizer.zero_grad()
            out, _ = model(data)
            loss = loss_fn(out, data.y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            cumulative += loss.clone().detach()
        losses.append(cumulative)
        accuracy, exp = test(model)
        accuracies.append(accuracy)
        expressivities.append(exp)
        scheduler.step()
        print(
            f"Epoch: {epoch}, loss: {cumulative:.8f}, accuracy: {accuracy:.8f}, expressivity: {exp:.4f}"
        )
    results["losses"] = losses
    results["accuracies"] = accuracies
    results["expressivities"] = expressivities
    return results, model
