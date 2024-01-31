import torch
from ogb.graphproppred import Evaluator
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from datasets.datasets import get_data
from models import models
from reproducibility.utils import set_seeds
from train import evaluate, train


def run_main(args, device):

    set_seeds(args.seed)

    # Get Data
    train_data, val_data, test_data, stats = get_data(args.dataset, perslay_feats=args.gnn=='linear')
    if args.gnn == 'linear':
        args.n_graph_features = train_data.graph_features.shape[1]

    args.num_node_features = stats["num_features"]
    args.num_classes = stats["num_classes"]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    if args.dataset == "ZINC":
        loss_fn = torch.nn.L1Loss(reduction='mean')

    evaluator = None
    if args.dataset == "ogbg-molhiv":
        evaluator = Evaluator(args.dataset)

    train_losses = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []

    model = models.get_model(args).to(device)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        min_lr=1e-6,
        patience=args.lr_decay_patience,
        verbose=True,
    )

    for epoch in range(1, args.max_epochs + 1):
        train_loss, val_loss, val_acc, test_loss, test_acc = train_eval(
            model,
            train_loader,
            val_loader,
            test_loader,
            loss_fn,
            optimizer,
            evaluator,
            device
        )

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)  # test losses

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)  # test losses

        train_losses.append(torch.tensor(train_loss).mean())  # train losses

        if (epoch - 1) % args.interval == 0:
            print(
                f"{epoch:3d}: Train Loss: {torch.tensor(train_loss).mean():.3f},"
                f" Val Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, "
                f"Test Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}"
            )

        scheduler.step(val_acc)

        if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2 - epochs_no_improve]:
            epochs_no_improve = epochs_no_improve + 1

        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.early_stop_patience:
            print("Early stopping!")
            break

    results = {
        "train_losses": tensor(train_losses),
        "test_accuracies": tensor(test_accuracies),
        "test_losses": tensor(test_losses),
        "val_accuracies": tensor(val_accuracies),
        "val_losses": tensor(val_losses),
    }

    return results



def train_eval(model, train_loader, val_loader, test_loader, loss_fn, optimizer, evaluator, device):
    train_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, evaluator)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, evaluator)
    return train_loss, val_loss, val_acc, test_loss, test_acc
