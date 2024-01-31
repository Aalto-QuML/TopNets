import torch

from torch import tensor

from datasets.datasets import get_data_perslay
from experiment import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reproducibility.utils import set_seeds
from train import train_perslay, evaluate_perslay


def run_perslay(args, device):
    set_seeds(args.seed)

    (
        train_data,
        train_features,
        train_labels,
        val_data,
        val_features,
        val_labels,
        test_data,
        test_features,
        test_labels,
    ) = get_data_perslay(args.dataset)

    train_losses = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    model, optimizer, loss_fn = get_model(args.dataset, train_features.shape[1])
    model = model.to(device)

    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        min_lr=1e-5,
        patience=args.lr_decay_patience,
        verbose=True,
    )

    for epoch in range(1, args.max_epochs + 1):
        # train
        train_loss = train_perslay(
            train_features,
            train_data,
            train_labels,
            model,
            loss_fn,
            optimizer,
            device,
            batch_size=args.batch_size,
        )

        # test
        val_loss, val_acc = evaluate_perslay(
            model, val_features, val_data, val_labels, loss_fn, device
        )

        # test
        test_loss, test_acc = evaluate_perslay(
            model, test_features, test_data, test_labels, loss_fn, device
        )

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)  # test losses

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)  # test losses

        train_losses.append(torch.tensor(train_loss).mean())  # train losses

        if (epoch - 1) % args.interval == 0:
            print(
                f"{epoch:3d}: Train Loss: {torch.tensor(train_loss).mean():.7f},"
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
