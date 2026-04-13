import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TAGConv
from tqdm import tqdm

from msd_dataset import MSDFloorplanGraphDataset
from msd_model import Model
from utils import accuracy


def predict_graphs(model, dataset, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1).cpu()
            predictions.append(
                {
                    "graph_id": int(data.graph_id.item()),
                    "node_ids": data.original_node_ids.cpu(),
                    "pred": pred,
                }
            )
    return predictions


def build_subset(items, indices):
    return [items[i] for i in indices]


def apply_semantic_dropout(data, base_feature_dim, semantic_feature_dim, dropout_prob):
    if semantic_feature_dim <= 0 or dropout_prob <= 0.0:
        return data
    if not hasattr(data, "batch"):
        return data

    drop_graph_mask = torch.rand(data.num_graphs, device=data.x.device) < dropout_prob
    if not torch.any(drop_graph_mask):
        return data

    data.x = data.x.clone()
    semantic_start = base_feature_dim
    semantic_end = base_feature_dim + semantic_feature_dim
    node_drop_mask = drop_graph_mask[data.batch]
    data.x[node_drop_mask, semantic_start:semantic_end] = 0.0
    return data


def save_training_plot(outpath, model_name, num_layers, lr, epochs, step, gamma, loss_ep, tr_acc_ep, val_acc_ep, te_acc_ep):
    epoch_axis = np.arange(1, len(loss_ep) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epoch_axis, loss_ep, label="Loss", color="tab:red", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_axis, tr_acc_ep, label="Train Acc", color="tab:blue", linewidth=2)
    if not np.all(np.isnan(val_acc_ep)):
        axes[1].plot(epoch_axis, val_acc_ep, label="Val Acc", color="tab:green", linewidth=2)
    if not np.all(np.isnan(te_acc_ep)):
        axes[1].plot(epoch_axis, te_acc_ep, label="Test Acc", color="tab:orange", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{model_name}{num_layers}: lr={lr}, epochs={epochs}, step={step}, gamma={gamma}")
    fig.tight_layout()

    plot_path = outpath / f"{model_name}{num_layers}_curves_{lr}_{epochs}_{step}_{gamma}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=["mlp", "gcn", "gat", "sage", "tagcn"], default="sage", help="Type of model")
    parser.add_argument("--hidden", type=int, default=2, help="Number of hidden/message passing layers")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.004, help="Learning rate")
    parser.add_argument("--step", type=int, default=10, help="Step size for exponential learning rate scheduling")
    parser.add_argument("--gamma", type=float, default=0.8, help="Decay rate for exponential learning rate scheduling")
    parser.add_argument("--bs", type=int, default=128, help="Batch size for training")
    parser.add_argument("--outpath", type=str, default="./results", help="Path to save results")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of labeled training graphs to use for validation")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/validation split")
    parser.add_argument("--train_graph_in_dir", type=str, required=True, help="Path to training graph_in directory")
    parser.add_argument("--train_graph_out_dir", type=str, required=True, help="Path to training graph_out directory")
    parser.add_argument("--test_graph_in_dir", type=str, default=None, help="Path to test graph_in directory")
    parser.add_argument("--test_graph_out_dir", type=str, default=None, help="Optional path to test graph_out directory")
    parser.add_argument("--keep_non_rooms", action="store_true", help="Keep non-room labels instead of filtering to room classes")
    parser.add_argument("--save_test_predictions", action="store_true", help="Save test predictions even when labels are available")
    parser.add_argument("--feature_mode", choices=["structural", "zoning"], default="structural", help="Node feature construction mode")
    parser.add_argument("--semantic_features", choices=["none", "room_shape"], default="none", help="Optional semantic feature columns to append")
    parser.add_argument("--semantic_dropout", type=float, default=0.0, help="Probability of zeroing semantic columns per training graph")
    args = parser.parse_args()

    models = {
        "mlp": Linear,
        "gcn": GCNConv,
        "gat": GATConv,
        "sage": SAGEConv,
        "tagcn": TAGConv,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    outpath = pathlib.Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    train_dataset = MSDFloorplanGraphDataset(
        graph_in_dir=args.train_graph_in_dir,
        graph_out_dir=args.train_graph_out_dir,
        keep_non_rooms=args.keep_non_rooms,
        feature_mode=args.feature_mode,
        semantic_features=args.semantic_features,
    )
    all_labeled = [train_dataset[i] for i in range(len(train_dataset))]

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in the range [0.0, 1.0).")

    num_graphs = len(all_labeled)
    if num_graphs < 2:
        raise ValueError("Need at least 2 labeled graphs to create a validation split.")

    val_size = int(round(num_graphs * args.val_ratio))
    if args.val_ratio > 0.0 and val_size == 0:
        val_size = 1
    if val_size >= num_graphs:
        val_size = num_graphs - 1

    split_generator = torch.Generator().manual_seed(args.split_seed)
    shuffled_indices = torch.randperm(num_graphs, generator=split_generator).tolist()
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train = build_subset(all_labeled, train_indices)
    val = build_subset(all_labeled, val_indices)

    print(f"labeled graphs: {len(all_labeled)}")
    print(f"train graphs: {len(train)}")
    print(f"validation graphs: {len(val)}")

    test_dataset = None
    test = []
    if args.test_graph_in_dir is not None:
        test_dataset = MSDFloorplanGraphDataset(
            graph_in_dir=args.test_graph_in_dir,
            graph_out_dir=args.test_graph_out_dir,
            keep_non_rooms=args.keep_non_rooms,
            num_zoning_types=train_dataset.num_zoning_types,
            feature_mode=args.feature_mode,
            semantic_features=args.semantic_features,
        )
        test = [test_dataset[i] for i in range(len(test_dataset))]
        print(f"test graphs: {len(test)}")
        print(f"test labels available: {test_dataset.has_labels}")

    in_channels = train[0].x.shape[1]
    base_feature_dim = train_dataset.base_feature_dim
    semantic_feature_dim = train_dataset.semantic_feature_dim
    num_classes = int(torch.cat([data.y for data in all_labeled]).max().item()) + 1

    model = Model(
        layer_type=models[args.model],
        n_hidden=args.hidden,
        in_channels=in_channels,
        out_channels=num_classes,
    )
    print(model)
    model = model.to(device)

    trainloader = DataLoader(train, batch_size=args.bs, shuffle=True)
    trainloader_full = DataLoader(train, batch_size=len(train))
    valloader = DataLoader(val, batch_size=len(val)) if len(val) > 0 else None
    testloader = None
    if len(test) > 0 and test_dataset.has_labels:
        testloader = DataLoader(test, batch_size=len(test))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    loss_ep = []
    tr_acc_ep = []
    val_acc_ep = []
    te_acc_ep = []
    best_val_acc = float("-inf")
    best_epoch = None
    best_checkpoint = None

    for epoch in range(args.epoch):
        model.train()
        loss = 0.0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epoch}", leave=False)
        for data in pbar:
            data = data.to(device)
            data = apply_semantic_dropout(data, base_feature_dim, semantic_feature_dim, args.semantic_dropout)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss_ = criterion(out, data.y)
            loss_.backward()
            optimizer.step()

            loss += loss_.item()
            pbar.set_postfix({"batch_loss": f"{loss_.item():.4f}"})

        exp_lr_scheduler.step()
        loss /= len(trainloader)

        tr_acc = accuracy(model, trainloader_full)
        if valloader is not None:
            val_acc = accuracy(model, valloader)
            val_msg = f"{val_acc:.6f}"
        else:
            val_acc = float("nan")
            val_msg = "N/A"

        if testloader is not None:
            te_acc = accuracy(model, testloader)
            test_msg = f"{te_acc:.6f}"
        else:
            te_acc = float("nan")
            test_msg = "N/A (no test labels)"

        loss_ep.append(loss)
        tr_acc_ep.append(tr_acc)
        val_acc_ep.append(val_acc)
        te_acc_ep.append(te_acc)

        if not np.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_checkpoint = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "model_name": args.model,
                "n_hidden": args.hidden,
                "in_channels": in_channels,
                "out_channels": num_classes,
                "num_zoning_types": train_dataset.num_zoning_types,
                "keep_non_rooms": args.keep_non_rooms,
                "feature_mode": args.feature_mode,
                "semantic_features": args.semantic_features,
                "base_feature_dim": base_feature_dim,
                "semantic_feature_dim": semantic_feature_dim,
                "semantic_dropout": args.semantic_dropout,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }

        print(
            f"Epoch [{epoch + 1}/{args.epoch}] "
            f"Loss: {loss:.10f}, Train Acc: {tr_acc:.6f}, Val Acc: {val_msg}, Test Acc: {test_msg}"
        )

    result = np.array([loss_ep, tr_acc_ep, val_acc_ep, te_acc_ep]).T
    np.savetxt(
        outpath / f"{type(model.layer1).__name__}{len(model.layer2) + 1}_loss_tracc_valacc_teacc_{args.lr}_{args.epoch}_{args.step}_{args.gamma}.txt",
        result,
    )
    save_training_plot(
        outpath=outpath,
        model_name=type(model.layer1).__name__,
        num_layers=len(model.layer2) + 1,
        lr=args.lr,
        epochs=args.epoch,
        step=args.step,
        gamma=args.gamma,
        loss_ep=loss_ep,
        tr_acc_ep=tr_acc_ep,
        val_acc_ep=val_acc_ep,
        te_acc_ep=te_acc_ep,
    )

    if best_checkpoint is not None:
        print(f"\nBest Validation Accuracy at Epoch {best_epoch}: {best_val_acc:.6f}\n")
    elif testloader is not None:
        max_idx = int(np.nanargmax(result[:, 3]))
        print(f"\nMax Test Accuracy at Epoch {max_idx + 1}: {result[max_idx]}\n")
    else:
        print("\nTest labels were not provided, so no test accuracy was computed.\n")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model,
        "n_hidden": args.hidden,
        "in_channels": in_channels,
        "out_channels": num_classes,
        "num_zoning_types": train_dataset.num_zoning_types,
        "keep_non_rooms": args.keep_non_rooms,
        "feature_mode": args.feature_mode,
        "semantic_features": args.semantic_features,
        "base_feature_dim": base_feature_dim,
        "semantic_feature_dim": semantic_feature_dim,
        "semantic_dropout": args.semantic_dropout,
    }
    torch.save(checkpoint, outpath / "model.pt")
    if best_checkpoint is not None:
        torch.save(best_checkpoint, outpath / "best_model.pt")
        print(f"Saved best validation checkpoint to {outpath / 'best_model.pt'}")

    if test_dataset is not None and (args.save_test_predictions or not test_dataset.has_labels):
        predictions = predict_graphs(model, test_dataset, device)
        torch.save(predictions, outpath / "test_predictions.pt")
        print(f"Saved test predictions to {outpath / 'test_predictions.pt'}")
