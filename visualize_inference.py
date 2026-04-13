import argparse

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TAGConv

from msd_dataset import MSDFloorplanGraphDataset, ROOM_NAMES
from msd_model import Model


def adapt_feature_dim(data, checkpoint):
    expected_dim = checkpoint["in_channels"]
    current_dim = data.x.shape[1]
    if current_dim == expected_dim:
        return data

    base_dim = checkpoint.get("base_feature_dim")
    semantic_dim = checkpoint.get("semantic_feature_dim", 0)
    if (
        base_dim is not None
        and semantic_dim > 0
        and current_dim == base_dim
        and expected_dim == base_dim + semantic_dim
    ):
        zeros = torch.zeros((data.num_nodes, semantic_dim), dtype=data.x.dtype, device=data.x.device)
        data.x = torch.cat([data.x, zeros], dim=1)
        return data

    raise ValueError(
        f"Feature dimension mismatch: graph has {current_dim} features, "
        f"but model expects {expected_dim}."
    )


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    models = {
        "mlp": Linear,
        "gcn": GCNConv,
        "gat": GATConv,
        "sage": SAGEConv,
        "tagcn": TAGConv,
    }

    model = Model(
        layer_type=models[checkpoint["model_name"]],
        n_hidden=checkpoint["n_hidden"],
        in_channels=checkpoint["in_channels"],
        out_channels=checkpoint["out_channels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_graph_data(args, checkpoint, device):
    if args.graph_pt:
        payload = torch.load(args.graph_pt, map_location=device, weights_only=False)
        data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        if not isinstance(data, Data):
            raise ValueError(f"{args.graph_pt} did not contain a PyG Data object.")
        if data.x is None:
            raise ValueError(f"{args.graph_pt} is missing node features.")
        data = adapt_feature_dim(data, checkpoint)
        graph_label = args.graph_pt
        node_ids = torch.arange(data.num_nodes)
        return data.to(device), graph_label, node_ids

    if args.graph_in_dir is None or args.graph_id is None:
        raise ValueError("Provide either --graph-pt, or both --graph-in-dir and --graph-id.")

    dataset = MSDFloorplanGraphDataset(
        graph_in_dir=args.graph_in_dir,
        graph_out_dir=None,
        keep_non_rooms=checkpoint.get("keep_non_rooms", False),
        num_zoning_types=checkpoint.get("num_zoning_types"),
        feature_mode=checkpoint.get("feature_mode", "structural"),
        semantic_features=checkpoint.get("semantic_features", "none"),
    )

    try:
        graph_index = dataset.graph_ids.index(args.graph_id)
    except ValueError as exc:
        raise ValueError(f"Graph id {args.graph_id} was not found in {args.graph_in_dir}") from exc

    data = adapt_feature_dim(dataset[graph_index], checkpoint)
    graph_label = args.graph_id
    node_ids = data.original_node_ids.clone()
    return data.to(device), graph_label, node_ids


def room_name(class_id):
    if 0 <= class_id < len(ROOM_NAMES):
        return ROOM_NAMES[class_id]
    return str(class_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained checkpoint")
    parser.add_argument("--graph-in-dir", help="Directory containing graph_in pickles")
    parser.add_argument("--graph-id", type=int, help="Numeric graph id to visualize")
    parser.add_argument("--graph-pt", help="Path to a .pt graph payload such as generated_graph.pt")
    parser.add_argument("--save-path", default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.model_path, device)
    data, graph_label, node_ids = load_graph_data(args, checkpoint, device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu()

    edge_index = data.edge_index.cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        graph.add_edge(u, v)

    labels = {
        i: f"{int(node_ids[i])}\n{room_name(int(pred[i]))}"
        for i in range(data.num_nodes)
    }
    colors = pred.tolist()

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=colors,
        cmap=plt.get_cmap("tab10"),
        node_size=1800,
        edgecolors="black",
    )
    nx.draw_networkx_edges(graph, pos, width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

    counts = {}
    for class_id in pred.tolist():
        name = room_name(int(class_id))
        counts[name] = counts.get(name, 0) + 1
    summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))

    plt.title(f"Predicted node labels for {graph_label}\n{summary}")
    plt.axis("off")
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()
