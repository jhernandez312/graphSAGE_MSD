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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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


def predict_node_classes(model, data):
    """Return CPU class ids for every node in a PyG graph."""
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).argmax(dim=1).cpu()


def pyg_data_to_networkx(data):
    """Convert PyG connectivity to an undirected NetworkX graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.detach().cpu().numpy()
    for index in range(edge_index.shape[1]):
        graph.add_edge(int(edge_index[0, index]), int(edge_index[1, index]))
    return graph


def graph_layout(graph, seed=42):
    """Create one reusable layout for raw and classified renderings."""
    return nx.spring_layout(graph, seed=seed)


def render_raw_graph(graph, positions, save_path, show=False):
    """Render an unlabeled topology using precomputed node positions."""
    fig, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="lightsteelblue",
        node_size=1800,
        edgecolors="black",
        ax=axis,
    )
    nx.draw_networkx_edges(graph, positions, width=1.5, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=9, ax=axis)
    axis.set_title("Generated graph topology")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def render_classified_graph(
    graph,
    predictions,
    node_ids,
    positions,
    save_path=None,
    graph_label="generated graph",
    show=False,
):
    """Render GraphSAGE predictions using caller-provided node positions."""
    predictions = predictions.detach().cpu()
    node_ids = node_ids.detach().cpu()
    labels = {
        index: f"{int(node_ids[index])}\n{room_name(int(predictions[index]))}"
        for index in range(len(predictions))
    }

    fig, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=predictions.tolist(),
        cmap=plt.get_cmap("tab10"),
        node_size=1800,
        edgecolors="black",
        ax=axis,
    )
    nx.draw_networkx_edges(graph, positions, width=1.5, ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=axis)

    counts = {}
    for class_id in predictions.tolist():
        name = room_name(int(class_id))
        counts[name] = counts.get(name, 0) + 1
    summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))

    axis.set_title(f"Predicted node labels for {graph_label}\n{summary}")
    axis.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


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

    pred = predict_node_classes(model, data)
    graph = pyg_data_to_networkx(data)
    pos = graph_layout(graph, seed=42)
    render_classified_graph(
        graph,
        pred,
        node_ids,
        pos,
        save_path=args.save_path,
        graph_label=graph_label,
        show=True,
    )


if __name__ == "__main__":
    main()
