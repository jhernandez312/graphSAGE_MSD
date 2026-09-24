import argparse
import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data

try:
    from .generate import load_model_from_config, generate
except ImportError:  # Preserve direct execution from the legacy directory.
    from generate import load_model_from_config, generate


STRUCTURAL_FEATURE_NAMES = [
    "degree",
    "normalized_degree",
    "clustering_coefficient",
    "betweenness_centrality",
    "pagerank",
    "is_leaf",
]


def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to PyG edge_index."""
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index


def build_structural_features(adj: np.ndarray) -> torch.Tensor:
    """
    Build 6 node features so the classifier can run.

    These are NOT the original House-GAN semantic/geometric features.
    They are graph-structural stand-ins:
      0. degree
      1. normalized degree
      2. clustering coefficient
      3. betweenness centrality
      4. pagerank
      5. is_leaf
    """
    G = nx.from_numpy_array(adj)

    n = adj.shape[0]
    degrees = np.array([d for _, d in G.degree()], dtype=np.float32)

    if n > 1:
        norm_degree = degrees / (n - 1)
    else:
        norm_degree = np.zeros_like(degrees)

    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(n)], dtype=np.float32)

    betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    betweenness = np.array([betweenness_dict[i] for i in range(n)], dtype=np.float32)

    pagerank_dict = nx.pagerank(G) if n > 0 else {}
    pagerank = np.array([pagerank_dict.get(i, 0.0) for i in range(n)], dtype=np.float32)

    is_leaf = (degrees == 1).astype(np.float32)

    x = np.stack(
        [degrees, norm_degree, clustering, betweenness, pagerank, is_leaf],
        axis=1
    )
    return torch.tensor(x, dtype=torch.float)


def build_zero_features(adj: np.ndarray) -> torch.Tensor:
    """Fallback: all-zero 6D node features."""
    n = adj.shape[0]
    return torch.zeros((n, 6), dtype=torch.float)


def adjacency_to_pyg_data(adj: np.ndarray, feature_mode: str = "structural") -> Data:
    """Build classifier input from a previously sampled adjacency matrix."""
    adj = (np.asarray(adj) > 0).astype(np.int64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square.")
    if feature_mode == "structural":
        x = build_structural_features(adj)
    elif feature_mode == "zeros":
        x = build_zero_features(adj)
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return Data(x=x, edge_index=adj_to_edge_index(adj))


def save_graph_payload(adj, output_path, feature_mode="structural", **extra):
    """Save a backward-compatible generated graph payload."""
    adj = (np.asarray(adj) > 0).astype(np.int64)
    data = adjacency_to_pyg_data(adj, feature_mode=feature_mode)
    payload = {
        "data": data,
        "adj_matrix": adj,
        "feature_mode": feature_mode,
        **extra,
    }
    torch.save(payload, output_path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to GraphRNN checkpoint")
    parser.add_argument("-n", "--nodes", dest="num_nodes", type=int, default=10,
                        help="Requested number of nodes to generate")
    parser.add_argument("--feature-mode", choices=["structural", "zeros"], default="structural",
                        help="How to create 6D node features for classifier input")
    parser.add_argument("--out", default="generated_graph.pt",
                        help="Output .pt file to save a PyG Data object")
    args = parser.parse_args()

    node_model, edge_model, input_size, edge_gen_function, mode = load_model_from_config(args.model_path)
    adj_matrix = generate(args.num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)

    # Make sure adjacency is binary/int for undirected graph use
    adj_matrix = (adj_matrix > 0).astype(np.int64)

    payload = save_graph_payload(adj_matrix, args.out, feature_mode=args.feature_mode)
    data = payload["data"]
    edge_index = data.edge_index
    x = data.x

    print(f"Saved generated graph to: {args.out}")
    print("Adjacency shape:", adj_matrix.shape)
    print("Number of nodes:", adj_matrix.shape[0])
    print("Number of directed edge entries:", edge_index.shape[1])
    print("Node feature shape:", tuple(x.shape))


if __name__ == "__main__":
    main()
