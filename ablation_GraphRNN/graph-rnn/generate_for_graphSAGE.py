import argparse
import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data

from generate import load_model_from_config, generate


def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to PyG edge_index."""
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index


def build_structural_features(adj: np.ndarray) -> torch.Tensor:
    """
    Build the same 7 structural node features used by `MSDFloorplanGraphDataset`.

    These are graph-structural stand-ins for generated graphs:
      0. normalized degree
      1. clustering coefficient
      2. betweenness centrality
      3. pagerank
      4. is_leaf
      5. closeness centrality
      6. normalized average neighbor degree
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

    closeness_dict = nx.closeness_centrality(G)
    closeness = np.array([closeness_dict[i] for i in range(n)], dtype=np.float32)

    avg_neighbor_deg = np.array(
        [np.mean([degrees[nb] for nb in G.neighbors(i)]) if degrees[i] > 0 else 0.0 for i in range(n)],
        dtype=np.float32,
    )
    avg_neighbor_deg_norm = avg_neighbor_deg / (n - 1) if n > 1 else np.zeros_like(avg_neighbor_deg)

    is_leaf = (degrees == 1).astype(np.float32)

    x = np.stack(
        [norm_degree, clustering, betweenness, pagerank, is_leaf, closeness, avg_neighbor_deg_norm],
        axis=1,
    )
    return torch.tensor(x, dtype=torch.float)


def build_zero_features(adj: np.ndarray) -> torch.Tensor:
    """Fallback: all-zero 7D node features."""
    n = adj.shape[0]
    return torch.zeros((n, 7), dtype=torch.float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to GraphRNN checkpoint")
    parser.add_argument("-n", "--nodes", dest="num_nodes", type=int, default=10,
                        help="Requested number of nodes to generate")
    parser.add_argument("--feature-mode", choices=["structural", "zeros"], default="structural",
                        help="How to create 7D node features for classifier input")
    parser.add_argument("--out", default="generated_graph.pt",
                        help="Output .pt file to save a PyG Data object")
    args = parser.parse_args()

    node_model, edge_model, input_size, edge_gen_function, mode = load_model_from_config(args.model_path)
    adj_matrix = generate(args.num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)

    # Make sure adjacency is binary/int for undirected graph use
    adj_matrix = (adj_matrix > 0).astype(np.int64)

    edge_index = adj_to_edge_index(adj_matrix)

    if args.feature_mode == "structural":
        x = build_structural_features(adj_matrix)
    else:
        x = build_zero_features(adj_matrix)

    data = Data(
        x=x,
        edge_index=edge_index,
        original_node_ids=torch.arange(adj_matrix.shape[0], dtype=torch.long),
    )

    # Save extra info too, since it helps debugging later
    payload = {
        "data": data,
        "adj_matrix": adj_matrix,
        "feature_mode": args.feature_mode,
    }

    torch.save(payload, args.out)

    print(f"Saved generated graph to: {args.out}")
    print("Adjacency shape:", adj_matrix.shape)
    print("Number of nodes:", adj_matrix.shape[0])
    print("Number of directed edge entries:", edge_index.shape[1])
    print("Node feature shape:", tuple(x.shape))


if __name__ == "__main__":
    main()
