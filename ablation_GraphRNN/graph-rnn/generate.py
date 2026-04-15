'''Code to use trained GraphRNN to generate a new graph.'''

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP


def m_seq_to_adj_mat(m_seq, m):
    n = m_seq.shape[0] + 1
    adj_mat = np.zeros((n, n))
    for i, prev_nodes in enumerate(m_seq):
        adj_mat[i+1, max(i+1-m, 0) : i+1] = list(reversed(prev_nodes[:i+1 - max(i+1-m, 0)]))
    return adj_mat


def sample_bernoulli(p):
    return int(np.random.random() < p)


def sample_softmax(x):
    num_classes = x.shape[0]
    c = np.random.choice(range(num_classes), p=torch.softmax(x, dim=0).numpy())
    one_hot = torch.zeros([num_classes])
    one_hot[c] = 1
    return one_hot


def adj_to_edge_index(adj):
    rows, cols = np.nonzero(adj)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def build_structural_features(adj):
    graph = nx.from_numpy_array(adj)
    num_nodes = adj.shape[0]
    degrees = np.array([degree for _, degree in graph.degree()], dtype=np.float32)

    if num_nodes > 1:
        norm_degree = degrees / (num_nodes - 1)
    else:
        norm_degree = np.zeros_like(degrees)

    clustering_dict = nx.clustering(graph)
    clustering = np.array([clustering_dict[i] for i in range(num_nodes)], dtype=np.float32)

    betweenness_dict = nx.betweenness_centrality(graph, normalized=True)
    betweenness = np.array([betweenness_dict[i] for i in range(num_nodes)], dtype=np.float32)

    pagerank_dict = nx.pagerank(graph) if num_nodes > 0 else {}
    pagerank = np.array([pagerank_dict.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)

    closeness_dict = nx.closeness_centrality(graph)
    closeness = np.array([closeness_dict[i] for i in range(num_nodes)], dtype=np.float32)

    avg_neighbor_deg = np.array(
        [np.mean([degrees[nb] for nb in graph.neighbors(i)]) if degrees[i] > 0 else 0.0 for i in range(num_nodes)],
        dtype=np.float32,
    )
    avg_neighbor_deg_norm = avg_neighbor_deg / (num_nodes - 1) if num_nodes > 1 else np.zeros_like(avg_neighbor_deg)

    is_leaf = (degrees == 1).astype(np.float32)

    x = np.stack(
        [norm_degree, clustering, betweenness, pagerank, is_leaf, closeness, avg_neighbor_deg_norm],
        axis=1,
    )
    return torch.tensor(x, dtype=torch.float)


def build_zero_features(adj):
    return torch.zeros((adj.shape[0], 7), dtype=torch.float)


def rnn_edge_gen(edge_rnn, h, num_edges, adj_vec_size, sample_fun, attempts=None):
    """
    Generates the edges coming from this node using RNN method.

    Arguments:
        edge_rnn: EdgeRNN model to use for generation
        h: Hidden state computed by the NodeLevelRNN
        num_edges: Number of edges to generate.
        adj_vec_size: Size of the padded adjacency vector to output.
            This should corresponds to the input size of the NodeLeveRNN.
        attempts: Not implemented!

    Returns: Adjacency vector of size [1, 1, adj_vec_size]
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_rnn.edge_feature_len], device=device)

    edge_rnn.set_first_layer_hidden(h)

    # SOS token
    x = torch.ones([1, 1, edge_rnn.edge_feature_len], device=device)

    for i in range(num_edges):
        # calculate probability of this edge existing
        prob = edge_rnn(x)
        # sample from this probability and assign value to adjacency vector
        # assign the value of this edge into the input of the next iteration
        x[0, 0, :] = sample_fun(prob[0, 0, :].detach())
        adj_vec[0, 0, i, :] = x[0, 0, :]

    return adj_vec


def mlp_edge_gen(edge_mlp, h, num_edges, adj_vec_size, sample_fun, attempts=1):
    """
    Generates the edges coming from this node using MLP method.

    Arguments:
        edge_mlp: EdgeMLP model to use for generation
        h: Hidden state computed by the NodeLevelRNN
        num_edges: Number of edges to generate.
        adj_vec_size: Size of the padded adjacency vector to output.
            This should correspond to the input size of the NodeLeveRNN.
        attempts: Number of retries that should be attempted if no
            edge is sampled.

    Returns: Adjacency vector of size [1, 1, adj_vec_size]
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_mlp.edge_feature_len], device=device)

    # calculate probabilities of all edges from this node existing
    edge_probs = edge_mlp(h)

    # update adj_vec with the sampled value from each edge probability
    for _ in range(attempts):
        for i in range(num_edges):
            adj_vec[0, 0, i, :] = sample_fun(edge_probs[0, 0, i, :].detach())
        # If we generated all zeros we will try again if there are
        # attempts left. If we have sampled at least one edge, we can go on.
        if (adj_vec[0, 0, :, :].data > 0).any():
            break

    return adj_vec


def generate(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode, edge_sample_attempts=1):
    """
    Generates a graph with the specified number of nodes using the given models.

    :param num_nodes: the number of nodes the outputted graph should have
    :param node_model: the torch model used to generate the nodes
    :param edge_model: the torch model used to generate the edges
    :param input_size: the number of inputs to be fed into model
    :param edge_gen_function: which function to use to generate edges (MLP or RNN)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    node_model.eval()
    edge_model.eval()

    sample_fun = sample_softmax if mode == 'directed-multiclass' else sample_bernoulli

    # initialize adjacency vector arbitrarily to all ones
    adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device)
    # data structure for storing final adjacency matrix
    list_adj_vecs = []

    node_model.reset_hidden()

    for i in range(1, num_nodes):
        # initialize graph state vector by running model on values from previous iteration
        # (or on the ones vector for first iteration)
        h = node_model(adj_vec)

        # run model to generate edges and save output
        adj_vec = edge_gen_function(edge_model, h, num_edges=min(i, input_size),
                                    adj_vec_size=input_size, sample_fun=sample_fun, attempts=edge_sample_attempts)
        if mode == 'undirected' or mode == 'directed-topsort':
            list_adj_vecs.append(adj_vec[0, 0, :min(num_nodes, input_size), 0].cpu().detach().int().numpy())
        else:
            # Turn one-hot into class index
            one_hot = adj_vec[0, 0, :min(num_nodes, input_size), :].cpu().detach().int().numpy()  # [num_nodes, 4]
            class_vec = np.zeros([min(num_nodes, input_size)])
            class_vec[:i] = one_hot[:i].nonzero()[1]  # [num_nodes]
            list_adj_vecs.append(class_vec)

        # EOS
        if np.array(list_adj_vecs[-1] == 0).all():
            break

    # Turn into full adjacency matrix
    adj = m_seq_to_adj_mat(np.array(list_adj_vecs), m=input_size)

    adj = adj + adj.T
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]

    adj = np.tril(adj)

    # Turn 0-3 classes into directed graph adjacency matrix
    if mode == 'directed-multiclass':
        adj = (adj % 2) + (adj // 2).T
    elif mode == 'undirected':
        adj = adj + adj.T

    # Remove isolated nodes as done in the GraphRNN paper.
    if mode == 'directed-multiclass' or mode == 'undirected':  # Don't do for topsort because it's not mirrored
        adj = adj[~np.all(adj == 0, axis=1)]
        adj = adj[:, ~np.all(adj == 0, axis=0)]

    return adj


def load_model_from_config(model_path):
    """Get model information from config and return models and model info."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    config = state['config']

    input_size = config['data']['m']

    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(input_size= config['data']['m'],
                                   output_size=config['model']['EdgeRNN']['hidden_size'],
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(device)
        edge_gen_function = rnn_edge_gen
    else:
        node_model = GraphLevelRNN(input_size= config['data']['m'],
                                   output_size=None,  # No output layer needed
                                   **config['model']['GraphRNN']).to(device)
        edge_model = EdgeLevelMLP(input_size=config['model']['GraphRNN']['hidden_size'],
                                  output_size=config['data']['m'],
                                  **config['model']['EdgeMLP']).to(device)
        edge_gen_function = mlp_edge_gen

    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])

    mode = config['model']['mode'] if 'mode' in config['model'] else 'undirected'

    return node_model, edge_model, input_size, edge_gen_function, mode


def draw_generated_graph(adj_matrix, file_name='test_new', directed=False):
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph if directed else nx.Graph)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=True, node_size=500)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{file_name}.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path of the model weights')
    parser.add_argument('-n', '--nodes', dest='num_nodes', required=False, default=10, type=int,
                        help='Number of nodes')
    parser.add_argument('--graphsage-out', default=None,
                        help='Optional .pt path to save a GraphSAGE-ready graph payload')
    parser.add_argument('--graphsage-feature-mode', choices=['structural', 'zeros'], default='structural',
                        help='Feature construction mode for the optional GraphSAGE payload')
    args = parser.parse_args()

    node_model, edge_model, input_size, edge_gen_function, mode = load_model_from_config(args.model_path)
    adj_matrix = generate(args.num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)

    draw_generated_graph(adj_matrix, 'test_new', directed=mode != 'undirected')

    if args.graphsage_out is not None:
        binary_adj = (adj_matrix > 0).astype(np.int64)
        if args.graphsage_feature_mode == 'structural':
            x = build_structural_features(binary_adj)
        else:
            x = build_zero_features(binary_adj)

        payload = {
            'data': Data(
                x=x,
                edge_index=adj_to_edge_index(binary_adj),
                original_node_ids=torch.arange(binary_adj.shape[0], dtype=torch.long),
            ),
            'adj_matrix': binary_adj,
            'feature_mode': args.graphsage_feature_mode,
            'source': 'generate.py',
        }
        torch.save(payload, args.graphsage_out)
        print(f"Saved GraphSAGE-ready graph payload to: {args.graphsage_out}")

    print("Adjacency shape:", adj_matrix.shape)
    print("Number of edges:", adj_matrix.sum())
