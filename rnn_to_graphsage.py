import argparse
import pathlib
import sys

import torch

from visualize_inference import (
    build_predicted_graph_payload,
    load_graph_data,
    load_model,
    visualize_predictions,
)


GRAPH_RNN_DIR = pathlib.Path(__file__).resolve().parent / "ablation_GraphRNN" / "graph-rnn"
if str(GRAPH_RNN_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_RNN_DIR))

from generate import generate, load_model_from_config  # noqa: E402
from generate_for_graphSAGE import adj_to_edge_index, build_structural_features, build_zero_features  # noqa: E402
from torch_geometric.data import Data  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_RNN_MODEL_PATH = GRAPH_RNN_DIR / "configs" / "checkpoints" / "checkpoint-96000.pth"
DEFAULT_GRAPHSAGE_MODEL_PATH = REPO_ROOT / "results" / "best_model.pt"


def generate_graph_payload(rnn_model_path, num_nodes, feature_mode):
    node_model, edge_model, input_size, edge_gen_function, mode = load_model_from_config(rnn_model_path)
    adj_matrix = generate(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)
    adj_matrix = (adj_matrix > 0).astype("int64")

    if feature_mode == "structural":
        x = build_structural_features(adj_matrix)
    else:
        x = build_zero_features(adj_matrix)

    return {
        "data": Data(
            x=x,
            edge_index=adj_to_edge_index(adj_matrix),
            original_node_ids=torch.arange(adj_matrix.shape[0], dtype=torch.long),
        ),
        "adj_matrix": adj_matrix,
        "feature_mode": feature_mode,
        "source": "graph_rnn_generate",
    }


def resolve_path_argument(path_value, default_path):
    if path_value:
        return pathlib.Path(path_value)
    return pathlib.Path(default_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-pt", help="Path to a generated graph payload from the RNN side")
    parser.add_argument(
        "--rnn-model-path",
        help="Path to a GraphRNN checkpoint for direct generation. Defaults to the bundled checkpoint when omitted.",
    )
    parser.add_argument("--nodes", type=int, default=40, help="Requested node count when generating directly from GraphRNN")
    parser.add_argument("--feature-mode", choices=["structural", "zeros"], default="structural", help="How to build features for a directly generated graph")
    parser.add_argument("--generated-graph-out", help="Optional path to save the intermediate generated graph payload")
    parser.add_argument(
        "--graphsage-model-path",
        help="Path to the trained GraphSAGE checkpoint. Defaults to results/best_model.pt when omitted.",
    )
    parser.add_argument(
        "--labeled-graph-out",
        help="Path to save the final labeled graph payload. Defaults to results/labeled_generated_graph_<nodes>.pt.",
    )
    parser.add_argument("--save-path", help="Optional path to save the plotted labeled graph as an image")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive matplotlib display")
    args = parser.parse_args()

    graphsage_model_path = resolve_path_argument(args.graphsage_model_path, DEFAULT_GRAPHSAGE_MODEL_PATH)
    labeled_graph_out = resolve_path_argument(
        args.labeled_graph_out,
        REPO_ROOT / "results" / f"labeled_generated_graph_{args.nodes}.pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(str(graphsage_model_path), device)

    if args.graph_pt and args.rnn_model_path:
        raise ValueError("Provide at most one of --graph-pt or --rnn-model-path.")

    if args.graph_pt:
        graph_pt = args.graph_pt
    else:
        rnn_model_path = resolve_path_argument(args.rnn_model_path, DEFAULT_RNN_MODEL_PATH)
        generated_graph_out = resolve_path_argument(
            args.generated_graph_out,
            REPO_ROOT / "results" / f"generated_graph_{args.nodes}.pt",
        )
        generated_payload = generate_graph_payload(str(rnn_model_path), args.nodes, args.feature_mode)
        torch.save(generated_payload, generated_graph_out)
        print(f"Saved generated graph payload to {generated_graph_out}")
        graph_pt = str(generated_graph_out)

    load_args = argparse.Namespace(graph_pt=graph_pt, graph_in_dir=None, graph_id=None)
    data, graph_label, node_ids, source_payload = load_graph_data(load_args, checkpoint, device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1).cpu()

    payload = build_predicted_graph_payload(
        data=data,
        pred=pred,
        node_ids=node_ids,
        graph_label=graph_label,
        source_payload=source_payload,
    )
    torch.save(payload, labeled_graph_out)
    print(f"Saved labeled graph payload to {labeled_graph_out}")
    visualize_predictions(
        data=data,
        pred=pred,
        node_ids=node_ids,
        graph_label=graph_label,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
