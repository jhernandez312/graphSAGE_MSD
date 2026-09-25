"""Run GraphRNN generation and GraphSAGE classification as one pipeline.

Edit CONFIG below, then run:

    uv run python run_pipeline.py
"""

import argparse
import importlib
import json
import random
import sys
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from msd_train import train_graphsage
from msd_data import resolve_msd_paths
from visualize_inference import (
    adapt_feature_dim,
    graph_layout,
    load_model,
    predict_node_classes,
    pyg_data_to_networkx,
    render_classified_graph,
    render_raw_graph,
    room_name,
)


REPO_ROOT = Path(__file__).resolve().parent
GRAPH_RNN_ROOT = REPO_ROOT / "ablation_GraphRNN" / "graph-rnn"
PIPELINE_SCHEMA_VERSION = 1


@dataclass
class DeviceConfig:
    mode: str = "auto"  # auto, cpu, or cuda
    cuda_index: int = 0


@dataclass
class SeedConfig:
    value: int = 42


@dataclass
class OutputConfig:
    root: Path = Path("runs")
    run_name: Optional[str] = None


@dataclass
class CheckpointConfig:
    graph_rnn: Path = Path(
        "ablation_GraphRNN/graph-rnn/configs/checkpoints/checkpoint-96000.pth"
    )
    graph_sage: Path = Path("results/best_model.pt")


@dataclass
class GraphRNNTrainingConfig:
    enabled: bool = False
    config_file: Path = Path("ablation_GraphRNN/graph-rnn/configs/config_swiss.yaml")
    restore_checkpoint: Optional[Path] = None
    graph_dir: Optional[Path] = None


@dataclass
class GraphSAGETrainingConfig:
    enabled: bool = False
    train_graph_in_dir: Optional[Path] = None
    train_graph_out_dir: Optional[Path] = None
    test_graph_in_dir: Optional[Path] = None
    test_graph_out_dir: Optional[Path] = None
    model: str = "sage"
    hidden: int = 2
    epochs: int = 100
    learning_rate: float = 0.004
    scheduler_step: int = 10
    scheduler_gamma: float = 0.8
    batch_size: int = 128
    validation_ratio: float = 0.1
    split_seed: int = 42
    keep_non_rooms: bool = False
    save_test_predictions: bool = False
    feature_mode: str = "structural"
    semantic_features: str = "room_shape"
    semantic_dropout: float = 0.5


@dataclass
class TrainingConfig:
    graph_rnn: GraphRNNTrainingConfig = field(default_factory=GraphRNNTrainingConfig)
    graph_sage: GraphSAGETrainingConfig = field(default_factory=GraphSAGETrainingConfig)


@dataclass
class GenerationConfig:
    num_graphs: int = 1
    num_nodes: int = 10
    feature_mode: str = "structural"
    edge_sample_attempts: int = 1


@dataclass
class PipelineConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class GraphArtifacts:
    directory: Path
    graph_payload: Path
    raw_graph: Path
    classified_graph: Path
    predictions: Path
    metadata: Path
    requested_nodes: int
    actual_nodes: int


@dataclass
class PipelineResult:
    run_directory: Path
    graph_rnn_checkpoint: Path
    graph_sage_checkpoint: Path
    graphs: List[GraphArtifacts]


# User-editable configuration. Training is deliberately disabled by default.
CONFIG = PipelineConfig()


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _resolve_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _require_file(path: Path, description: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} was not found: {resolved}")
    return resolved


def _create_run_directory(output: OutputConfig) -> Path:
    root = _resolve_path(output.root)
    root.mkdir(parents=True, exist_ok=True)
    if output.run_name is None:
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    else:
        name = output.run_name.strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("run_name must be one directory name without path separators.")
    run_directory = root / name
    run_directory.mkdir(parents=False, exist_ok=False)
    return run_directory.resolve()


def _resolve_device(config: DeviceConfig) -> torch.device:
    mode = config.mode.lower()
    if mode == "auto":
        return torch.device(
            f"cuda:{config.cuda_index}" if torch.cuda.is_available() else "cpu"
        )
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        if config.cuda_index < 0 or config.cuda_index >= torch.cuda.device_count():
            raise ValueError(
                f"cuda_index {config.cuda_index} is outside the available device range."
            )
        return torch.device(f"cuda:{config.cuda_index}")
    raise ValueError("device.mode must be 'auto', 'cpu', or 'cuda'.")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_legacy_module(module_name: str):
    """Import a bundled graph-rnn module without adding its path globally."""
    package_name = "_floorplan_graph_rnn"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = str(GRAPH_RNN_ROOT / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(GRAPH_RNN_ROOT)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{module_name}")


def _graphsage_args(config: GraphSAGETrainingConfig, output_dir: Path):
    if config.train_graph_in_dir is None and config.train_graph_out_dir is None:
        dataset_paths = resolve_msd_paths()
        train_graph_in_dir = dataset_paths.train_graph_in
        train_graph_out_dir = dataset_paths.train_graph_out
        test_graph_in_dir = config.test_graph_in_dir or dataset_paths.test_graph_in
        test_graph_out_dir = config.test_graph_out_dir or dataset_paths.test_graph_out
    elif config.train_graph_in_dir is None or config.train_graph_out_dir is None:
        raise ValueError(
            "GraphSAGE training requires both training paths, or neither when "
            "MSD_DATA_ROOT is set."
        )
    else:
        train_graph_in_dir = config.train_graph_in_dir
        train_graph_out_dir = config.train_graph_out_dir
        test_graph_in_dir = config.test_graph_in_dir
        test_graph_out_dir = config.test_graph_out_dir
    return argparse.Namespace(
        model=config.model,
        hidden=config.hidden,
        epoch=config.epochs,
        lr=config.learning_rate,
        step=config.scheduler_step,
        gamma=config.scheduler_gamma,
        bs=config.batch_size,
        outpath=str(output_dir),
        val_ratio=config.validation_ratio,
        split_seed=config.split_seed,
        train_graph_in_dir=str(_resolve_path(train_graph_in_dir)),
        train_graph_out_dir=str(_resolve_path(train_graph_out_dir)),
        test_graph_in_dir=(
            str(_resolve_path(test_graph_in_dir))
            if test_graph_in_dir is not None
            else None
        ),
        test_graph_out_dir=(
            str(_resolve_path(test_graph_out_dir))
            if test_graph_out_dir is not None
            else None
        ),
        keep_non_rooms=config.keep_non_rooms,
        save_test_predictions=config.save_test_predictions,
        feature_mode=config.feature_mode,
        semantic_features=config.semantic_features,
        semantic_dropout=config.semantic_dropout,
    )


def _select_checkpoints(
    config: PipelineConfig,
    run_directory: Path,
    device: torch.device,
    stages: Dict[str, Dict[str, Any]],
):
    graph_rnn_training = config.training.graph_rnn
    if graph_rnn_training.enabled:
        stages["graph_rnn_training"]["status"] = "running"
        stage_dir = run_directory / "training" / "graph_rnn"
        checkpoint_dir = stage_dir / "checkpoints"
        log_dir = stage_dir / "log"
        config_file = _require_file(graph_rnn_training.config_file, "GraphRNN config")
        restore_path = (
            _require_file(graph_rnn_training.restore_checkpoint, "GraphRNN restore checkpoint")
            if graph_rnn_training.restore_checkpoint is not None
            else None
        )
        if graph_rnn_training.graph_dir is None:
            graph_dir = resolve_msd_paths().train_graph_in
        else:
            graph_dir = _resolve_path(graph_rnn_training.graph_dir)
        trainer = _load_legacy_module("train")
        graph_rnn_checkpoint = Path(
            trainer.train_graph_rnn(
                config_file=config_file,
                restore_path=restore_path,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                device=device,
                graph_dir=graph_dir,
            )
        ).resolve()
        graph_rnn_checkpoint = _require_file(
            graph_rnn_checkpoint, "new GraphRNN checkpoint"
        )
        stages["graph_rnn_training"].update(
            status="completed", checkpoint=str(graph_rnn_checkpoint)
        )
    else:
        graph_rnn_checkpoint = _require_file(
            config.checkpoints.graph_rnn, "GraphRNN checkpoint"
        )
        stages["graph_rnn_training"].update(
            status="skipped", checkpoint=str(graph_rnn_checkpoint)
        )

    graph_sage_training = config.training.graph_sage
    if graph_sage_training.enabled:
        stages["graph_sage_training"]["status"] = "running"
        stage_dir = run_directory / "training" / "graph_sage"
        stage_dir.mkdir(parents=True, exist_ok=False)
        args = _graphsage_args(graph_sage_training, stage_dir)
        graph_sage_checkpoint = Path(
            train_graphsage(args, device=device, seed=config.seed.value)
        ).resolve()
        graph_sage_checkpoint = _require_file(
            graph_sage_checkpoint, "new GraphSAGE checkpoint"
        )
        stages["graph_sage_training"].update(
            status="completed", checkpoint=str(graph_sage_checkpoint)
        )
    else:
        graph_sage_checkpoint = _require_file(
            config.checkpoints.graph_sage, "GraphSAGE checkpoint"
        )
        stages["graph_sage_training"].update(
            status="skipped", checkpoint=str(graph_sage_checkpoint)
        )

    return graph_rnn_checkpoint, graph_sage_checkpoint


def _validate_generation_config(config: GenerationConfig) -> None:
    if config.num_graphs < 1:
        raise ValueError("generation.num_graphs must be at least 1.")
    if config.num_nodes < 2:
        raise ValueError("generation.num_nodes must be at least 2.")
    if config.feature_mode != "structural":
        raise ValueError("The consolidated pipeline supports structural features only.")
    if config.edge_sample_attempts < 1:
        raise ValueError("generation.edge_sample_attempts must be at least 1.")


def _validate_classifier_checkpoint(checkpoint: Dict[str, Any]) -> int:
    feature_mode = checkpoint.get("feature_mode", "structural")
    if feature_mode == "zoning":
        raise ValueError(
            "Zoning-based GraphSAGE checkpoints cannot classify generated graphs because "
            "GraphRNN does not generate zoning attributes."
        )
    if feature_mode != "structural":
        raise ValueError(f"Unsupported GraphSAGE feature mode: {feature_mode}")

    expected_dim = int(checkpoint["in_channels"])
    base_dim = checkpoint.get("base_feature_dim")
    semantic_dim = int(checkpoint.get("semantic_feature_dim", 0))
    if expected_dim == 6 and (base_dim is None or int(base_dim) == 6):
        return 0
    if (
        base_dim is not None
        and int(base_dim) == 6
        and semantic_dim > 0
        and expected_dim == 6 + semantic_dim
    ):
        return semantic_dim
    raise ValueError(
        "The GraphSAGE checkpoint feature dimensions are not explained by the six "
        "structural columns plus declared semantic columns."
    )


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = (np.asarray(adjacency) > 0).astype(np.int64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("GraphRNN returned a non-square adjacency matrix.")
    if adjacency.shape[0] == 0:
        raise RuntimeError("GraphRNN generated an empty graph.")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Only undirected GraphRNN output is supported by this pipeline.")
    if np.any(np.diag(adjacency)):
        raise ValueError("GraphRNN generated self-loops, which are not supported here.")
    return adjacency


def _predictions_document(predictions: torch.Tensor) -> Dict[str, Any]:
    rows = []
    for node_id, class_id in enumerate(predictions.tolist()):
        rows.append(
            {
                "node_id": node_id,
                "class_id": int(class_id),
                "room_name": room_name(int(class_id)),
            }
        )
    return {"schema_version": PIPELINE_SCHEMA_VERSION, "predictions": rows}


def _layout_document(positions) -> Dict[str, List[float]]:
    return {
        str(node): [float(coordinates[0]), float(coordinates[1])]
        for node, coordinates in positions.items()
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run enabled training stages, generate graphs, and write one artifact bundle."""
    _validate_generation_config(config.generation)
    run_directory = _create_run_directory(config.output)
    config_path = run_directory / "config.json"
    manifest_path = run_directory / "manifest.json"
    _write_json(config_path, asdict(config))

    stages = {
        "graph_rnn_training": {
            "enabled": config.training.graph_rnn.enabled,
            "status": "pending",
        },
        "graph_sage_training": {
            "enabled": config.training.graph_sage.enabled,
            "status": "pending",
        },
        "generation": {"status": "pending"},
    }
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_file": config_path.name,
        "stages": stages,
        "graphs": [],
    }
    _write_json(manifest_path, manifest)

    try:
        device = _resolve_device(config.device)
        manifest["device"] = str(device)

        graph_rnn_checkpoint, graph_sage_checkpoint = _select_checkpoints(
            config, run_directory, device, stages
        )

        graph_rnn_generate = _load_legacy_module("generate")
        graph_converter = _load_legacy_module("generate_for_graphSAGE")
        node_model, edge_model, input_size, edge_generator, mode = (
            graph_rnn_generate.load_model_from_config(
                graph_rnn_checkpoint, device=device
            )
        )
        if mode != "undirected":
            raise ValueError(
                f"GraphRNN mode '{mode}' is unsupported; this pipeline requires undirected mode."
            )

        classifier, classifier_checkpoint = load_model(
            graph_sage_checkpoint, device
        )
        semantic_padding = _validate_classifier_checkpoint(classifier_checkpoint)

        graph_results = []
        stages["generation"]["status"] = "running"
        for graph_index in range(config.generation.num_graphs):
            graph_seed = config.seed.value + graph_index
            _seed_everything(graph_seed)

            adjacency = graph_rnn_generate.generate(
                config.generation.num_nodes,
                node_model,
                edge_model,
                input_size,
                edge_generator,
                mode,
                edge_sample_attempts=config.generation.edge_sample_attempts,
            )
            adjacency = _validate_adjacency(adjacency)

            data = graph_converter.adjacency_to_pyg_data(
                adjacency, feature_mode=config.generation.feature_mode
            )
            data = adapt_feature_dim(data, classifier_checkpoint)
            classifier_data = data.clone().to(device)
            predictions = predict_node_classes(classifier, classifier_data)
            data = data.cpu()

            graph = pyg_data_to_networkx(data)
            positions = graph_layout(graph, seed=graph_seed)
            graph_directory = run_directory / f"graph_{graph_index + 1:03d}"
            graph_directory.mkdir(parents=False, exist_ok=False)

            raw_path = graph_directory / "raw_graph.png"
            payload_path = graph_directory / "graph.pt"
            classified_path = graph_directory / "classified_graph.png"
            predictions_path = graph_directory / "predictions.json"
            metadata_path = graph_directory / "metadata.json"

            render_raw_graph(graph, positions, raw_path, show=False)
            render_classified_graph(
                graph,
                predictions,
                torch.arange(data.num_nodes),
                positions,
                save_path=classified_path,
                graph_label=f"graph {graph_index + 1}",
                show=False,
            )
            predictions_document = _predictions_document(predictions)
            _write_json(predictions_path, predictions_document)

            metadata = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "seed": graph_seed,
                "graph_mode": mode,
                "requested_node_count": config.generation.num_nodes,
                "actual_node_count": int(adjacency.shape[0]),
                "edge_count": int(np.triu(adjacency, k=1).sum()),
                "directed_edge_entries": int(data.edge_index.shape[1]),
                "feature_mode": config.generation.feature_mode,
                "feature_schema": list(graph_converter.STRUCTURAL_FEATURE_NAMES),
                "model_input_features": int(data.x.shape[1]),
                "zero_filled_semantic_columns": semantic_padding,
                "graph_rnn_checkpoint": str(graph_rnn_checkpoint),
                "graph_sage_checkpoint": str(graph_sage_checkpoint),
                "layout": _layout_document(positions),
                "artifacts": {
                    "raw_graph": raw_path.name,
                    "graph_payload": payload_path.name,
                    "classified_graph": classified_path.name,
                    "predictions": predictions_path.name,
                    "metadata": metadata_path.name,
                },
            }
            _write_json(metadata_path, metadata)

            payload = {
                "data": data,
                "adj_matrix": adjacency,
                "feature_mode": config.generation.feature_mode,
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "provenance": metadata,
                "predictions": predictions_document["predictions"],
            }
            torch.save(payload, payload_path)

            graph_artifacts = GraphArtifacts(
                directory=graph_directory,
                graph_payload=payload_path,
                raw_graph=raw_path,
                classified_graph=classified_path,
                predictions=predictions_path,
                metadata=metadata_path,
                requested_nodes=config.generation.num_nodes,
                actual_nodes=int(adjacency.shape[0]),
            )
            graph_results.append(graph_artifacts)
            manifest["graphs"].append(
                {
                    "directory": graph_directory.name,
                    "seed": graph_seed,
                    "requested_node_count": config.generation.num_nodes,
                    "actual_node_count": int(adjacency.shape[0]),
                }
            )

        stages["generation"].update(
            status="completed", graph_count=len(graph_results)
        )
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            graph_rnn_checkpoint=str(graph_rnn_checkpoint),
            graph_sage_checkpoint=str(graph_sage_checkpoint),
        )
        _write_json(manifest_path, manifest)
        return PipelineResult(
            run_directory=run_directory,
            graph_rnn_checkpoint=graph_rnn_checkpoint,
            graph_sage_checkpoint=graph_sage_checkpoint,
            graphs=graph_results,
        )
    except Exception as error:
        manifest.update(
            status="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    result = run_pipeline(CONFIG)
    print(f"Pipeline completed: {result.run_directory}")
    for graph in result.graphs:
        print(
            f"  {graph.directory.name}: {graph.actual_nodes} nodes, "
            f"artifacts in {graph.directory}"
        )


if __name__ == "__main__":
    main()
