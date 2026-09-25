import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Data

import run_pipeline as pipeline


class DummyClassifier:
    def eval(self):
        return self

    def __call__(self, x, edge_index):
        logits = torch.zeros((x.shape[0], 9), device=x.device)
        logits[torch.arange(x.shape[0]), torch.arange(x.shape[0]) % 9] = 1.0
        return logits


def fake_converter_module():
    def adjacency_to_pyg_data(adjacency, feature_mode="structural"):
        rows, columns = np.nonzero(adjacency)
        edge_index = torch.tensor(np.vstack([rows, columns]), dtype=torch.long)
        x = torch.ones((adjacency.shape[0], 6), dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    return types.SimpleNamespace(
        adjacency_to_pyg_data=adjacency_to_pyg_data,
        STRUCTURAL_FEATURE_NAMES=[
            "degree",
            "normalized_degree",
            "clustering_coefficient",
            "betweenness_centrality",
            "pagerank",
            "is_leaf",
        ],
    )


def fake_generator_module(mode="undirected"):
    adjacency = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int64,
    )
    return types.SimpleNamespace(
        load_model_from_config=lambda checkpoint, device: (
            object(),
            object(),
            3,
            object(),
            mode,
        ),
        generate=lambda *args, **kwargs: adjacency.copy(),
    )


def write_fake_image(*args, **kwargs):
    save_path = kwargs.get("save_path")
    if save_path is None:
        save_path = args[2]
    Path(save_path).write_bytes(b"fake png")


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_consistent_artifact_bundle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_rnn_checkpoint = root / "graph_rnn.pth"
            graph_sage_checkpoint = root / "graph_sage.pt"
            graph_rnn_checkpoint.touch()
            graph_sage_checkpoint.touch()
            config = pipeline.PipelineConfig(
                output=pipeline.OutputConfig(root=root, run_name="complete"),
                generation=pipeline.GenerationConfig(num_graphs=2, num_nodes=3),
                device=pipeline.DeviceConfig(mode="cpu"),
            )

            def select_checkpoints(config, run_directory, device, stages):
                stages["graph_rnn_training"].update(status="skipped")
                stages["graph_sage_training"].update(status="skipped")
                return graph_rnn_checkpoint, graph_sage_checkpoint

            def load_legacy_module(name):
                if name == "generate":
                    return fake_generator_module()
                if name == "generate_for_graphSAGE":
                    return fake_converter_module()
                raise AssertionError(name)

            checkpoint = {
                "feature_mode": "structural",
                "in_channels": 13,
                "base_feature_dim": 6,
                "semantic_feature_dim": 7,
            }
            with mock.patch.object(pipeline, "_select_checkpoints", select_checkpoints), mock.patch.object(
                pipeline, "_load_legacy_module", load_legacy_module
            ), mock.patch.object(
                pipeline, "load_model", return_value=(DummyClassifier(), checkpoint)
            ), mock.patch.object(
                pipeline, "render_raw_graph", side_effect=write_fake_image
            ), mock.patch.object(
                pipeline, "render_classified_graph", side_effect=write_fake_image
            ):
                result = pipeline.run_pipeline(config)

            self.assertEqual(len(result.graphs), 2)
            self.assertEqual(result.graphs[1].directory.name, "graph_002")
            graph = result.graphs[0]
            payload = torch.load(graph.graph_payload, weights_only=False)
            metadata = json.loads(graph.metadata.read_text(encoding="utf-8"))
            predictions = json.loads(graph.predictions.read_text(encoding="utf-8"))
            manifest = json.loads(
                (result.run_directory / "manifest.json").read_text(encoding="utf-8")
            )
            edge_pairs = set(map(tuple, payload["data"].edge_index.t().tolist()))
            adjacency_pairs = set(map(tuple, np.argwhere(payload["adj_matrix"]).tolist()))

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(edge_pairs, adjacency_pairs)
            self.assertEqual(tuple(payload["data"].x.shape), (3, 13))
            self.assertEqual(metadata["zero_filled_semantic_columns"], 7)
            self.assertEqual(len(predictions["predictions"]), 3)
            self.assertTrue(graph.raw_graph.is_file())
            self.assertTrue(graph.classified_graph.is_file())

    def test_both_training_switches_use_new_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_rnn_config = root / "graph_rnn.yaml"
            graph_rnn_config.write_text("model: {}\n", encoding="utf-8")
            data_directory = root / "data"
            data_directory.mkdir()
            run_directory = root / "run"
            run_directory.mkdir()
            config = pipeline.PipelineConfig(
                training=pipeline.TrainingConfig(
                    graph_rnn=pipeline.GraphRNNTrainingConfig(
                        enabled=True,
                        config_file=graph_rnn_config,
                        graph_dir=data_directory,
                    ),
                    graph_sage=pipeline.GraphSAGETrainingConfig(
                        enabled=True,
                        train_graph_in_dir=data_directory,
                        train_graph_out_dir=data_directory,
                    ),
                )
            )
            stages = {
                "graph_rnn_training": {"enabled": True, "status": "pending"},
                "graph_sage_training": {"enabled": True, "status": "pending"},
            }

            class Trainer:
                @staticmethod
                def train_graph_rnn(**kwargs):
                    checkpoint = Path(kwargs["checkpoint_dir"]) / "checkpoint-1.pth"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.touch()
                    return checkpoint

            def train_graphsage(args, device, seed):
                checkpoint = Path(args.outpath) / "best_model.pt"
                checkpoint.touch()
                return checkpoint

            with mock.patch.object(pipeline, "_load_legacy_module", return_value=Trainer), mock.patch.object(
                pipeline, "train_graphsage", side_effect=train_graphsage
            ):
                graph_rnn, graph_sage = pipeline._select_checkpoints(
                    config, run_directory, torch.device("cpu"), stages
                )

            self.assertTrue(graph_rnn.is_file())
            self.assertTrue(graph_sage.is_file())
            self.assertEqual(stages["graph_rnn_training"]["status"], "completed")
            self.assertEqual(stages["graph_sage_training"]["status"], "completed")

    def test_failure_is_recorded_and_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = pipeline.PipelineConfig(
                output=pipeline.OutputConfig(
                    root=Path(temporary_directory), run_name="failed"
                ),
                generation=pipeline.GenerationConfig(num_nodes=3),
                device=pipeline.DeviceConfig(mode="cpu"),
            )
            with mock.patch.object(
                pipeline, "_select_checkpoints", side_effect=RuntimeError("training failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    pipeline.run_pipeline(config)

            manifest = json.loads(
                (Path(temporary_directory) / "failed" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["type"], "RuntimeError")
            self.assertFalse((Path(temporary_directory) / "failed" / "graph_001").exists())

    def test_directed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_rnn_checkpoint = root / "graph_rnn.pth"
            graph_sage_checkpoint = root / "graph_sage.pt"
            graph_rnn_checkpoint.touch()
            graph_sage_checkpoint.touch()
            config = pipeline.PipelineConfig(
                output=pipeline.OutputConfig(root=root, run_name="directed"),
                generation=pipeline.GenerationConfig(num_nodes=3),
                device=pipeline.DeviceConfig(mode="cpu"),
            )
            with mock.patch.object(
                pipeline,
                "_select_checkpoints",
                return_value=(graph_rnn_checkpoint, graph_sage_checkpoint),
            ), mock.patch.object(
                pipeline, "_load_legacy_module", return_value=fake_generator_module("directed-topsort")
            ):
                with self.assertRaisesRegex(ValueError, "requires undirected mode"):
                    pipeline.run_pipeline(config)

    def test_zoning_and_unexplained_feature_checkpoints_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Zoning-based"):
            pipeline._validate_classifier_checkpoint(
                {"feature_mode": "zoning", "in_channels": 6}
            )
        with self.assertRaisesRegex(ValueError, "not explained"):
            pipeline._validate_classifier_checkpoint(
                {"feature_mode": "structural", "in_channels": 8}
            )

    def test_invalid_generation_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_graphs"):
            pipeline._validate_generation_config(
                pipeline.GenerationConfig(num_graphs=0)
            )
        with self.assertRaisesRegex(ValueError, "num_nodes"):
            pipeline._validate_generation_config(
                pipeline.GenerationConfig(num_nodes=1)
            )

    def test_existing_run_name_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pipeline.OutputConfig(
                root=Path(temporary_directory), run_name="same-name"
            )
            pipeline._create_run_directory(output)
            with self.assertRaises(FileExistsError):
                pipeline._create_run_directory(output)

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.pt"
            with self.assertRaisesRegex(FileNotFoundError, "missing.pt"):
                pipeline._require_file(missing, "checkpoint")


if __name__ == "__main__":
    unittest.main()
