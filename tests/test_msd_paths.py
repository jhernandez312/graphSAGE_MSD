import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from msd_data import resolve_msd_paths


def create_layout(root: Path, outer_directory: str) -> Path:
    data_root = root / outer_directory
    for relative in (
        "train/graph_in",
        "train/graph_out",
        "test/graph_in",
    ):
        (data_root / relative).mkdir(parents=True, exist_ok=True)
    return data_root


class MSDPathTests(unittest.TestCase):
    def test_resolves_legacy_layout_from_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = create_layout(root, "actual_data")
            with mock.patch.dict(os.environ, {"MSD_DATA_ROOT": str(root)}):
                paths = resolve_msd_paths()
            self.assertEqual(paths.data_root, data_root.resolve())
            self.assertEqual(paths.train_graph_in, data_root.resolve() / "train/graph_in")

    def test_resolves_kaggle_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = create_layout(root, "modified-swiss-dwellings-v2")
            paths = resolve_msd_paths(root)
            self.assertEqual(paths.data_root, data_root.resolve())

    def test_missing_environment_variable_is_clear(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MSD_DATA_ROOT is not set"):
                resolve_msd_paths()

    def test_missing_layout_is_clear(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "Expected directories"):
                resolve_msd_paths(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
