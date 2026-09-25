import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_msd import download_msd, extract_dataset_archive, kaggle_download_command


class DownloadMSDTests(unittest.TestCase):
    def test_command_pins_cli_python_and_dataset_version(self):
        command = kaggle_download_command(Path("download"))
        self.assertEqual(command[:5], ("uvx", "--python", "3.11", "--from", "kaggle==2.2.4"))
        self.assertIn("caspervanengelenburg/modified-swiss-dwellings/6", command)

    def test_graph_only_extraction_normalizes_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr(
                    "modified-swiss-dwellings-v2/train/graph_in/1.pickle", b"graph"
                )
                target.writestr(
                    "modified-swiss-dwellings-v2/train/struct_in/1.npy", b"large"
                )
            output = root / "output"
            extract_dataset_archive(archive, output, graphs_only=True)

            self.assertEqual(
                (output / "actual_data/train/graph_in/1.pickle").read_bytes(),
                b"graph",
            )
            self.assertFalse((output / "actual_data/train/struct_in/1.npy").exists())

    def test_extraction_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("../outside.txt", b"bad")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                extract_dataset_archive(archive, root / "output", graphs_only=False)

    def test_download_stages_and_installs_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "installed"

            def fake_runner(command, check):
                self.assertTrue(check)
                download_directory = Path(command[command.index("--path") + 1])
                with zipfile.ZipFile(download_directory / "dataset.zip", "w") as archive:
                    archive.writestr(
                        "modified-swiss-dwellings-v2/train/graph_in/1.pickle",
                        b"graph",
                    )

            result = download_msd(
                destination,
                graphs_only=True,
                check_space=False,
                runner=fake_runner,
            )

            self.assertEqual(result, destination.resolve())
            self.assertTrue(destination.is_dir())
            receipt = json.loads(
                (destination / "dataset_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["kaggle_version"], 6)
            self.assertTrue(receipt["graphs_only"])
            self.assertNotIn("validation", receipt)

    def test_download_never_overwrites_destination(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "installed"
            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                download_msd(destination, check_space=False)


if __name__ == "__main__":
    unittest.main()
