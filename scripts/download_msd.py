"""Download the pinned Modified Swiss Dwellings release outside the repository."""

import argparse
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msd_data import DATA_ROOT_ENV  # noqa: E402


GIB = 1024 ** 3
KAGGLE_DATASET = "caspervanengelenburg/modified-swiss-dwellings"
KAGGLE_URL = "https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings"
KAGGLE_VERSION = 6
KAGGLE_CLI_VERSION = "2.2.4"
KAGGLE_CLI_PYTHON = "3.11"
KAGGLE_REPORTED_BYTES = 17399563903


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent directory for {path}")
        candidate = candidate.parent
    return candidate


def _check_free_space(destination: Path, graphs_only: bool) -> None:
    existing_parent = _nearest_existing_parent(destination.parent)
    free_bytes = shutil.disk_usage(existing_parent).free
    required_bytes = KAGGLE_REPORTED_BYTES + 2 * GIB
    if not graphs_only:
        required_bytes += KAGGLE_REPORTED_BYTES
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space on {existing_parent}: "
            f"{free_bytes / GIB:.1f} GiB available, approximately "
            f"{required_bytes / GIB:.1f} GiB required."
        )


def kaggle_download_command(download_directory: Path) -> Sequence[str]:
    return (
        "uvx",
        "--python",
        KAGGLE_CLI_PYTHON,
        "--from",
        f"kaggle=={KAGGLE_CLI_VERSION}",
        "kaggle",
        "datasets",
        "download",
        f"{KAGGLE_DATASET}/{KAGGLE_VERSION}",
        "--path",
        str(download_directory),
    )


def _safe_member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(member.filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe path in Kaggle archive: {member.filename}")
    if ":" in path.parts[0]:
        raise ValueError(f"Unsafe drive path in Kaggle archive: {member.filename}")
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"Symbolic links are not accepted in the archive: {member.filename}")
    return path


def _graph_only_target(path: PurePosixPath) -> Optional[Path]:
    parts = path.parts
    for index in range(len(parts) - 2):
        split = parts[index]
        kind = parts[index + 1]
        filename = parts[index + 2]
        if (
            split in {"train", "test"}
            and kind in {"graph_in", "graph_out"}
            and index + 3 == len(parts)
            and filename.endswith(".pickle")
            and (split, kind) != ("test", "graph_out")
        ):
            return Path("actual_data") / split / kind / filename
    return None


def extract_dataset_archive(archive: Path, output: Path, graphs_only: bool) -> None:
    """Extract the archive without allowing paths to escape ``output``."""
    output.mkdir(parents=True, exist_ok=False)
    output_resolved = output.resolve()
    extracted_files = 0
    targets = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_path = _safe_member_path(member)
            if member.is_dir():
                continue
            relative = _graph_only_target(member_path) if graphs_only else Path(*member_path.parts)
            if relative is None:
                continue
            target = (output / relative).resolve()
            if output_resolved not in target.parents:
                raise ValueError(f"Unsafe extraction target: {member.filename}")
            if target in targets:
                raise ValueError(f"Duplicate extraction target: {member.filename}")
            targets.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            extracted_files += 1
    if extracted_files == 0:
        raise ValueError("The Kaggle archive did not contain any files to extract.")


def _write_receipt(destination: Path, graphs_only: bool) -> None:
    receipt = {
        "schema_version": 1,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_url": KAGGLE_URL,
        "kaggle_dataset": KAGGLE_DATASET,
        "kaggle_version": KAGGLE_VERSION,
        "kaggle_cli_version": KAGGLE_CLI_VERSION,
        "graphs_only": graphs_only,
    }
    (destination / "dataset_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def download_msd(
    destination: Path,
    *,
    graphs_only: bool = False,
    check_space: bool = True,
    runner=subprocess.run,
) -> Path:
    destination = Path(destination).expanduser().resolve()
    try:
        destination.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Dataset destination must be outside the Git repository: {destination}"
        )
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists and will not be overwritten: {destination}"
        )

    if check_space:
        _check_free_space(destination, graphs_only)
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.download-", dir=destination.parent)
    ).resolve()
    download_directory = staging / "download"
    download_directory.mkdir()
    ready_directory = staging / "ready"

    try:
        print("Downloading pinned Kaggle dataset release...")
        runner(kaggle_download_command(download_directory), check=True)

        archives = sorted(download_directory.rglob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(
                f"Expected one Kaggle archive in {download_directory}, found {len(archives)}."
            )

        print("Extracting dataset...")
        extract_dataset_archive(archives[0], ready_directory, graphs_only)
        _write_receipt(ready_directory, graphs_only)
        ready_directory.replace(destination)
        shutil.rmtree(staging, ignore_errors=True)
        return destination
    except Exception:
        print(
            f"Download was not installed. Staging files were retained at: {staging}",
            file=sys.stderr,
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Modified Swiss Dwellings outside Git."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help=f"External destination to use as {DATA_ROOT_ENV}.",
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Keep only graph files used by the current training code.",
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip the conservative free-space preflight.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    destination = download_msd(
        args.destination,
        graphs_only=args.graphs_only,
        check_space=not args.skip_space_check,
    )
    print(f"Dataset installed at: {destination}")
    print(f"PowerShell: $env:{DATA_ROOT_ENV} = \"{destination}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
