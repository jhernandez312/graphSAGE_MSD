"""Resolve external Modified Swiss Dwellings data directories."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DATA_ROOT_ENV = "MSD_DATA_ROOT"
REQUIRED_GRAPH_DIRECTORIES = (
    ("train", "graph_in"),
    ("train", "graph_out"),
    ("test", "graph_in"),
)


@dataclass(frozen=True)
class MSDPaths:
    configured_root: Path
    data_root: Path
    train_graph_in: Path
    train_graph_out: Path
    test_graph_in: Path
    test_graph_out: Optional[Path]


def _looks_like_data_root(path: Path) -> bool:
    return all((path / split / kind).is_dir() for split, kind in REQUIRED_GRAPH_DIRECTORIES)


def _candidate_roots(configured_root: Path) -> Iterable[Path]:
    yield configured_root / "actual_data"
    yield configured_root / "modified-swiss-dwellings-v2"
    yield configured_root
    if configured_root.is_dir():
        for child in sorted(configured_root.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                yield child


def resolve_msd_paths(root: Optional[Path] = None) -> MSDPaths:
    """Resolve the graph directories below ``root`` or ``MSD_DATA_ROOT``."""
    if root is None:
        configured = os.environ.get(DATA_ROOT_ENV)
        if not configured:
            raise RuntimeError(
                f"{DATA_ROOT_ENV} is not set. Point it to the external MSD dataset "
                "directory or pass an explicit root."
            )
        root = Path(configured)

    configured_root = Path(root).expanduser().resolve()
    if not configured_root.is_dir():
        raise FileNotFoundError(f"MSD dataset directory was not found: {configured_root}")

    seen = set()
    for candidate in _candidate_roots(configured_root):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_data_root(candidate):
            test_graph_out = candidate / "test" / "graph_out"
            return MSDPaths(
                configured_root=configured_root,
                data_root=candidate,
                train_graph_in=candidate / "train" / "graph_in",
                train_graph_out=candidate / "train" / "graph_out",
                test_graph_in=candidate / "test" / "graph_in",
                test_graph_out=test_graph_out if test_graph_out.is_dir() else None,
            )

    expected = ", ".join(
        f"{split}/{kind}" for split, kind in REQUIRED_GRAPH_DIRECTORIES
    )
    raise FileNotFoundError(
        f"No supported MSD layout was found below {configured_root}. "
        f"Expected directories: {expected}."
    )
