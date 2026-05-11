from pathlib import Path

SCRATCH_DIR_NAMES = {
    "analysis-empty-check",
    "smoke",
    "smoke-trending",
}


def find_scratch_artifacts(data_dir: Path) -> list[Path]:
    """Find disposable smoke-test artifacts under the data directory."""
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name in SCRATCH_DIR_NAMES
    )


def remove_paths(paths: list[Path]) -> int:
    """Remove files or directories and return the number of removed paths."""
    removed = 0
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            remove_tree(path)
        else:
            path.unlink()
        removed += 1
    return removed


def remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
    path.rmdir()
