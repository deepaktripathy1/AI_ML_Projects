"""Helper file for path serialization."""

from pathlib import Path


def to_serializable_path(path: Path | str) -> str:
    """Convert a path or string into a cross-platform JSON safe string."""

    if not isinstance(path, Path):
        path = Path(path)

    return path.as_posix()


def from_serializable_path(path_str: str) -> Path:
    """Convert a serialized POSIX-style path back to a Path object."""

    return Path(path_str.replace("\\", "/"))
