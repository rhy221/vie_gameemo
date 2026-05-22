"""I/O helpers: safe file read/write, path resolution, JSON/YAML round-trips.

All writes are atomic: data is written to a `.tmp` sibling then renamed,
so a crash mid-write never leaves a corrupt file.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> dict | list:
    """Read a JSON file with UTF-8 encoding.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON content (dict or list).

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """Atomically write data as JSON to path.

    Args:
        data: JSON-serializable data.
        path: Output path. Parent dirs created if missing.
        indent: JSON indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    tmp.replace(path)


def read_yaml(path: Path) -> dict:
    """Read a YAML file with UTF-8 encoding.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML content as dict.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict, path: Path) -> None:
    """Atomically write a dict as YAML.

    Args:
        data: Dict to serialize.
        path: Output path. Parent dirs created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
    tmp.replace(path)


def ensure_dir(path: Path) -> Path:
    """Create directory if missing, return it. Idempotent.

    Args:
        path: Directory path to create.

    Returns:
        The same path after creation.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path: Path) -> str:
    """SHA256 hash of file contents (for cache invalidation).

    Args:
        path: Path to file.

    Returns:
        Hex digest string.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
