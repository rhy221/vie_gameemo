"""Configuration loading with YAML + experiment overrides.

Provides a typed `Config` namespace loaded from `config.yaml` with optional
experiment file overrides (deep merge) and CLI argument overrides (highest
precedence).

Example:
    >>> cfg = load_config("config.yaml", experiment="strategy_c_dual_path")
    >>> cfg.fusion.n_modalities
    4
    >>> cfg.training.perception.epochs
    30
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def load_config(
    config_path: str | Path,
    experiment: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Load configuration from YAML with optional experiment + CLI overrides.

    Args:
        config_path: Path to main config.yaml file.
        experiment: Name of experiment override file (without .yaml extension)
            in configs/experiments/. If None, no experiment overrides applied.
        cli_overrides: Dict of dot-path → value overrides from CLI args.
            E.g., {"training.perception.epochs": 5}.

    Returns:
        Nested SimpleNamespace with all config fields accessible via dot notation.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If required fields are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    if experiment is not None:
        exp_path = config_path.parent / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        base = deep_merge(base, override)

    if cli_overrides:
        for dot_path, value in cli_overrides.items():
            _apply_dot_path(base, dot_path, value)

    _validate_required(base)
    _resolve_paths(base, config_path.parent)

    return dict_to_namespace(base)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preserving nested structure.

    Args:
        base: Base dict to merge into (not mutated).
        override: Dict with keys to override.

    Returns:
        New dict with merged values.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert nested dict to SimpleNamespace for dot access.

    Args:
        d: Dictionary to convert.

    Returns:
        SimpleNamespace with nested SimpleNamespace children.
    """
    ns = SimpleNamespace()
    for key, val in d.items():
        if isinstance(val, dict):
            setattr(ns, key, dict_to_namespace(val))
        else:
            setattr(ns, key, val)
    return ns


def _apply_dot_path(d: dict, dot_path: str, value: Any) -> None:
    """Apply a dot-separated path override into dict d (mutates in place).

    Args:
        d: Dict to mutate.
        dot_path: Dot-separated key path, e.g. "training.perception.epochs".
        value: Value to set at that path.
    """
    keys = dot_path.split(".")
    node = d
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _validate_required(d: dict) -> None:
    """Check required top-level fields exist.

    Args:
        d: Config dict.

    Raises:
        ValueError: If required fields are missing.
    """
    required = ["seed", "paths", "training"]
    missing = [r for r in required if r not in d]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")


def _resolve_paths(d: dict, project_root: Path) -> None:
    """Resolve relative path strings in the 'paths' block to absolute Paths.

    Only converts values in the top-level 'paths' key and 'logging.file'.

    Args:
        d: Config dict (mutated in place).
        project_root: Root directory relative to which paths are resolved.
    """
    if "paths" in d and isinstance(d["paths"], dict):
        for key, val in d["paths"].items():
            if isinstance(val, str):
                p = Path(val)
                d["paths"][key] = str(project_root / p) if not p.is_absolute() else val

    if "logging" in d and isinstance(d["logging"], dict):
        log_file = d["logging"].get("file")
        if log_file and isinstance(log_file, str):
            p = Path(log_file)
            if not p.is_absolute():
                d["logging"]["file"] = str(project_root / p)
