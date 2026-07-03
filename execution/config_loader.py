import os
from pathlib import Path

import yaml


def load_config(config_path: str, cli_overrides: dict | None = None) -> dict:

    config = _load_with_base(config_path)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            if value is not None:
                _set_nested(config, dotted_key, value)

    _validate(config, config_path)

    return config


def _load_with_base(config_path: str) -> dict:

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Working directory: {os.getcwd()}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if "_base" in config:
        base_name = config.pop("_base")
        base_path = config_path.parent / base_name
        base = _load_with_base(str(base_path))
        config = _deep_merge(base, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:

    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(config: dict, dotted_key: str, value) -> None:

    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _validate(config: dict, source: str) -> None:

    errors = []

    dataset_path = config.get("dataset", {}).get("path")
    if not dataset_path:
        errors.append("'dataset.path' is missing or empty.")

    threads = config.get("threading", {}).get("num_threads")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        errors.append(
            f"'threading.num_threads' must be a positive integer, got: {threads!r}"
        )

    epsilon = config.get("experiment", {}).get("epsilon")
    if epsilon is not None:
        if not isinstance(epsilon, list) or len(epsilon) == 0:
            errors.append("'experiment.epsilon' must be a non-empty list.")
        else:
            bad = [e for e in epsilon if not isinstance(e, (int, float)) or e <= 0]
            if bad:
                errors.append(
                    f"All epsilon values must be positive numbers. Bad values: {bad}"
                )

    repeats = config.get("experiment", {}).get("repeats")
    if repeats is not None and (not isinstance(repeats, int) or repeats < 1):
        errors.append(
            f"'experiment.repeats' must be a positive integer, got: {repeats!r}"
        )

    if errors:
        raise ValueError(
            f"Invalid configuration loaded from '{source}':\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
