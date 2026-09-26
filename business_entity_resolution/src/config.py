"""Shared configuration loading with cross-platform path resolution."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML configuration and resolve data/output paths.

    Environment overrides are intentionally supported for large local artifacts:

    - ``AMAZON_ML_DATA_ROOT``: extracted ``student_resource`` directory.
    - ``AMAZON_ML_OUTPUT_ROOT``: indexes, reports, features, and submissions.
    """
    path = _resolve_path(config_path or PROJECT_ROOT / "configs/config.yaml", PROJECT_ROOT)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    config = deepcopy(config)
    data_value = os.environ.get("AMAZON_ML_DATA_ROOT", config["data"]["root"])
    output_value = os.environ.get(
        "AMAZON_ML_OUTPUT_ROOT", config["output"]["directory"]
    )

    config["data"]["root"] = str(_resolve_path(data_value, PROJECT_ROOT))
    config["output"]["directory"] = str(_resolve_path(output_value, PROJECT_ROOT))
    config["_meta"] = {
        "project_root": str(PROJECT_ROOT),
        "config_path": str(path),
    }
    return config
