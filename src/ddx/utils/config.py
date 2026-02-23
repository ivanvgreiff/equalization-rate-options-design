"""Configuration loading from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config file by name (without extension)."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_analysis_config() -> dict:
    return load_config("analysis")


def load_contracts_config() -> dict:
    return load_config("contracts")


def load_events_config() -> dict:
    return load_config("events")
