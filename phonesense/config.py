"""Load config.yaml and resolve paths relative to the repo root.

Usage:
    from phonesense.config import SEED, get_config, get_path

    rng = numpy.random.default_rng(SEED)
    phones_path = get_path("phones")       # -> absolute Path to data/processed/phones.jsonl

The repo root is the parent of the ``phonesense`` package (works with ``pip install -e .``).
Override it with the ``PHONESENSE_ROOT`` environment variable, and override the config file
with ``PHONESENSE_CONFIG`` (absolute path, or relative to the repo root).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    env = os.environ.get("PHONESENSE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    env = os.environ.get("PHONESENSE_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_absolute() else repo_root() / p
    return repo_root() / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the parsed config.yaml (cached). Treat the result as read-only."""
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg.get("seed"), int):
        raise ValueError("config.yaml must define an integer 'seed'")
    return cfg


def get_path(key: str) -> Path:
    """Absolute path for an entry under ``paths:`` (or ``dataset.raw_*``) in config.yaml."""
    cfg = get_config()
    paths = cfg.get("paths", {})
    if key in paths:
        rel = paths[key]
    elif key in cfg.get("dataset", {}) and key.startswith("raw_"):
        rel = cfg["dataset"][key]
    else:
        raise KeyError(f"unknown path key {key!r}; known: {sorted(paths)}")
    p = Path(rel)
    return p if p.is_absolute() else repo_root() / p


def get_seed() -> int:
    return int(get_config()["seed"])


SEED: int = get_seed()
