"""
Utility helpers — config loading, logging setup, common transforms.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


# ── Paths ────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def _expand_env(value: str) -> str:
    """Replace ${VAR:-default} patterns with env values."""
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return pattern.sub(_replacer, value)


def _walk_expand(obj: Any) -> Any:
    """Recursively expand env vars in a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load and return the YAML config with env-var expansion."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _walk_expand(raw)


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>",
    )


def normalize_drug_name(name: str) -> str:
    """Lowercase, strip common salt suffixes and extra whitespace."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # Remove common salt forms
    for suffix in [
        " hydrochloride", " hcl", " sodium", " potassium",
        " mesylate", " maleate", " tartrate", " sulfate",
        " acetate", " succinate", " fumarate",
    ]:
        name = name.replace(suffix, "")
    return re.sub(r"\s+", " ", name).strip()


def normalize_event_term(term: str) -> str:
    """Lowercase and normalize adverse event terms."""
    if not isinstance(term, str):
        return ""
    return re.sub(r"\s+", " ", term.lower().strip())


# Pre-load config for convenience
try:
    CONFIG: Dict[str, Any] = load_config()
except FileNotFoundError:
    CONFIG = {}
