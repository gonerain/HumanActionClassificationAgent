from __future__ import annotations

"""Configuration loader for backend service.

This module centralizes reading and writing of the backend configuration. The
configuration file is optional; sensible defaults are used when it is missing.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

# Configuration lives next to this file to keep paths predictable
CONFIG_FILE = Path(__file__).resolve().with_name("backend_config.json")


def load_config(path: Path | None = None, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load configuration from ``path``.

    If the file is absent or malformed, ``default`` is returned. ``default`` is
    copied to avoid mutating the caller's data.
    """
    cfg_path = path or CONFIG_FILE
    config: Dict[str, Any] = {
        "source": 0,
        "region": None,
    }
    if default:
        config.update(default)
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                if "region" in data and data["region"] is not None:
                    data["region"] = [tuple(pt) for pt in data["region"]]
                config.update(data)
        except Exception:
            pass
    return config


def save_config(config: Dict[str, Any], path: Path | None = None) -> None:
    """Persist ``config`` to ``path``.

    The region is stored as a list of ``[x, y]`` pairs to keep the file JSON
    serializable.
    """
    cfg_path = path or CONFIG_FILE
    to_save = dict(config)
    region = to_save.get("region")
    if region is not None:
        to_save["region"] = [list(pt) for pt in region]
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(to_save, fh)

