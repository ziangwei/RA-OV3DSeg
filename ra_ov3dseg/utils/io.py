from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def save_npz(path: str | Path, **arrays: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path
