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


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_npz(path: str | Path, **arrays: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def load_text_lines(path: str | Path) -> list[str]:
    path = Path(path)
    lines = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if item:
                lines.append(item)
    return lines
