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
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_npz(path: str | Path, **arrays: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    tmp_path.replace(path)
    return path


def is_valid_npz(path: str | Path, required_keys: list[str] | tuple[str, ...] | None = None) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if required_keys is not None:
                missing_keys = [key for key in required_keys if key not in data.files]
                if missing_keys:
                    return False
        return True
    except Exception:
        return False


def load_text_lines(path: str | Path) -> list[str]:
    path = Path(path)
    lines = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if item:
                lines.append(item)
    return lines


def load_sample_indices(path: str | Path) -> list[int]:
    """Load a reproducible sample-index list from JSON or plain text.

    JSON files may be either a list of integers or a dict with one of the
    common manifest keys used by the project. Text files may use one index per
    line, whitespace, or comma-separated values; lines starting with ``#`` are
    ignored.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"sample index file is empty: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("sample_indices", "requested_sample_indices", "indices"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(
                    "sample index JSON must be a list or contain one of: "
                    "sample_indices, requested_sample_indices, indices"
                )
        if not isinstance(payload, list):
            raise ValueError(f"sample index JSON must resolve to a list, got {type(payload).__name__}")
        raw_items = payload
    else:
        raw_items = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            raw_items.extend(item for item in line.replace(",", " ").split() if item)

    indices = [int(item) for item in raw_items]
    if any(index < 0 for index in indices):
        raise ValueError(f"sample indices must be non-negative: {path}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"sample index file contains duplicates: {path}")
    return indices
