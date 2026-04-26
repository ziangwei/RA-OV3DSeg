from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
