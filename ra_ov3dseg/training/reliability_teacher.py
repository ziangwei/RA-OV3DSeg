from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _scalar_to_str(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return None


def _scalar_to_int(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        try:
            return int(array.item())
        except (TypeError, ValueError):
            return None
    if array.size == 1:
        try:
            return int(array.reshape(-1)[0])
        except (TypeError, ValueError):
            return None
    return None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _rank_calibrate(raw_weight: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    raw_weight = np.nan_to_num(raw_weight.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    calibrated = np.zeros(raw_weight.shape, dtype=np.float32)
    valid = valid_mask.astype(bool) & np.isfinite(raw_weight) & (raw_weight > 0)
    if not np.any(valid):
        return calibrated
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(raw_weight[valid_indices], kind="mergesort")]
    if order.shape[0] == 1:
        calibrated[order[0]] = 1.0
        return calibrated
    calibrated[order] = np.linspace(0.0, 1.0, num=order.shape[0], dtype=np.float32)
    return calibrated


def _coord_fingerprint(coord: np.ndarray) -> str:
    coord = np.asarray(coord, dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] < 3:
        raise ValueError(f"coord must have shape [N, >=3], got {coord.shape}")
    num_points = coord.shape[0]
    if num_points == 0:
        return "0:empty"
    indices = np.linspace(0, num_points - 1, num=min(num_points, 64), dtype=np.int64)
    sample = np.round(coord[indices, :3], 3)
    digest = hashlib.sha1(sample.tobytes()).hexdigest()
    return f"{num_points}:{digest}"


class ReliabilityTeacherCache:
    """Load Stage 4 reliability weights and Stage 3 dense teacher logits.

    The cache is keyed by the project sample index file naming convention
    (`sample_XXXX_*`) and, when available, by nuScenes sample token. A coordinate
    fingerprint fallback exists only for diagnostics when Pointcept does not
    expose an index or token in its data dict.
    """

    def __init__(
        self,
        reliability_dir: str | Path,
        dense_point_dir: str | Path,
        sample_index_manifest: str | Path | None = None,
        teacher_class_start: int = 0,
        teacher_num_classes: int = 16,
        component_mode: str = "full",
        component_calibration: str = "rank",
        strict: bool = True,
        max_coord_error: float = 0.05,
    ) -> None:
        self.reliability_dir = Path(reliability_dir).expanduser().resolve()
        self.dense_point_dir = Path(dense_point_dir).expanduser().resolve()
        self.teacher_class_start = int(teacher_class_start)
        self.teacher_num_classes = int(teacher_num_classes)
        self.component_mode = component_mode
        self.component_calibration = component_calibration
        self.strict = bool(strict)
        self.max_coord_error = float(max_coord_error)
        self.token_to_sample_idx: dict[str, int] = {}
        self.timestamp_to_sample_idx: dict[int, int] = {}
        self._token_index_built = False
        self._fingerprint_to_sample_idx: dict[str, int] | None = None

        if not self.reliability_dir.exists():
            raise FileNotFoundError(f"reliability_dir does not exist: {self.reliability_dir}")
        if not self.dense_point_dir.exists():
            raise FileNotFoundError(f"dense_point_dir does not exist: {self.dense_point_dir}")
        if self.component_mode not in {"full", "no_distance", "no_geometric", "no_semantic", "uniform"}:
            raise ValueError(f"Unsupported reliability component_mode: {self.component_mode}")
        if self.component_calibration not in {"rank", "raw"}:
            raise ValueError(f"Unsupported component_calibration: {self.component_calibration}")
        if sample_index_manifest:
            self._load_manifest(Path(sample_index_manifest).expanduser().resolve())

    def _load_manifest(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"sample_index_manifest not found: {path}")
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("samples", []):
                token = item.get("sample_token")
                sample_idx = item.get("sample_idx")
                if token is not None and sample_idx is not None:
                    self.token_to_sample_idx[str(token)] = int(sample_idx)
                timestamp = item.get("timestamp")
                if timestamp is not None and sample_idx is not None:
                    self.timestamp_to_sample_idx[int(timestamp)] = int(sample_idx)
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if re.fullmatch(r"\d+", line):
                continue
            raise ValueError(
                "Text sample_index_manifest files can only provide integer sample indices; "
                "use JSON with samples[].sample_token for token lookup."
            )

    def _build_token_index(self) -> None:
        if self._token_index_built:
            return
        for path in sorted(self.reliability_dir.glob("sample_*_reliability.npz")):
            sample_idx = self._sample_idx_from_path(path)
            if sample_idx is None:
                continue
            try:
                data = _load_npz(path)
            except Exception as exc:  # pragma: no cover - corrupt cache diagnostics.
                LOGGER.warning("Skipping unreadable reliability cache %s: %s", path, exc)
                continue
            token = _scalar_to_str(data.get("sample_token"))
            if token:
                self.token_to_sample_idx.setdefault(token, sample_idx)
        self._token_index_built = True

    def _build_fingerprint_index(self) -> None:
        if self._fingerprint_to_sample_idx is not None:
            return
        index: dict[str, int] = {}
        for path in sorted(self.reliability_dir.glob("sample_*_reliability.npz")):
            sample_idx = self._sample_idx_from_path(path)
            if sample_idx is None:
                continue
            try:
                data = _load_npz(path)
                fingerprint = _coord_fingerprint(data["point_xyz"])
            except Exception as exc:  # pragma: no cover - corrupt cache diagnostics.
                LOGGER.warning("Skipping cache fingerprint for %s: %s", path, exc)
                continue
            index.setdefault(fingerprint, sample_idx)
        self._fingerprint_to_sample_idx = index

    @staticmethod
    def _sample_idx_from_path(path: Path) -> int | None:
        match = re.search(r"sample_(\d+)", path.name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _candidate_sample_indices(data_dict: dict[str, Any]) -> list[int]:
        candidates: list[int] = []
        for key in ("sample_idx", "sample_index", "index", "idx"):
            value = _scalar_to_int(data_dict.get(key))
            if value is not None:
                candidates.append(value)
        for key in ("name", "sample_name"):
            value = _scalar_to_str(data_dict.get(key))
            if value is None:
                continue
            match = re.search(r"sample[_-]?(\d+)", value)
            if match:
                candidates.append(int(match.group(1)))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _candidate_tokens(data_dict: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for key in ("sample_token", "token", "name"):
            value = _scalar_to_str(data_dict.get(key))
            if value:
                candidates.append(value)
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _candidate_timestamps(data_dict: dict[str, Any]) -> list[int]:
        candidates: list[int] = []
        for key in ("timestamp", "lidar_timestamp"):
            value = _scalar_to_int(data_dict.get(key))
            if value is not None:
                candidates.append(value)
        return list(dict.fromkeys(candidates))

    def _resolve_sample_idx(self, data_dict: dict[str, Any]) -> int | None:
        for sample_idx in self._candidate_sample_indices(data_dict):
            if self._paths_for_sample_idx(sample_idx)[0].exists():
                return sample_idx

        for timestamp in self._candidate_timestamps(data_dict):
            if timestamp in self.timestamp_to_sample_idx:
                return self.timestamp_to_sample_idx[timestamp]

        for token in self._candidate_tokens(data_dict):
            if token in self.token_to_sample_idx:
                return self.token_to_sample_idx[token]
        if self._candidate_tokens(data_dict):
            self._build_token_index()
            for token in self._candidate_tokens(data_dict):
                if token in self.token_to_sample_idx:
                    return self.token_to_sample_idx[token]

        coord = data_dict.get("coord")
        if coord is not None:
            self._build_fingerprint_index()
            fingerprint = _coord_fingerprint(np.asarray(coord))
            if self._fingerprint_to_sample_idx is not None:
                return self._fingerprint_to_sample_idx.get(fingerprint)
        return None

    def _paths_for_sample_idx(self, sample_idx: int) -> tuple[Path, Path]:
        prefix = f"sample_{sample_idx:04d}"
        return (
            self.reliability_dir / f"{prefix}_reliability.npz",
            self.dense_point_dir / f"{prefix}_dense_point_logits.npz",
        )

    def _component_weight(self, reliability: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        point_valid_mask = reliability["point_valid_mask"].astype(bool)
        if self.component_mode == "full":
            raw = reliability.get("reliability_weight_raw", reliability["reliability_weight"]).astype(np.float32)
            weight = reliability["reliability_weight"].astype(np.float32)
            return raw, weight
        if self.component_mode == "uniform":
            raw = point_valid_mask.astype(np.float32)
            return raw, raw
        else:
            distance = reliability["distance_weight"].astype(np.float32)
            geometric = reliability["geometric_weight"].astype(np.float32)
            semantic = reliability["semantic_weight"].astype(np.float32)
            factors = {
                "no_distance": (geometric, semantic),
                "no_geometric": (distance, semantic),
                "no_semantic": (distance, geometric),
            }[self.component_mode]
            raw = factors[0] * factors[1]
            raw[~point_valid_mask] = 0.0
        weight = _rank_calibrate(raw, point_valid_mask) if self.component_calibration == "rank" else raw
        return raw.astype(np.float32), weight.astype(np.float32)

    def _empty_teacher(self, num_points: int) -> dict[str, np.ndarray]:
        num_classes = max(self.teacher_num_classes, 1)
        return {
            "teacher_logits": np.zeros((num_points, num_classes), dtype=np.float32),
            "teacher_valid_mask": np.zeros(num_points, dtype=bool),
            "reliability_weight": np.zeros(num_points, dtype=np.float32),
            "reliability_weight_raw": np.zeros(num_points, dtype=np.float32),
        }

    def load_for_data_dict(self, data_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        coord = np.asarray(data_dict.get("coord"))
        if coord.ndim != 2:
            raise ValueError("Pointcept data_dict must contain rank-2 `coord` before reliability injection.")
        num_points = int(coord.shape[0])
        sample_idx = self._resolve_sample_idx(data_dict)
        if sample_idx is None:
            if self.strict:
                keys = sorted(str(key) for key in data_dict.keys())
                raise KeyError(f"Could not resolve sample_idx/token for reliability cache. data_dict keys={keys}")
            return self._empty_teacher(num_points)

        reliability_path, dense_point_path = self._paths_for_sample_idx(sample_idx)
        if not reliability_path.exists() or not dense_point_path.exists():
            if self.strict:
                raise FileNotFoundError(
                    f"Missing Stage 4 cache for sample_idx={sample_idx}: "
                    f"reliability={reliability_path.exists()}, dense_point={dense_point_path.exists()}"
                )
            return self._empty_teacher(num_points)

        reliability = _load_npz(reliability_path)
        dense_point = _load_npz(dense_point_path)
        cache_xyz = reliability["point_xyz"].astype(np.float32)
        if cache_xyz.shape[0] != num_points:
            if self.strict:
                raise ValueError(
                    f"Point count mismatch for sample_idx={sample_idx}: "
                    f"Pointcept coord={num_points}, reliability cache={cache_xyz.shape[0]}. "
                    "Use POINTCEPT_SWEEPS=1 and inject reliability before point sampling."
                )
            return self._empty_teacher(num_points)

        coord_error = float(np.max(np.abs(coord[:, :3].astype(np.float32) - cache_xyz[:, :3]))) if num_points else 0.0
        if coord_error > self.max_coord_error:
            if self.strict:
                raise ValueError(
                    f"Point coordinate mismatch for sample_idx={sample_idx}: max_abs_error={coord_error:.4f}. "
                    "The teacher cache is not aligned with the Pointcept point order."
                )
            return self._empty_teacher(num_points)

        teacher_logits = dense_point["point_teacher_logits"].astype(np.float32)
        if self.teacher_num_classes > 0:
            start = self.teacher_class_start
            end = start + self.teacher_num_classes
            if end > teacher_logits.shape[1]:
                raise ValueError(
                    f"teacher logits have {teacher_logits.shape[1]} classes, requested slice [{start}:{end}]"
                )
            teacher_logits = teacher_logits[:, start:end]

        raw_weight, reliability_weight = self._component_weight(reliability)
        dense_valid = dense_point["point_dense_valid_mask"].astype(bool)
        teacher_valid_mask = reliability["point_valid_mask"].astype(bool) & dense_valid
        return {
            "teacher_logits": teacher_logits.astype(np.float32),
            "teacher_valid_mask": teacher_valid_mask.astype(bool),
            "reliability_weight": reliability_weight.astype(np.float32),
            "reliability_weight_raw": raw_weight.astype(np.float32),
        }


class RALoadReliabilityTeacher:
    """Pointcept transform that injects teacher logits and reliability weights."""

    point_keys = (
        "teacher_logits",
        "teacher_valid_mask",
        "reliability_weight",
        "reliability_weight_raw",
    )

    def __init__(self, **kwargs: Any) -> None:
        self.cache = ReliabilityTeacherCache(**kwargs)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        data_dict.update(self.cache.load_for_data_dict(data_dict))
        index_valid_keys = list(
            data_dict.get(
                "index_valid_keys",
                [
                    "coord",
                    "color",
                    "normal",
                    "superpoint",
                    "strength",
                    "segment",
                    "instance",
                ],
            )
        )
        for key in self.point_keys:
            if key not in index_valid_keys:
                index_valid_keys.append(key)
        data_dict["index_valid_keys"] = index_valid_keys
        return data_dict


def _resolve_pointcept_transforms_registry() -> Any | None:
    """Find Pointcept's transform registry across small upstream layout shifts."""

    candidate_modules = (
        "pointcept.datasets.transform",
        "pointcept.datasets.transform.builder",
        "pointcept.datasets.transforms",
        "pointcept.datasets.transforms.builder",
        "pointcept.datasets.builder",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        transforms = getattr(module, "TRANSFORMS", None)
        if transforms is not None:
            return transforms
    return None


TRANSFORMS = _resolve_pointcept_transforms_registry()

if TRANSFORMS is not None:
    TRANSFORMS.register_module("RALoadReliabilityTeacher")(RALoadReliabilityTeacher)
