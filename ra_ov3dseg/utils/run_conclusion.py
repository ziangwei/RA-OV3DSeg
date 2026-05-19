"""Standardized run conclusion block.

Every training, evaluation, and extraction script MUST emit a RunConclusion
block as its last action. The block is parseable, one key per line, and
human-readable.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from pathlib import Path
from typing import Literal

Status = Literal["success", "failed", "stopped_by_gate", "crashed"]


@dataclasses.dataclass
class RunConclusion:
    stage: str
    experiment: str
    status: Status
    gate: str
    gate_passed: bool
    primary_metric_name: str
    primary_metric_value: float
    secondary: dict[str, float]
    runtime_seconds: float
    checkpoint: str | None
    artifacts: list[str]
    next_step: str
    notes: str = "-"

    def print_block(self) -> None:
        lines = [
            "========== RUN_CONCLUSION ==========",
            f"stage:             {self.stage}",
            f"experiment:        {self.experiment}",
            f"status:            {self.status}",
            f"gate:              {self.gate}",
            f"gate_passed:       {'yes' if self.gate_passed else 'no'}",
            "result:",
            f"  primary_metric:  {self.primary_metric_name} = {self.primary_metric_value:.4f}",
        ]
        if self.secondary:
            sec = ", ".join(f"{k}={v:.4f}" for k, v in self.secondary.items())
            lines.append(f"  secondary:       {sec}")
        else:
            lines.append("  secondary:       -")
        lines.append(f"runtime:           {self._format_runtime()}")
        lines.append(f"checkpoint:        {self.checkpoint or '-'}")
        if self.artifacts:
            lines.append(f"artifacts:         {self.artifacts[0]}")
            for art in self.artifacts[1:]:
                lines.append(f"                   {art}")
        else:
            lines.append("artifacts:         -")
        lines.append(f"next_step:         {self.next_step}")
        lines.append(f"notes:             {self.notes}")
        lines.append("====================================")
        for line in lines:
            print(line)

    def _format_runtime(self) -> str:
        secs = int(self.runtime_seconds)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def append_to_recap(self, recap_path: Path) -> None:
        """Append one row to the experiment ledger.

        Runtime scripts default to an untracked local recap to avoid dirtying
        `docs/EXPERIMENT_RECAP.md` on shared training servers. Set
        `RA_OV3DSEG_RECAP_MODE=tracked` for a deliberate tracked-doc append,
        `RA_OV3DSEG_RECAP_PATH=/path/to/file.md` for a custom ledger, or
        `RA_OV3DSEG_RECAP_MODE=off` to skip appending.
        """

        row = " | ".join(
            [
                datetime.date.today().isoformat(),
                self.stage,
                self.experiment,
                self.status,
                f"{self.primary_metric_name}={self.primary_metric_value:.4f}",
                (self.notes or "-").replace("\n", " ")[:80],
            ]
        )
        recap_path = self._resolve_recap_path(Path(recap_path))
        if recap_path is None:
            return
        recap_path.parent.mkdir(parents=True, exist_ok=True)
        if not recap_path.exists():
            recap_path.write_text(
                "| Date | Stage | Experiment | Status | Primary Metric | Notes |\n"
                "|---|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        with recap_path.open("a", encoding="utf-8") as f:
            f.write(f"| {row} |\n")

    @staticmethod
    def _resolve_recap_path(default_path: Path) -> Path | None:
        override = os.environ.get("RA_OV3DSEG_RECAP_PATH")
        if override:
            return Path(override)

        mode = os.environ.get("RA_OV3DSEG_RECAP_MODE", "local").strip().lower()
        if mode in {"off", "none", "skip"}:
            return None
        if mode == "tracked":
            return default_path
        if mode != "local":
            raise ValueError("RA_OV3DSEG_RECAP_MODE must be one of: local, tracked, off")
        return Path("outputs/run_conclusions/EXPERIMENT_RECAP.local.md")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)
