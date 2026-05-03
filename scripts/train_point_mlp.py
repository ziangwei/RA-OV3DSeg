from __future__ import annotations

import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    print(
        "[WARN] scripts/train_point_mlp.py is kept for compatibility. "
        "Use scripts/train_3d_segmentor.py --backbone debug_point_mlp instead.",
        file=sys.stderr,
    )
    module_globals = runpy.run_path(str(ROOT / "scripts" / "train_3d_segmentor.py"))
    main = module_globals["main"]
    raise SystemExit(main())
