from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.teacher_registry import SUPPORTED_TEACHERS, describe_teacher  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect registered 2D teacher backends.")
    parser.add_argument(
        "--teacher_backend",
        default=None,
        choices=list(SUPPORTED_TEACHERS),
        help="Inspect one teacher. If omitted, prints all registered teachers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("check_teacher_backend")
    teacher_names = [args.teacher_backend] if args.teacher_backend is not None else list(SUPPORTED_TEACHERS)

    for teacher_name in teacher_names:
        spec = describe_teacher(teacher_name)
        logger.info(
            "%s | role=%s | granularity=%s | baseline=%s | %s",
            spec.name,
            spec.role,
            spec.feature_granularity,
            spec.is_baseline,
            spec.description,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
