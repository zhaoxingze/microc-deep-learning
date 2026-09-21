"""Run Task 2A genome-wide candidate detection in smoke or full mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.pipeline import PipelineError
from src.task2.pipeline import Task2AConfig, execute_task2a


def build_parser() -> argparse.ArgumentParser:
    raw_root = PROJECT_ROOT.parent / "micro-c数据"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--detector-version", choices=("legacy", "refined"), default="legacy",
        help="Use the original mean-fusion detector or the versioned refined OR-Max detector.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=raw_root)
    parser.add_argument(
        "--rep1-cool", type=Path,
        default=raw_root / "GSE272159_37C_rep1.mapq_30.10.cool",
    )
    parser.add_argument(
        "--rep2-cool", type=Path,
        default=raw_root / "GSE272159_37C_rep2.mapq_30.10.cool",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Task2AConfig(
        project_root=args.project_root.resolve(), raw_root=args.raw_root.resolve(),
        rep1_cool=args.rep1_cool.resolve(), rep2_cool=args.rep2_cool.resolve(),
        mode=args.mode, detector_version=args.detector_version,
        dry_run=args.dry_run, resume=args.resume,
    )
    try:
        execute_task2a(config)
    except PipelineError as exc:
        print(f"Task 2A pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
