"""Run the frozen Task2A.4 cross-scale calibration equalization."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task2.equalized_pipeline import execute_equalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_equalized(args.project_root)
    except Exception as exc:
        print(f"Task2A.4 equalized pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"Task2A.4 outputs: {result.output_root}")
    print(f"Task2A.4 elapsed seconds: {result.elapsed_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
