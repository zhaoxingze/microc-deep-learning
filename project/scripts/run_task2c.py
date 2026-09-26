"""Run frozen Task2C unsupervised grouping in dry-run, smoke or full mode."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task2c.pipeline import execute_task2c


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = execute_task2c(args.project_root, mode=args.mode,
                                dry_run=args.dry_run, resume=args.resume)
    except Exception as exc:
        print(f"Task2C failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"Task2C mode={args.mode} dry_run={result.dry_run} resumed={result.resumed} "
          f"output={result.output_root} elapsed_seconds={result.elapsed_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
