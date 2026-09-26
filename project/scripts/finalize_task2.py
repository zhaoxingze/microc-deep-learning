"""Verify frozen Task2A/B/C/D and generate final Task2 evidence artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task2d.finalization import finalize_task2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='rehash all source and final outputs')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = finalize_task2(args.project_root, verify=args.verify)
    except Exception as exc:
        print(f'Task2 finalization failed: {exc}', file=sys.stderr, flush=True)
        return 1
    print(f'TASK2 FINALIZED: {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
