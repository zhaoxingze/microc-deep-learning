"""Build aligned rep1/rep2 known-structure datasets with stable IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.known_dataset import (
    attach_structure_ids,
    build_replicate,
    combine_replicate_metadata,
    write_replicate_build,
)


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, default=PROJECT_ROOT / "data" / "processed" / "structures.csv")
    parser.add_argument("--rep1-cool", type=Path, required=True)
    parser.add_argument("--rep2-cool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--window-bp", type=int, default=6400)
    parser.add_argument("--target-bin-size", type=int, default=100)
    args = parser.parse_args()

    structures = attach_structure_ids(pd.read_csv(args.structures.resolve()))
    builds = {}
    for replicate, cool_path in (("rep1", args.rep1_cool), ("rep2", args.rep2_cool)):
        print(f"Building {replicate} from {cool_path.resolve()}...", flush=True)
        builds[replicate] = build_replicate(
            structures,
            cool_path,
            replicate,
            window_bp=args.window_bp,
            target_bin_size=args.target_bin_size,
        )
        write_replicate_build(builds[replicate], args.output_root.resolve() / replicate)
        print(f"  {replicate}: {builds[replicate].arrays['raw'].shape}", flush=True)

    rep1, rep2 = builds["rep1"], builds["rep2"]
    for input_type in ("raw", "log1p", "oe"):
        if rep1.arrays[input_type].shape != rep2.arrays[input_type].shape:
            raise SystemExit(f"Shape mismatch for {input_type}: {rep1.arrays[input_type].shape} vs {rep2.arrays[input_type].shape}")
    if not np.array_equal(rep1.labels, rep2.labels):
        raise SystemExit("rep1/rep2 labels differ")

    paired = combine_replicate_metadata(rep1.metadata, rep2.metadata)
    paired_path = args.output_root.resolve() / "paired_known_metadata.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    structures.to_csv(args.output_root.resolve() / "structures_with_ids.csv", index=False, encoding="utf-8-sig")
    print(f"Paired metadata: {paired_path} ({len(paired)} rows)")
    print("Alignment check: PASS")


if __name__ == "__main__":
    main()
