"""Build and independently validate the fixed genomic-group MILP split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.splitting import (
    SPLITS,
    assign_genomic_groups,
    solve_group_milp,
    split_summary,
    validate_group_split,
)


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, default=PROJECT_ROOT / "data" / "processed" / "structures_with_ids.csv")
    parser.add_argument("--paired-metadata", type=Path, default=PROJECT_ROOT / "data" / "processed" / "paired_known_metadata.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "splits" / "group_split.csv")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "task1" / "split_summary.csv")
    parser.add_argument("--window-bp", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    structures = pd.read_csv(args.structures.resolve())
    grouped = assign_genomic_groups(structures, window_bp=args.window_bp)
    split = solve_group_milp(grouped, seed=args.seed)
    validate_group_split(split)
    summary = split_summary(split)

    output_columns = ["structure_id", "type", "genomic_group_id", "split"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split[output_columns].to_csv(args.output, index=False, encoding="utf-8-sig")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False, encoding="utf-8-sig")

    paired = pd.read_csv(args.paired_metadata.resolve())
    paired = paired.drop(columns=["genomic_group_id"], errors="ignore").merge(
        split[["structure_id", "genomic_group_id"]],
        on="structure_id",
        how="left",
        validate="many_to_one",
    )
    if paired["genomic_group_id"].isna().any():
        raise SystemExit("Paired metadata contains structures without a genomic group")
    paired.to_csv(args.paired_metadata.resolve(), index=False, encoding="utf-8-sig")

    group_stats = split.groupby("genomic_group_id").agg(
        size=("structure_id", "size"),
        class_count=("type", "nunique"),
    )
    group_sets = {name: set(split.loc[split.split == name, "genomic_group_id"]) for name in SPLITS}
    leakage = {
        "train_val": sorted(group_sets["train"] & group_sets["val"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "val_test": sorted(group_sets["val"] & group_sets["test"]),
    }
    diagnostics = {
        "structures": int(len(split)),
        "genomic_groups": int(split["genomic_group_id"].nunique()),
        "mixed_class_groups": int((group_stats["class_count"] > 1).sum()),
        "largest_group_size": int(group_stats["size"].max()),
        "leakage": leakage,
        "summary": summary.to_dict(orient="records"),
    }
    diagnostics_path = args.summary.parent / "split_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Genomic groups: {diagnostics['genomic_groups']}")
    print(f"Mixed-class groups: {diagnostics['mixed_class_groups']}; largest group: {diagnostics['largest_group_size']}")
    print(summary.to_string(index=False))
    print(f"Group leakage intersections: {leakage}")
    print("Group leakage check: PASS")


if __name__ == "__main__":
    main()
