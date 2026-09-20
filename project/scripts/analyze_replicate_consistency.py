"""Compute per-structure and class-level rep1/rep2 matrix consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.consistency import matrix_correlations


def _summaries(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in detail if column.startswith(("pearson_", "spearman_"))]
    rows: list[dict[str, object]] = []
    groups = [("overall", detail), *[(label, detail.loc[detail["type"] == label]) for label in ("CHIN", "OPCID", "CHID")]]
    for group_name, subset in groups:
        for metric in metrics:
            values = subset[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "N": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "Q1": float(np.percentile(values, 25)) if len(values) else np.nan,
                    "Q3": float(np.percentile(values, 75)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "task1")
    parser.add_argument("--exclude-diagonal-band", type=int, default=2)
    args = parser.parse_args()

    root = args.processed_root.resolve()
    rep1 = np.load(root / "rep1" / "known_windows_raw.npy", mmap_mode="r", allow_pickle=False)[:, 0]
    rep2 = np.load(root / "rep2" / "known_windows_raw.npy", mmap_mode="r", allow_pickle=False)[:, 0]
    meta1 = pd.read_csv(root / "rep1" / "known_metadata.csv")
    meta2 = pd.read_csv(root / "rep2" / "known_metadata.csv")
    if meta1["structure_id"].tolist() != meta2["structure_id"].tolist() or rep1.shape != rep2.shape:
        raise SystemExit("Replicate arrays or structure IDs are not aligned")

    rows = []
    for index, structure_id in enumerate(meta1["structure_id"]):
        rows.append(
            {
                "structure_id": structure_id,
                "type": meta1.iloc[index]["type"],
                **matrix_correlations(rep1[index], rep2[index], args.exclude_diagonal_band),
            }
        )
    detail = pd.DataFrame(rows)
    summary = _summaries(detail)
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_root / "replicate_consistency.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_root / "replicate_consistency_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Replicate pairs: {len(detail)}")
    print(summary.loc[summary["metric"].isin(["pearson_upper", "spearman_upper", f"pearson_exclude_band{args.exclude_diagonal_band}", f"spearman_exclude_band{args.exclude_diagonal_band}"])].to_string(index=False))


if __name__ == "__main__":
    main()
