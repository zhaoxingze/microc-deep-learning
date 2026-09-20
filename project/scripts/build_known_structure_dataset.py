"""Build deterministic 64x64 datasets for known CHIN/OPCID/CHID structures."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_microc import load_local_matrix
from src.data.normalize import compute_oe, log1p_normalize


LABELS = {"CHIN": 0, "OPCID": 1, "CHID": 2}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cool_path(raw_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--cool-path does not exist: {path}")
        return path

    candidates = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file()
        and path.name.casefold().endswith(".cool")
        and "wt" in path.name.casefold()
        and re.search(r"rep(?:licate)?[_-]?1", path.name.casefold())
    )
    if len(candidates) == 1:
        return candidates[0].resolve()

    all_standalone = sorted(
        path.resolve() for path in raw_dir.rglob("*")
        if path.is_file() and path.name.casefold().endswith((".cool", ".cool.gz"))
    )
    rendered = "\n".join(f"  - {path}" for path in all_standalone) or "  (none)"
    raise ValueError(
        "WT rep1 could not be identified uniquely from an explicit 'WT' filename. "
        "Pass --cool-path. Standalone candidates:\n" + rendered
    )


def _validate_structures(frame: pd.DataFrame) -> None:
    required = {"type", "chrom", "start", "end", "center", "length_bp", "source_file", "source_sheet"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"structures.csv is missing columns: {missing}")
    unknown = sorted(set(frame["type"].dropna()) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown structure labels: {unknown}")
    if frame[["type", "chrom", "start", "end", "center"]].isna().any(axis=None):
        bad = frame.index[frame[["type", "chrom", "start", "end", "center"]].isna().any(axis=1)].tolist()
        raise ValueError(f"Missing required structure values at row indices: {bad[:20]}")
    invalid = frame.index[frame["end"] <= frame["start"]].tolist()
    if invalid:
        raise ValueError(f"Invalid intervals (end <= start) at row indices: {invalid[:20]}")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, default=PROJECT_ROOT / "data" / "processed" / "structures.csv")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT.parent / "micro-c数据")
    parser.add_argument("--cool-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--window-bp", type=int, default=6400)
    parser.add_argument("--target-bin-size", type=int, default=100)
    parser.add_argument("--balance", action="store_true", help="Use Cooler weights; unavailable for the current files.")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    structures_path = args.structures.resolve()
    if not structures_path.is_file():
        raise SystemExit(f"structures.csv not found: {structures_path}; run find_structure_annotations.py first.")
    structures = pd.read_csv(structures_path)
    _validate_structures(structures)
    cool_path = _resolve_cool_path(args.raw_dir.resolve(), args.cool_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_size = args.window_bp // args.target_bin_size
    if args.window_bp % args.target_bin_size or expected_size != 64:
        raise SystemExit(
            f"This phase requires 64x64 output; window_bp/target_bin_size is {args.window_bp}/{args.target_bin_size}."
        )
    print(f"Using Cooler explicitly/resolved as: {cool_path}")
    print(f"Building {len(structures)} deterministic windows...", flush=True)

    raw_matrices: list[np.ndarray] = []
    log_matrices: list[np.ndarray] = []
    oe_matrices: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, row in structures.iterrows():
        try:
            raw, window_metadata = load_local_matrix(
                cool_path=cool_path,
                chrom=str(row["chrom"]),
                center=float(row["center"]),
                window_bp=args.window_bp,
                target_bin_size=args.target_bin_size,
                balance=args.balance,
            )
            if raw.shape != (64, 64):
                raise ValueError(f"Expected (64, 64), got {raw.shape}")
            log_matrix = log1p_normalize(raw).astype(np.float32)
            oe_matrix = compute_oe(raw).astype(np.float32)
            all_versions = (raw, log_matrix, oe_matrix)
            if any(np.isnan(matrix).any() for matrix in all_versions):
                raise ValueError("NaN detected in raw/log1p/OE matrix")
            if any(np.isinf(matrix).any() for matrix in all_versions):
                raise ValueError("Inf detected in raw/log1p/OE matrix")

            symmetry_error = float(np.max(np.abs(raw.astype(np.float64) - raw.T.astype(np.float64))))
            zero_row_count = int(np.all(raw == 0, axis=1).sum())
            zero_col_count = int(np.all(raw == 0, axis=0).sum())
            raw_matrices.append(raw)
            log_matrices.append(log_matrix)
            oe_matrices.append(oe_matrix)
            labels.append(LABELS[str(row["type"])])
            metadata_rows.append(
                {
                    "sample_index": len(raw_matrices) - 1,
                    "label_id": LABELS[str(row["type"])],
                    "label": str(row["type"]),
                    "chrom": str(row["chrom"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "center": float(row["center"]),
                    "length_bp": int(row["length_bp"]),
                    "source_file": str(row["source_file"]),
                    "source_sheet": str(row.get("source_sheet", "")),
                    **window_metadata,
                    "shape": "64x64",
                    "raw_min": float(raw.min()),
                    "raw_max": float(raw.max()),
                    "raw_mean": float(raw.mean()),
                    "raw_sum": float(raw.sum()),
                    "all_zero": bool(np.count_nonzero(raw) == 0),
                    "zero_row_count": zero_row_count,
                    "zero_col_count": zero_col_count,
                    "has_zero_axis": bool(zero_row_count or zero_col_count),
                    "has_nan": False,
                    "has_inf": False,
                    "symmetry_max_abs_error": symmetry_error,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "structure_row": int(index),
                    "type": row.get("type", ""),
                    "chrom": row.get("chrom", ""),
                    "center": row.get("center", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (index + 1) % 25 == 0 or index + 1 == len(structures):
            print(f"  processed {index + 1}/{len(structures)}; failures={len(failures)}", flush=True)

    if failures:
        failure_path = output_dir / "known_build_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
        raise SystemExit(f"Dataset build stopped: {len(failures)} samples failed. See {failure_path}")

    raw_array = np.stack(raw_matrices).astype(np.float32)[:, np.newaxis, :, :]
    log_array = np.stack(log_matrices).astype(np.float32)[:, np.newaxis, :, :]
    oe_array = np.stack(oe_matrices).astype(np.float32)[:, np.newaxis, :, :]
    label_array = np.asarray(labels, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)

    output_paths = {
        "raw": output_dir / "known_windows_raw.npy",
        "log1p": output_dir / "known_windows_log1p.npy",
        "oe": output_dir / "known_windows_oe.npy",
        "labels": output_dir / "known_labels.npy",
        "metadata": output_dir / "known_metadata.csv",
    }
    np.save(output_paths["raw"], raw_array, allow_pickle=False)
    np.save(output_paths["log1p"], log_array, allow_pickle=False)
    np.save(output_paths["oe"], oe_array, allow_pickle=False)
    np.save(output_paths["labels"], label_array, allow_pickle=False)
    metadata.to_csv(output_paths["metadata"], index=False, encoding="utf-8-sig")

    contact_map = cooler.Cooler(str(cool_path))
    manifest = {
        "phase": "known_structure_data_engineering",
        "random_seed": args.seed,
        "cool_path": str(cool_path),
        "cool_sha256": _sha256(cool_path),
        "structures_path": str(structures_path),
        "structures_sha256": _sha256(structures_path),
        "window_bp": args.window_bp,
        "source_bin_size": int(contact_map.binsize),
        "target_bin_size": args.target_bin_size,
        "balance": args.balance,
        "label_mapping": LABELS,
        "x_shape": list(raw_array.shape),
        "y_shape": list(label_array.shape),
        "class_counts": metadata["label"].value_counts().sort_index().to_dict(),
        "padded_samples": int(metadata["padded"].sum()),
        "all_zero_samples": int(metadata["all_zero"].sum()),
        "zero_axis_samples": int(metadata["has_zero_axis"].sum()),
        "max_symmetry_error": float(metadata["symmetry_max_abs_error"].max()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cooler_version": cooler.__version__,
        "output_sha256": {name: _sha256(path) for name, path in output_paths.items()},
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"X shape: {raw_array.shape}; y shape: {label_array.shape}")
    print(f"Class counts: {manifest['class_counts']}")
    print(
        f"Padded={manifest['padded_samples']}; all-zero={manifest['all_zero_samples']}; "
        f"zero-axis={manifest['zero_axis_samples']}; max symmetry error={manifest['max_symmetry_error']}"
    )
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
