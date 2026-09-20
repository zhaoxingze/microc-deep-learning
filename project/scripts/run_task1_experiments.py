"""Run the fair raw/log1p/OE comparison, locking all checkpoints before test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.task1.evaluation import select_best_input
from train_task1 import run_test, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--split-path", type=Path, default=PROJECT_ROOT / "data" / "splits" / "group_split.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "task1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--train-replicates", default="rep1,rep2")
    args = parser.parse_args()

    trained: dict[str, tuple[object, dict[str, object]]] = {}
    for input_type in ("raw", "log1p", "oe"):
        print(f"TRAIN PHASE: {input_type}", flush=True)
        trained[input_type] = run_training(
            input_type,
            args.processed_root.resolve(),
            args.split_path.resolve(),
            (args.output_root / input_type).resolve(),
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            args.seed,
            False,
            args.train_replicates,
        )

    rows: list[dict[str, object]] = []
    for input_type in ("raw", "log1p", "oe"):
        print(f"LOCKED TEST PHASE: {input_type}", flush=True)
        training, config = trained[input_type]
        test_metrics = run_test(
            input_type,
            args.processed_root.resolve(),
            args.split_path.resolve(),
            (args.output_root / input_type).resolve(),
            args.batch_size,
            args.seed,
            False,
            args.train_replicates,
        )
        rows.append(
            {
                "input_type": input_type,
                "best_epoch": training.best_epoch,
                "best_val_accuracy": training.best_validation["val_structure_accuracy"],
                "best_val_macro_f1": training.best_validation["val_structure_macro_f1"],
                "best_val_loss": training.best_validation["val_structure_loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                **{
                    f"test_{label}_{metric}": test_metrics[f"{label}_{metric}"]
                    for label in ("CHIN", "OPCID", "CHID")
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    comparison = pd.DataFrame(rows)
    best_input = select_best_input(comparison)
    comparison["best_input_type"] = comparison["input_type"].eq(best_input)
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_root / "input_comparison.csv", index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False))
    print(f"Validation-selected best input: {best_input}")


if __name__ == "__main__":
    main()
