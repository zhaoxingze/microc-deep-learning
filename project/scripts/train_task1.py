"""Train or independently evaluate one Task 1 CNN input experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.dataset import (
    PairedMicroCDataset,
    build_records,
    compute_structure_class_weights,
)
from src.task1.evaluation import evaluate_locked_checkpoint
from src.task1.model import SmallMicroCCNN
from src.task1.reporting import write_evaluation_outputs, write_training_history
from src.task1.trainer import TrainingResult, set_reproducible_seed, train_experiment


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load_frames(processed_root: Path, split_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(processed_root / "paired_known_metadata.csv")
    split = pd.read_csv(split_path)
    return paired, split


def _loader(
    records: pd.DataFrame,
    processed_root: Path,
    input_type: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PairedMicroCDataset(records, processed_root, input_type)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def participation_counts(records_by_split: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    return {
        split_name: {
            "structures": int(records["structure_id"].nunique()),
            "replicates": int(len(records)),
        }
        for split_name, records in records_by_split.items()
    }


def prepare_records(
    processed_root: Path,
    split_path: Path,
    exclude_zero_axis: bool,
    train_replicates: str,
) -> tuple[dict[str, pd.DataFrame], torch.Tensor]:
    paired, split = _load_frames(processed_root, split_path)
    records = {
        "train": build_records(
            paired,
            split,
            "train",
            train_replicates=train_replicates,
            exclude_zero_axis=exclude_zero_axis,
            structure_balanced=exclude_zero_axis,
        ),
        "val": build_records(
            paired,
            split,
            "val",
            train_replicates=("rep1", "rep2"),
            exclude_zero_axis=exclude_zero_axis,
        ),
        "test": build_records(
            paired,
            split,
            "test",
            train_replicates=("rep1", "rep2"),
            exclude_zero_axis=exclude_zero_axis,
        ),
    }
    for split_name, frame in records.items():
        if frame.empty:
            raise ValueError(f"No available records for {split_name}")
    class_weights = compute_structure_class_weights(records["train"])
    return records, class_weights


def run_training(
    input_type: str,
    processed_root: Path,
    split_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    exclude_zero_axis: bool = False,
    train_replicates: str = "rep1,rep2",
    device: torch.device | None = None,
) -> tuple[TrainingResult, dict[str, object]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, class_weights = prepare_records(processed_root, split_path, exclude_zero_axis, train_replicates)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(seed)
    train_loader = _loader(records["train"], processed_root, input_type, batch_size, True, seed)
    val_loader = _loader(records["val"], processed_root, input_type, batch_size, False, seed)
    set_reproducible_seed(seed)
    model = SmallMicroCCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    checkpoint_path = output_dir / "best_model.pth"
    training = train_experiment(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    write_training_history(training.history, output_dir)
    validation = evaluate_locked_checkpoint(
        checkpoint_path,
        SmallMicroCCNN,
        val_loader,
        class_weights,
        device,
    )
    write_evaluation_outputs(validation, output_dir / "validation")
    config: dict[str, object] = {
        "input_type": input_type,
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_ran": int(len(training.history)),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "exclude_zero_axis": exclude_zero_axis,
        "train_replicates": train_replicates,
        "class_weights": class_weights.tolist(),
        "best_epoch": training.best_epoch,
        "best_validation": training.best_validation,
        "participation": participation_counts(records),
        "device": str(device),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return training, config


def run_test(
    input_type: str,
    processed_root: Path,
    split_path: Path,
    output_dir: Path,
    batch_size: int,
    seed: int,
    exclude_zero_axis: bool = False,
    train_replicates: str = "rep1,rep2",
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, class_weights = prepare_records(processed_root, split_path, exclude_zero_axis, train_replicates)
    test_loader = _loader(records["test"], processed_root, input_type, batch_size, False, seed)
    result = evaluate_locked_checkpoint(
        output_dir / "best_model.pth",
        SmallMicroCCNN,
        test_loader,
        class_weights,
        device,
    )
    write_evaluation_outputs(result, output_dir)
    return result.summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("raw", "log1p", "oe"), required=True)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--split-path", type=Path, default=PROJECT_ROOT / "data" / "splits" / "group_split.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--exclude-zero-axis", action="store_true")
    parser.add_argument("--train-replicates", default="rep1,rep2")
    return parser


def main() -> None:
    _configure_utf8()
    args = build_parser().parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "task1" / args.input_type
    if args.mode in {"train", "both"}:
        run_training(
            args.input_type,
            args.processed_root.resolve(),
            args.split_path.resolve(),
            output_dir.resolve(),
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            args.seed,
            args.exclude_zero_axis,
            args.train_replicates,
        )
    if args.mode in {"test", "both"}:
        run_test(
            args.input_type,
            args.processed_root.resolve(),
            args.split_path.resolve(),
            output_dir.resolve(),
            args.batch_size,
            args.seed,
            args.exclude_zero_axis,
            args.train_replicates,
        )


if __name__ == "__main__":
    main()
