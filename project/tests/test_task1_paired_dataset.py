from __future__ import annotations

import pandas as pd
import pytest

from src.data.known_dataset import (
    attach_structure_ids,
    combine_replicate_metadata,
    stable_structure_id,
)


def _structure_row() -> dict[str, object]:
    return {
        "type": "CHIN",
        "chrom": "MG1655",
        "start": 10,
        "end": 30,
        "center": 20.0,
    }


def test_stable_structure_id_ignores_mapping_order() -> None:
    row = _structure_row()
    reversed_row = dict(reversed(list(row.items())))
    assert stable_structure_id(row) == stable_structure_id(reversed_row)


def test_stable_structure_id_changes_when_coordinate_changes() -> None:
    row = _structure_row()
    changed = {**row, "end": 31}
    assert stable_structure_id(row) != stable_structure_id(changed)


def test_attach_structure_ids_rejects_duplicate_stable_keys() -> None:
    frame = pd.DataFrame([_structure_row(), _structure_row()])
    with pytest.raises(ValueError, match="Duplicate stable structure_id"):
        attach_structure_ids(frame)


def test_pair_metadata_aligns_same_structure_ids_and_labels() -> None:
    structures = attach_structure_ids(
        pd.DataFrame(
            [
                _structure_row(),
                {"type": "OPCID", "chrom": "MG1655", "start": 40, "end": 80, "center": 60.0},
            ]
        )
    )
    base = structures[["structure_id", "type", "chrom", "start", "end", "center"]].copy()
    rep1 = base.assign(replicate="rep1", has_zero_axis=False, zero_row_count=0, zero_col_count=0, padded=False, available=True)
    rep2 = base.assign(replicate="rep2", has_zero_axis=False, zero_row_count=0, zero_col_count=0, padded=False, available=True)
    groups = structures[["structure_id"]].assign(genomic_group_id=["group_0001", "group_0002"])

    paired = combine_replicate_metadata(rep1, rep2, groups)

    assert paired.loc[paired.replicate == "rep1", "structure_id"].tolist() == paired.loc[paired.replicate == "rep2", "structure_id"].tolist()
    assert paired.groupby("structure_id")["type"].nunique().max() == 1
    assert paired.groupby("structure_id")["genomic_group_id"].nunique().max() == 1


def test_pair_metadata_rejects_misaligned_replicates() -> None:
    rep1 = pd.DataFrame([{"structure_id": "A", "type": "CHIN", "replicate": "rep1"}])
    rep2 = pd.DataFrame([{"structure_id": "B", "type": "CHIN", "replicate": "rep2"}])
    with pytest.raises(ValueError, match="structure_id order"):
        combine_replicate_metadata(rep1, rep2)
