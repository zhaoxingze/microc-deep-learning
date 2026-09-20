from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import pytest

from src.data.load_microc import load_local_matrix


@pytest.fixture()
def toy_cool(tmp_path: Path) -> Path:
    path = tmp_path / "toy.cool"
    starts = np.arange(0, 1000, 10)
    bins = pd.DataFrame({"chrom": "NC_000913.3", "start": starts, "end": starts + 10})
    pixels = pd.DataFrame(
        {
            "bin1_id": np.arange(len(bins)),
            "bin2_id": np.arange(len(bins)),
            "count": np.ones(len(bins), dtype=int),
        }
    )
    cooler.create_cooler(str(path), bins, pixels, assembly="MG1655")
    return path


def test_local_fetch_aggregates_counts_and_maps_mg1655_alias(toy_cool: Path) -> None:
    matrix, metadata = load_local_matrix(
        toy_cool, "MG1655", center=100, window_bp=200, target_bin_size=20
    )
    assert matrix.shape == (10, 10)
    assert np.allclose(np.diag(matrix), 2.0)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0
    assert metadata["chrom_used"] == "NC_000913.3"
    assert metadata["padded"] is False


def test_boundary_window_is_zero_padded(toy_cool: Path) -> None:
    matrix, metadata = load_local_matrix(
        toy_cool, "MG1655", center=0, window_bp=200, target_bin_size=20
    )
    assert matrix.shape == (10, 10)
    assert np.count_nonzero(matrix[:5]) == 0
    assert metadata["padded"] is True


def test_balance_requires_weight_column(toy_cool: Path) -> None:
    with pytest.raises(ValueError, match="weight"):
        load_local_matrix(toy_cool, "MG1655", 100, 200, 20, balance=True)
