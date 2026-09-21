from pathlib import Path

import numpy as np

from src.task2.window_scan import generate_windows, scan_replicates, stable_window_id


def test_generate_windows_keeps_only_complete_half_open_intervals() -> None:
    windows = generate_windows("chr", 20_000, window_bp=6_400, step_bp=4_000)
    assert [(item.start, item.end, item.center) for item in windows] == [
        (0, 6_400, 3_200),
        (4_000, 10_400, 7_200),
        (8_000, 14_400, 11_200),
        (12_000, 18_400, 15_200),
    ]
    assert stable_window_id("chr", 0, 6_400) == windows[0].window_id


def test_scan_replicates_writes_aligned_arrays_and_long_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_loader(cool_path, chrom, center, window_bp, target_bin_size, balance):
        value = 1.0 if "rep1" in str(cool_path) else 2.0
        matrix = np.full((4, 4), value, dtype=np.float32)
        if center == 2_000 and value == 2.0:
            matrix[0] = 0
        return matrix, {"padded": False, "chrom_used": "chr"}

    monkeypatch.setattr("src.task2.window_scan.load_local_matrix", fake_loader)
    metadata = scan_replicates(
        tmp_path / "rep1.cool",
        tmp_path / "rep2.cool",
        "chr",
        10_000,
        tmp_path,
        window_bp=4_000,
        step_bp=2_000,
        target_bin_size=1_000,
    )

    rep1 = np.load(tmp_path / "genome_windows_rep1.npy", mmap_mode="r")
    rep2 = np.load(tmp_path / "genome_windows_rep2.npy", mmap_mode="r")
    assert rep1.shape == rep2.shape == (4, 1, 4, 4)
    assert np.all(rep1 == 1) and rep2[0, 0, 0].sum() == 0
    assert metadata.groupby("replicate")["window_id"].apply(list).iloc[0] == metadata.groupby("replicate")["window_id"].apply(list).iloc[1]
    assert metadata.loc[(metadata.replicate == "rep2") & (metadata.array_index == 0), "zero_axis"].item()
