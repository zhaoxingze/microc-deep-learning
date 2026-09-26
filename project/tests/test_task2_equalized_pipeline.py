from pathlib import Path

import pytest

from src.task2.equalized_pipeline import (
    EqualizedPaths,
    frozen_evaluation_domain,
    assert_protected_unchanged,
    protected_tree_hashes,
    resolve_equalized_inputs,
    validate_prior_direct_inputs,
)

import json
import pandas as pd


def test_equalized_input_contract_reads_only_existing_postprocessing_artifacts(tmp_path):
    paths = EqualizedPaths(tmp_path)
    inputs = resolve_equalized_inputs(paths)
    score_paths = {inputs[f"{scale}_scores"] for scale in ("small", "medium", "large")}
    assert score_paths == {
        tmp_path / "outputs" / "task2a_multiscale" / scale / "window_scores.csv"
        for scale in ("small", "medium", "large")
    }
    forbidden_suffixes = {".cool", ".npy", ".pt", ".pth", ".ckpt"}
    assert not any(path.suffix.lower() in forbidden_suffixes for path in inputs.values())
    assert paths.output_root == tmp_path / "outputs" / "task2a_multiscale_equalized"


def test_protected_hash_audit_detects_any_source_mutation(tmp_path):
    protected = tmp_path / "outputs" / "task2a_multiscale"
    protected.mkdir(parents=True)
    source = protected / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    before = protected_tree_hashes(tmp_path)
    source.write_text("a\n2\n", encoding="utf-8")
    after = protected_tree_hashes(tmp_path)
    with pytest.raises(ValueError, match="protected Task2A.3"):
        assert_protected_unchanged(before, after)


def test_protected_hash_scope_excludes_equalized_output(tmp_path):
    old = tmp_path / "outputs" / "task2a_multiscale"
    new = tmp_path / "outputs" / "task2a_multiscale_equalized"
    data = tmp_path / "data" / "task2_multiscale"
    for root, name in ((old, "old.txt"), (new, "new.txt"), (data, "data.txt")):
        root.mkdir(parents=True)
        (root / name).write_text(name, encoding="utf-8")
    hashes = protected_tree_hashes(tmp_path)
    assert str((old / "old.txt").resolve()) in hashes
    assert str((data / "data.txt").resolve()) in hashes
    assert str((new / "new.txt").resolve()) not in hashes


def test_frozen_domain_accepts_legacy_linear_protocol_marker():
    protocol = {"evaluation_domain_size": 4_641_652, "random_domain": "linear"}
    known = pd.DataFrame({"chrom": ["MG1655", "MG1655"]})
    assert frozen_evaluation_domain(protocol, known) == ("MG1655", 4_641_652)


def test_prior_validation_does_not_touch_transitive_raw_or_model_inputs(tmp_path):
    direct = tmp_path / "window_scores.csv"
    direct.write_text("window_id\na\n", encoding="utf-8")
    from src.task1.manifest import sha256_file
    missing_cool = tmp_path / "archived.cool"
    manifest = tmp_path / "prior_manifest.json"
    manifest.write_text(json.dumps({
        "inputs": {str(missing_cool.resolve()): {"sha256": "not-accessed"}},
        "outputs": {str(direct.resolve()): {"sha256": sha256_file(direct)}},
    }), encoding="utf-8")
    payload = validate_prior_direct_inputs(manifest, [direct])
    assert str(missing_cool.resolve()) in payload["inputs"]
