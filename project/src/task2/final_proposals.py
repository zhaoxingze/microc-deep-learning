"""Frozen six-channel Task2A proposal construction without known labels."""
from __future__ import annotations

import hashlib
from bisect import bisect_left, insort

import numpy as np
import pandas as pd


PROPOSAL_P_THRESHOLD = 0.05
SCALE_GEOMETRY = {"small": (3200, 400), "medium": (6400, 800), "large": (12800, 1600)}
BRANCHES = ("density", "shape")
CHANNELS = tuple((scale, branch) for scale in SCALE_GEOMETRY for branch in BRANCHES)


def generate_channel_proposals(scores: pd.DataFrame) -> pd.DataFrame:
    """Threshold and NMS each scale × branch independently."""
    required = {"window_id", "chrom", "start", "end", "center", "window_bp", "scale",
                "zero_axis_rep1", "zero_axis_rep2"}
    required |= {f"{rep}_{branch}_p" for rep in ("rep1", "rep2") for branch in BRANCHES}
    if missing := required.difference(scores.columns):
        raise ValueError(f"missing score columns: {sorted(missing)}")
    if scores.window_id.duplicated().any() or not set(scores.scale).issubset(SCALE_GEOMETRY):
        raise ValueError("scores require unique IDs and known scales")
    output = []
    for scale, branch in CHANNELS:
        frame = scores.loc[scores.scale.eq(scale)].copy()
        if frame.empty:
            continue
        bp, _ = SCALE_GEOMETRY[scale]
        if not frame.window_bp.eq(bp).all():
            raise ValueError(f"{scale} window geometry mismatch")
        p1 = frame[f"rep1_{branch}_p"].to_numpy(dtype=float)
        p2 = frame[f"rep2_{branch}_p"].to_numpy(dtype=float)
        if not np.isfinite(p1).all() or not np.isfinite(p2).all() or (
            (p1 <= 0) | (p1 > 1) | (p2 <= 0) | (p2 > 1)
        ).any():
            raise ValueError("invalid empirical p-values")
        frame["best_p"] = np.sqrt(p1 * p2)
        frame["replicate_min_anomaly"] = -np.log10(np.maximum(p1, p2))
        frame["rep1_supported"] = p1 <= PROPOSAL_P_THRESHOLD
        frame["rep2_supported"] = p2 <= PROPOSAL_P_THRESHOLD
        frame = frame.loc[frame.best_p.le(PROPOSAL_P_THRESHOLD)]
        frame = frame.sort_values(
            ["best_p", "replicate_min_anomaly", "chrom", "start", "end", "window_id"],
            ascending=[True, False, True, True, True, True], kind="mergesort",
        )
        centers: dict[str, list[int]] = {}
        for row in frame.itertuples(index=False):
            occupied = centers.setdefault(str(row.chrom), [])
            center = int(row.center)
            position = bisect_left(occupied, center)
            neighbors = occupied[max(0, position - 1):position + 1]
            if any(abs(center - neighbor) < bp / 2 for neighbor in neighbors):
                continue
            insort(occupied, center)
            output.append({
                "proposal_id": f"{scale}_{branch}_{row.window_id}",
                "channel": f"{scale}_{branch}",
                "scale": scale, "branch": branch, "window_id": row.window_id,
                "chrom": row.chrom, "start": int(row.start), "end": int(row.end),
                "center": int(row.center), "window_bp": bp,
                "best_p": float(row.best_p),
                "replicate_min_anomaly": float(row.replicate_min_anomaly),
                "rep1_supported": bool(row.rep1_supported),
                "rep2_supported": bool(row.rep2_supported),
                "zero_axis_flag": bool(row.zero_axis_rep1 or row.zero_axis_rep2),
            })
    columns = ["proposal_id", "channel", "scale", "branch", "window_id", "chrom",
               "start", "end", "center", "window_bp", "best_p",
               "replicate_min_anomaly", "rep1_supported", "rep2_supported", "zero_axis_flag"]
    return pd.DataFrame(output, columns=columns)


def merge_proposals(proposals: pd.DataFrame) -> pd.DataFrame:
    """Merge transitive positive interval overlaps after the six channels are complete."""
    if proposals.empty:
        raise ValueError("no proposals passed the fixed p threshold")
    required = {"chrom", "start", "end", "center", "channel", "window_id", "scale",
                "branch", "best_p", "replicate_min_anomaly", "rep1_supported",
                "rep2_supported", "zero_axis_flag"}
    if missing := required.difference(proposals.columns):
        raise ValueError(f"proposal columns missing: {sorted(missing)}")
    ordered = proposals.sort_values(["chrom", "start", "end", "window_id", "channel"])
    components: list[pd.DataFrame] = []
    current: list[int] = []
    current_chrom = None
    current_end = -1
    for index, row in ordered.iterrows():
        if current and (row.chrom != current_chrom or int(row.start) >= current_end):
            components.append(ordered.loc[current])
            current = []
            current_end = -1
        current.append(index)
        current_chrom = row.chrom
        current_end = max(current_end, int(row.end))
    if current:
        components.append(ordered.loc[current])
    regions = []
    for members in components:
        representative = members.sort_values(
            ["best_p", "replicate_min_anomaly", "chrom", "start", "end", "window_id", "channel"],
            ascending=[True, False, True, True, True, True, True], kind="mergesort",
        ).iloc[0]
        start, end = int(members.start.min()), int(members.end.max())
        key = f"{representative.chrom}|{start}|{end}|{representative.window_id}"
        region_id = "FREG_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        channels = set(members.channel.astype(str))
        regions.append({
            "region_id": region_id, "chrom": representative.chrom,
            "start": start, "end": end, "center": int(representative.center),
            "length_bp": end - start,
            "representative_window_id": representative.window_id,
            "best_p": float(representative.best_p),
            "best_scale": representative.scale, "best_branch": representative.branch,
            **{f"from_{scale}_{branch}": f"{scale}_{branch}" in channels
               for scale, branch in CHANNELS},
            "n_support_channels": len(channels),
            "n_support_proposals": len(members),
            "rep1_support": int(members.rep1_supported.sum()),
            "rep2_support": int(members.rep2_supported.sum()),
            "zero_axis_flag": bool(members.zero_axis_flag.any()),
        })
    return pd.DataFrame(regions).sort_values(["chrom", "start", "end", "region_id"]).reset_index(drop=True)


def final_engineering_gate(
    candidates: pd.DataFrame, *, protocol_frozen: bool, artifacts_complete: bool,
    replicates_accessible: bool, no_leakage: bool, known_label_tuning: bool,
) -> dict[str, object]:
    valid = bool(
        not candidates.empty
        and {"region_id", "chrom", "start", "end", "center"}.issubset(candidates.columns)
        and candidates.region_id.is_unique
        and np.isfinite(candidates[["start", "end", "center"]].to_numpy(dtype=float)).all()
        and candidates.end.gt(candidates.start).all()
        and candidates.center.ge(candidates.start).all()
        and candidates.center.lt(candidates.end).all()
    )
    checks = {
        "proposal_protocol_frozen": bool(protocol_frozen),
        "candidate_regions_valid": valid,
        "task2a_artifacts_complete": bool(artifacts_complete),
        "rep1_rep2_accessible": bool(replicates_accessible),
        "no_train_val_calibration_leakage": bool(no_leakage),
        "known_labels_not_used_for_proposals": not known_label_tuning,
    }
    return {"checks": checks, "protocol_frozen": bool(protocol_frozen),
            "candidate_regions_valid": valid, "known_label_tuning": bool(known_label_tuning),
            "ready_for_task2b": bool(all(checks.values()))}
