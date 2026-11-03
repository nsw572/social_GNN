"""Deterministic physical-time clip planning for State Atlas v1.

This module never changes state assignments.  It selects review intervals from
already-decoded bouts and writes extraction metadata/scripts.  Video intervals
are expressed in physical seconds; frame indices are traceability metadata
derived from a previously validated CFR packet-PTS grid.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CLIP_CATEGORIES = (
    "high_confidence",
    "random_representative",
    "short_bout",
    "boundary_adjacent",
)


def _path_available(path: str | Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _overlap_fraction(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    denominator = max(min(left[1] - left[0], right[1] - right[0]), 1e-12)
    return overlap / denominator


def _spread_select(
    candidates: pd.DataFrame,
    *,
    count: int,
    max_overlap_fraction: float,
) -> pd.DataFrame:
    """Select deterministically while rotating across trials when possible."""

    if count < 1 or candidates.empty:
        return candidates.iloc[0:0].copy()
    ordered = candidates.sort_values(
        ["selection_score", "trial_id", "focal_start_s", "candidate_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    queues = {
        trial: frame.reset_index(drop=True)
        for trial, frame in ordered.groupby("trial_id", sort=True)
    }
    cursor = {trial: 0 for trial in queues}
    selected: list[pd.Series] = []
    chosen_intervals: dict[str, list[tuple[float, float]]] = {
        trial: [] for trial in queues
    }
    trials = sorted(queues)
    while len(selected) < count:
        progressed = False
        for trial in trials:
            queue = queues[trial]
            while cursor[trial] < len(queue):
                row = queue.iloc[cursor[trial]]
                cursor[trial] += 1
                interval = (float(row.clip_start_s), float(row.clip_end_s))
                if any(
                    _overlap_fraction(interval, old) > max_overlap_fraction
                    for old in chosen_intervals[trial]
                ):
                    continue
                selected.append(row)
                chosen_intervals[trial].append(interval)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if not selected:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def _clip_bounds(
    frame: pd.DataFrame,
    *,
    context_s: float,
    video_duration_by_trial: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    result["clip_start_s"] = np.maximum(
        0.0, result["focal_start_s"].to_numpy(float) - context_s
    )
    result["clip_end_s"] = result["focal_end_s"].to_numpy(float) + context_s
    for trial, duration in video_duration_by_trial.items():
        mask = result["trial_id"].astype(str).eq(str(trial))
        result.loc[mask, "clip_end_s"] = np.minimum(
            result.loc[mask, "clip_end_s"].to_numpy(float), float(duration)
        )
    result["clip_duration_s"] = result["clip_end_s"] - result["clip_start_s"]
    return result[result["clip_duration_s"] > 0].copy()


def _bout_candidates(
    bouts: pd.DataFrame,
    *,
    max_focal_s: float,
) -> pd.DataFrame:
    value = bouts.copy()
    duration = value["bout_duration_s"].to_numpy(float)
    center = 0.5 * (
        value["bout_start_s"].to_numpy(float)
        + value["bout_end_s"].to_numpy(float)
    )
    focal_duration = np.minimum(duration, max_focal_s)
    value["focal_start_s"] = center - 0.5 * focal_duration
    value["focal_end_s"] = center + 0.5 * focal_duration
    value["candidate_id"] = [
        f"{trial}:bout:{int(bout)}"
        for trial, bout in zip(value.trial_id, value.bout_id)
    ]
    return value


def plan_state_clips(
    *,
    bouts: pd.DataFrame,
    boundaries: pd.DataFrame,
    video_metadata: pd.DataFrame,
    n_states: int,
    clips_per_category: int = 3,
    context_s: float = 1.5,
    max_focal_s: float = 2.0,
    boundary_focal_s: float = 0.4,
    max_overlap_fraction: float = 0.25,
    random_seed: int = 20261101,
    high_posterior_quantile: float = 0.75,
    high_pose_quantile: float = 0.75,
    acceptable_dynamic_quantile: float = 0.50,
) -> pd.DataFrame:
    """Build the four requested per-state review categories.

    The selection is deterministic and rotates between trials.  High-confidence
    thresholds are global quantiles over valid bouts; if a state has too few
    qualifying bouts, ranked fallback candidates are used and explicitly
    flagged in ``selection_note``.
    """

    required = {
        "trial_id",
        "bout_id",
        "state",
        "bout_start_s",
        "bout_end_s",
        "bout_duration_s",
        "mean_state_confidence",
        "median_dynamic_quality",
        "median_pose_quality",
    }
    missing = sorted(required.difference(bouts.columns))
    if missing:
        raise ValueError(f"bout table is missing columns: {missing}")
    videos = video_metadata.set_index("trial_id", drop=False)
    duration_by_trial = {
        str(row.trial_id): float(row.video_duration_s)
        for row in videos.itertuples()
    }
    base = _clip_bounds(
        _bout_candidates(bouts, max_focal_s=max_focal_s),
        context_s=context_s,
        video_duration_by_trial=duration_by_trial,
    )
    valid_quality = base[
        np.isfinite(base.mean_state_confidence)
        & np.isfinite(base.median_pose_quality)
        & np.isfinite(base.median_dynamic_quality)
    ]
    thresholds = {
        "posterior": float(valid_quality.mean_state_confidence.quantile(high_posterior_quantile)),
        "pose": float(valid_quality.median_pose_quality.quantile(high_pose_quantile)),
        "dynamic": float(valid_quality.median_dynamic_quality.quantile(acceptable_dynamic_quantile)),
    }
    rng = np.random.default_rng(random_seed)
    selections: list[pd.DataFrame] = []
    for state in range(n_states):
        state_bouts = base[base.state.astype(int).eq(state)].copy()
        if state_bouts.empty:
            continue

        clean = state_bouts[
            (state_bouts.mean_state_confidence >= thresholds["posterior"])
            & (state_bouts.median_pose_quality >= thresholds["pose"])
            & (state_bouts.median_dynamic_quality >= thresholds["dynamic"])
        ].copy()
        clean["selection_score"] = (
            clean.mean_state_confidence
            + clean.median_pose_quality
            + clean.median_dynamic_quality
        ) / 3.0
        clean["selection_note"] = "met_global_clean_measurement_thresholds"
        if len(clean) < clips_per_category:
            fallback = state_bouts.loc[~state_bouts.index.isin(clean.index)].copy()
            fallback["selection_score"] = (
                fallback.mean_state_confidence.fillna(0)
                + fallback.median_pose_quality.fillna(0)
                + fallback.median_dynamic_quality.fillna(0)
            ) / 3.0
            fallback["selection_note"] = "ranked_fallback_below_one_or_more_global_thresholds"
            clean = pd.concat([clean, fallback], ignore_index=False)
        chosen = _spread_select(
            clean, count=clips_per_category, max_overlap_fraction=max_overlap_fraction
        )
        chosen["category"] = "high_confidence"
        selections.append(chosen)

        random_candidates = state_bouts.copy()
        random_candidates["selection_score"] = rng.random(len(random_candidates))
        random_candidates["selection_note"] = "deterministic_random_bout_sample"
        chosen = _spread_select(
            random_candidates,
            count=clips_per_category,
            max_overlap_fraction=max_overlap_fraction,
        )
        chosen["category"] = "random_representative"
        selections.append(chosen)

        state_q25 = float(state_bouts.bout_duration_s.quantile(0.25))
        short = state_bouts[state_bouts.bout_duration_s <= state_q25 + 1e-12].copy()
        short["selection_score"] = -short.bout_duration_s
        short["selection_note"] = f"state_duration_at_or_below_q25_{state_q25:.6f}s"
        chosen = _spread_select(
            short, count=clips_per_category, max_overlap_fraction=max_overlap_fraction
        )
        chosen["category"] = "short_bout"
        selections.append(chosen)

        state_boundaries = boundaries[
            boundaries.from_state.astype(int).eq(state)
            | boundaries.to_state.astype(int).eq(state)
        ].copy()
        if not state_boundaries.empty:
            center = state_boundaries.boundary_time_s.to_numpy(float)
            state_boundaries["focal_start_s"] = center - 0.5 * boundary_focal_s
            state_boundaries["focal_end_s"] = center + 0.5 * boundary_focal_s
            state_boundaries["candidate_id"] = [
                f"{trial}:boundary:{int(index)}"
                for trial, index in zip(
                    state_boundaries.trial_id, state_boundaries.boundary_index
                )
            ]
            state_boundaries["bout_id"] = -1
            state_boundaries["bout_start_s"] = state_boundaries.boundary_time_s
            state_boundaries["bout_end_s"] = state_boundaries.boundary_time_s
            state_boundaries["bout_duration_s"] = 0.0
            state_boundaries["state"] = state
            state_boundaries["mean_state_confidence"] = state_boundaries[
                "boundary_state_confidence"
            ]
            state_boundaries["median_dynamic_quality"] = state_boundaries[
                "boundary_dynamic_quality"
            ]
            state_boundaries["median_pose_quality"] = state_boundaries[
                "boundary_pose_quality"
            ]
            state_boundaries = _clip_bounds(
                state_boundaries,
                context_s=context_s,
                video_duration_by_trial=duration_by_trial,
            )
            state_boundaries["selection_score"] = (
                state_boundaries.boundary_state_confidence.fillna(0)
            )
            state_boundaries["selection_note"] = "transition_into_or_out_of_state"
            chosen = _spread_select(
                state_boundaries,
                count=clips_per_category,
                max_overlap_fraction=max_overlap_fraction,
            )
            chosen["category"] = "boundary_adjacent"
            selections.append(chosen)

    if not selections:
        return pd.DataFrame()
    catalog = pd.concat(selections, ignore_index=True, sort=False)
    catalog["state"] = catalog.state.astype(int)
    catalog["rank_within_category"] = (
        catalog.groupby(["state", "category"], sort=True).cumcount() + 1
    )
    catalog = catalog.merge(
        video_metadata[
            [
                "trial_id",
                "source_video",
                "video_time_offset_s",
                "nominal_frame_rate",
                "stream_start_time_s",
                "video_duration_s",
                "source_available",
            ]
        ],
        on="trial_id",
        how="left",
        validate="many_to_one",
    )
    catalog["video_clip_start_s"] = (
        catalog.clip_start_s + catalog.video_time_offset_s
    )
    catalog["video_clip_end_s"] = catalog.clip_end_s + catalog.video_time_offset_s
    rate = catalog.nominal_frame_rate.to_numpy(float)
    origin = catalog.stream_start_time_s.to_numpy(float)
    catalog["video_frame_start_reference"] = np.ceil(
        (catalog.video_clip_start_s.to_numpy(float) - origin) * rate - 1e-9
    ).astype(np.int64)
    catalog["video_frame_end_reference"] = (
        np.ceil((catalog.video_clip_end_s.to_numpy(float) - origin) * rate - 1e-9)
        - 1
    ).astype(np.int64)
    catalog["render_status"] = np.where(
        catalog.source_available.astype(bool), "planned_not_rendered", "source_unavailable"
    )
    catalog["physical_time_rule"] = "video_pts=social_time+video_time_offset_s"
    catalog["selection_threshold_posterior"] = thresholds["posterior"]
    catalog["selection_threshold_pose_quality"] = thresholds["pose"]
    catalog["selection_threshold_dynamic_quality"] = thresholds["dynamic"]
    return catalog.sort_values(
        ["state", "category", "rank_within_category"], kind="mergesort"
    ).reset_index(drop=True)


def attach_clip_paths(catalog: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    result = catalog.copy()
    paths: list[str] = []
    for row in result.itertuples():
        filename = (
            f"state_{int(row.state):02d}_{row.category}_"
            f"rank_{int(row.rank_within_category):02d}_{row.trial_id}.mp4"
        )
        paths.append(
            str(
                (
                    output_root
                    / f"state_{int(row.state):02d}"
                    / str(row.category)
                    / filename
                ).resolve()
            )
        )
    result["output_clip"] = paths
    return result


def render_clip_catalog(
    catalog: pd.DataFrame,
    *,
    ffmpeg: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Render individual clips from physical PTS intervals, sequentially."""

    result = catalog.copy()
    statuses: list[str] = []
    for row in result.itertuples():
        source = Path(str(row.source_video))
        output = Path(str(row.output_clip))
        if not _path_available(source):
            statuses.append("source_unavailable")
            continue
        if output.exists() and not overwrite:
            statuses.append("already_exists")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        duration = float(row.video_clip_end_s) - float(row.video_clip_start_s)
        command = [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{float(row.video_clip_start_s):.9f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.9f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        statuses.append("rendered" if completed.returncode == 0 else "ffmpeg_failed")
    result["render_status"] = statuses
    return result


def write_extraction_scripts(
    catalog_path: Path,
    *,
    powershell_path: Path,
    bash_path: Path,
) -> None:
    ps_catalog = str(catalog_path.resolve()).replace("'", "''")
    powershell = f"""param(
  [string]$Ffmpeg = 'D:\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe',
  [string]$Catalog = '{ps_catalog}'
)
$ErrorActionPreference = 'Stop'
$rows = Import-Csv -LiteralPath $Catalog
foreach ($row in $rows) {{
  try {{ $sourceAvailable = Test-Path -LiteralPath $row.source_video }}
  catch {{ $sourceAvailable = $false }}
  if (-not $sourceAvailable) {{
    Write-Warning \"Missing source: $($row.source_video)\"
    continue
  }}
  $parent = Split-Path -Parent $row.output_clip
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  $duration = [double]$row.video_clip_end_s - [double]$row.video_clip_start_s
  & $Ffmpeg -y -hide_banner -loglevel error -ss $row.video_clip_start_s `
    -i $row.source_video -t $duration -map 0:v:0 -an -c:v libx264 `
    -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart $row.output_clip
  if ($LASTEXITCODE -ne 0) {{ throw \"ffmpeg failed for $($row.output_clip)\" }}
}}
"""
    powershell_path.write_text(powershell, encoding="utf-8-sig")
    bash = """#!/usr/bin/env bash
set -euo pipefail
CATALOG=${1:?usage: extract_clips.sh CLIP_CATALOG.csv [FFMPEG]}
FFMPEG=${2:-ffmpeg}
python3 - "$CATALOG" "$FFMPEG" <<'PY'
import csv, pathlib, subprocess, sys
catalog, ffmpeg = sys.argv[1:]
with open(catalog, newline='', encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        source = pathlib.Path(row['source_video'])
        if not source.exists():
            print(f"missing source: {source}", file=sys.stderr)
            continue
        output = pathlib.Path(row['output_clip'])
        output.parent.mkdir(parents=True, exist_ok=True)
        duration = float(row['video_clip_end_s']) - float(row['video_clip_start_s'])
        subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', row['video_clip_start_s'], '-i', str(source), '-t', str(duration),
            '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(output)], check=True)
PY
"""
    bash_path.write_text(bash, encoding="utf-8", newline="\n")


def read_clip_catalog(path: str | Path) -> list[dict[str, str]]:
    """Small dependency-free helper for external tooling."""

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


__all__ = [
    "CLIP_CATEGORIES",
    "attach_clip_paths",
    "plan_state_clips",
    "read_clip_catalog",
    "render_clip_catalog",
    "write_extraction_scripts",
]
