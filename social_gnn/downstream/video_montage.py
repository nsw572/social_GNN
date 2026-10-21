"""Export traceable longest-bout video montages from saved HMM assignments.

This module is deliberately post-hoc: it reads the validated temporal audit,
never fits a model, and never changes an assignment. Source footage is cut by
the packet-PTS-validated frame ranges written by ``temporal_alignment.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_RUNS_ROOT = Path("evaluation_runs/gnn_9dim_downstream_v1")
DEFAULT_FFMPEG = Path(r"D:\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe")
DEFAULT_FFPROBE = Path(r"D:\ffmpeg-7.1-essentials_build\bin\ffprobe.exe")
OUTPUT_FPS = 30
SEPARATOR_FRAMES = 15
SEPARATOR_RGB = (128, 128, 128)
SIGNAL_KEYS = ("YMIN", "YMAX", "UMIN", "UMAX", "VMIN", "VMAX")


def _run(
    command: list[str], *, binary: bool = False, description: str
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(command, capture_output=True, text=not binary)
    if result.returncode:
        stderr = (
            result.stderr
            if isinstance(result.stderr, str)
            else result.stderr.decode(errors="replace")
        )
        raise RuntimeError(
            f"{description} failed ({result.returncode}): {stderr[-4000:]}"
        )
    return result


def canonical_run_directory_name(run_id: str) -> str:
    """Convert ``graph_full|e...|hmm|k6|d0`` to a safe directory name."""

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("_")
    if not value:
        raise ValueError("run_id does not contain a usable directory name")
    return value


def resolve_run(
    audit: pd.DataFrame,
    *,
    run_id: str | None,
    trial_id: str,
    encoder_condition: str | None,
    encoder_seed: int | None,
    downstream_algorithm: str,
    states: int,
    downstream_seed: int | None,
) -> pd.Series:
    """Resolve exactly one already-audited HMM run."""

    if downstream_algorithm != "hmm" or states != 6:
        raise ValueError("the completed temporal audit contains HMM K=6 runs only")
    candidates = audit[audit["trial_id"].astype(str).eq(str(trial_id))].copy()
    if run_id is not None:
        candidates = candidates[
            candidates["state_run_id"].astype(str).eq(str(run_id))
        ]
    else:
        missing = [
            name
            for name, value in (
                ("encoder_condition", encoder_condition),
                ("encoder_seed", encoder_seed),
                ("downstream_seed", downstream_seed),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "provide --run-id or the complete selector: " + ", ".join(missing)
            )
        candidates = candidates[
            candidates["encoder_condition"].astype(str).eq(str(encoder_condition))
            & candidates["encoder_seed"].astype(int).eq(int(encoder_seed))
            & candidates["downstream_seed"].astype(int).eq(int(downstream_seed))
        ]
    if len(candidates) != 1:
        ids = (
            candidates.get("state_run_id", pd.Series(dtype=str))
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"expected exactly one audited run, found {len(candidates)}: {ids}"
        )
    row = candidates.iloc[0]
    if not bool(row["bout_rows_exact"]) or not bool(row["partition_nonoverlap"]):
        raise ValueError(
            f"{row.state_run_id}: bout audit did not pass exact/nonoverlap checks"
        )
    return row


def select_longest_bouts(
    mapping: pd.DataFrame, *, top_n: int, states: int
) -> pd.DataFrame:
    """Select deterministic per-state bouts by duration then earliest start."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {
        "state_run_id",
        "trial_id",
        "state",
        "bout_id",
        "bout_start_s",
        "bout_end_s",
        "bout_duration_s",
        "video_frame_start",
        "video_frame_end",
        "alignment_status",
        "source_video",
        "video_time_offset_s",
    }
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"bout mapping is missing columns: {missing}")
    value = mapping.copy()
    numeric = [
        "state",
        "bout_id",
        "bout_start_s",
        "bout_end_s",
        "bout_duration_s",
        "video_frame_start",
        "video_frame_end",
        "video_time_offset_s",
    ]
    for column in numeric:
        value[column] = pd.to_numeric(value[column], errors="raise")
    value = value[value["state"].between(0, states - 1)].copy()
    selected: list[pd.DataFrame] = []
    for state_id in sorted(value["state"].astype(int).unique()):
        ranked = (
            value[value["state"].astype(int).eq(state_id)]
            .sort_values(
                ["bout_duration_s", "bout_start_s", "bout_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(top_n)
            .copy()
        )
        ranked["rank_by_duration"] = np.arange(
            1, len(ranked) + 1, dtype=np.int64
        )
        selected.append(ranked)
    if not selected:
        raise ValueError("the selected run has no occupied valid states")
    return pd.concat(selected, ignore_index=True)


def add_frame_timing(selected: pd.DataFrame, video: pd.Series) -> pd.DataFrame:
    """Attach validated-CFR PTS and endpoint quantization errors."""

    rate = float(video["nominal_frame_rate"])
    stream_start = float(video["stream_start_time_s"])
    frame_count = int(video["frame_pts_count"])
    if not np.isfinite(rate) or rate <= 0 or frame_count < 1:
        raise ValueError("video metadata has an invalid validated frame grid")
    value = selected.copy()
    start = value["video_frame_start"].to_numpy(dtype=np.int64)
    end = value["video_frame_end"].to_numpy(dtype=np.int64)
    invalid = (start < 0) | (end < start) | (end >= frame_count)
    if invalid.any():
        bad = value.loc[
            invalid,
            ["state", "bout_id", "video_frame_start", "video_frame_end"],
        ]
        raise ValueError(
            "selected clip exceeds validated video bounds:\n"
            + bad.to_string(index=False)
        )
    source_start = stream_start + start / rate
    source_end_exclusive = stream_start + (end + 1) / rate
    target_start = value["bout_start_s"].to_numpy(dtype=float) + value[
        "video_time_offset_s"
    ].to_numpy(dtype=float)
    target_end = value["bout_end_s"].to_numpy(dtype=float) + value[
        "video_time_offset_s"
    ].to_numpy(dtype=float)
    value["source_frame_count"] = end - start + 1
    value["source_first_pts_s"] = source_start
    value["source_end_exclusive_pts_s"] = source_end_exclusive
    value["start_quantization_error_s"] = source_start - target_start
    value["end_quantization_error_s"] = source_end_exclusive - target_end
    value["source_frame_interval_duration_s"] = (
        source_end_exclusive - source_start
    )
    value["frame_span_minus_bout_duration_s"] = (
        value["source_frame_interval_duration_s"] - value["bout_duration_s"]
    )
    return value


def montage_segment_plan(
    frame_counts: Iterable[int],
) -> tuple[list[dict[str, int | str]], list[tuple[int, int]]]:
    """Return an exact frame plan with no leading/trailing separator."""

    counts = [int(value) for value in frame_counts]
    if not counts or any(value < 1 for value in counts):
        raise ValueError("every montage clip must contain at least one frame")
    cursor = 0
    plan: list[dict[str, int | str]] = []
    separators: list[tuple[int, int]] = []
    for index, count in enumerate(counts):
        plan.append(
            {"kind": "clip", "start_frame": cursor, "end_frame": cursor + count - 1}
        )
        cursor += count
        if index + 1 < len(counts):
            separators.append((cursor, cursor + SEPARATOR_FRAMES - 1))
            plan.append(
                {
                    "kind": "separator",
                    "start_frame": cursor,
                    "end_frame": cursor + SEPARATOR_FRAMES - 1,
                }
            )
            cursor += SEPARATOR_FRAMES
    return plan, separators


def _extract_clip(
    *,
    ffmpeg: Path,
    source_video: str,
    row: pd.Series,
    output_path: Path,
    codec: str,
    preset: str,
    crf: int,
    seek_guard_s: float,
) -> None:
    first_pts = float(row["source_first_pts_s"])
    end_pts = float(row["source_end_exclusive_pts_s"])
    expected_frames = int(row["source_frame_count"])
    seek = max(0.0, first_pts - float(seek_guard_s))
    vf = (
        f"trim=start={first_pts:.12f}:end={end_pts:.12f},"
        f"setpts=N/({OUTPUT_FPS}*TB)"
    )
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{seek:.12f}",
        "-copyts",
        "-i",
        source_video,
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-frames:v",
        str(expected_frames),
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(OUTPUT_FPS),
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        "90000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(
        command,
        description=f"extract state {int(row.state)} bout {int(row.bout_id)}",
    )


def _make_gray_separator(
    *,
    ffmpeg: Path,
    width: int,
    height: int,
    output_path: Path,
    codec: str,
    preset: str,
    crf: int,
) -> None:
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x808080:s={width}x{height}:r={OUTPUT_FPS}:d=0.5",
        "-frames:v",
        str(SEPARATOR_FRAMES),
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(OUTPUT_FPS),
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        "90000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(command, description="create neutral-gray separator")


def _write_concat_file(path: Path, segments: Iterable[Path]) -> None:
    lines = []
    for segment in segments:
        escaped = segment.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_segments(ffmpeg: Path, concat_file: Path, output_path: Path) -> None:
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(command, description=f"concatenate {output_path.name}")


def _probe_video(ffprobe: Path, path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration:"
        "format=duration:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(
        _run(command, description=f"decode/probe {path.name}").stdout
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream, found {len(streams)}")
    stream = streams[0]
    frames = payload.get("frames", [])
    pts = np.asarray(
        [float(frame["best_effort_timestamp_time"]) for frame in frames],
        dtype=np.float64,
    )
    return {
        "codec_name": str(stream.get("codec_name", "")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": str(stream.get("pix_fmt", "")),
        "r_frame_rate": str(stream.get("r_frame_rate", "")),
        "avg_frame_rate": str(stream.get("avg_frame_rate", "")),
        "frame_count": int(stream.get("nb_read_frames", len(frames))),
        "decoded_frame_count": int(len(frames)),
        "stream_duration_s": float(stream.get("duration", "nan")),
        "format_duration_s": float(
            payload.get("format", {}).get("duration", "nan")
        ),
        "pts_finite": bool(pts.size and np.isfinite(pts).all()),
        "pts_strictly_increasing": bool(
            pts.size and np.all(np.diff(pts) > 0)
        ),
        "first_pts_s": float(pts[0]) if pts.size else float("nan"),
        "last_pts_s": float(pts[-1]) if pts.size else float("nan"),
    }


def _signal_stats(
    ffmpeg: Path, path: Path, ranges: list[tuple[int, int]]
) -> list[dict[str, int]]:
    if not ranges:
        return []
    expression = "+".join(
        f"between(n\\,{start}\\,{end})" for start, end in ranges
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-vf",
        f"select='{expression}',signalstats,metadata=mode=print",
        "-an",
        "-fps_mode",
        "vfr",
        "-f",
        "null",
        "NUL",
    ]
    stderr = _run(
        command, description=f"separator pixel QC for {path.name}"
    ).stderr
    records: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for line in stderr.splitlines():
        if re.search(r"\bframe:\d+", line):
            if current is not None:
                records.append(current)
            current = {}
        match = re.search(
            r"lavfi\.signalstats\.(YMIN|YMAX|UMIN|UMAX|VMIN|VMAX)=([0-9]+)",
            line,
        )
        if match and current is not None:
            current[match.group(1)] = int(match.group(2))
    if current is not None:
        records.append(current)
    return [
        record for record in records if all(key in record for key in SIGNAL_KEYS)
    ]


def _first_rgb_pixel(ffmpeg: Path, path: Path) -> tuple[int, int, int]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        "select='eq(n\\,0)',scale=1:1:flags=neighbor,format=rgb24",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    stdout = _run(
        command, binary=True, description="decode gray RGB reference"
    ).stdout
    if len(stdout) < 3:
        raise RuntimeError("gray reference produced no RGB pixel")
    return tuple(int(value) for value in stdout[:3])


def _manifest_columns() -> list[str]:
    return [
        "run_id",
        "trial_id",
        "state_id",
        "rank_by_duration",
        "bout_id",
        "bout_start_s",
        "bout_end_s",
        "bout_duration_s",
        "source_video_path",
        "source_frame_start",
        "source_frame_end",
        "source_frame_count",
        "source_first_pts_s",
        "source_end_exclusive_pts_s",
        "start_quantization_error_s",
        "end_quantization_error_s",
        "source_frame_interval_duration_s",
        "frame_span_minus_bout_duration_s",
        "exported_frame_count",
        "exported_clip_duration_s",
        "montage_frame_start",
        "montage_frame_end",
        "montage_start_s",
        "montage_end_s",
        "separator_after_frames",
        "alignment_status",
    ]


def _write_report(
    path: Path,
    *,
    run_id: str,
    trial_id: str,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    warnings: list[str],
) -> None:
    passed = bool(
        len(validation) and validation["status"].eq("PASS").all()
    )
    lines = [
        "# Longest-bout HMM video montage export",
        "",
        f"- Run-local state assignment: `{run_id}`",
        f"- Trial: `{trial_id}`",
        "- Selection: five longest valid continuous bouts per occupied state; duration ties use earliest start.",
        "- Extraction: validated CFR packet-PTS grid and half-open physical bout intervals; idtracker frame time was not used.",
        "- Output: 30 fps H.264/yuv420p, no audio, 15 decoded `#808080` frames between clips.",
        "- Models retrained: 0.",
        f"- Overall QC: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "State IDs are local to this HMM run and have not been assigned behavioral names.",
        "",
        "## Selected bouts",
        "",
    ]
    for state_id, group in selected.groupby("state_id", sort=True):
        lines.extend(
            [
                f"### State {int(state_id)}",
                "",
                "| Rank | Bout | Physical interval (s) | Bout duration (s) | Source frames | Exported duration (s) |",
                "| ---: | ---: | --- | ---: | --- | ---: |",
            ]
        )
        for row in group.sort_values("rank_by_duration").itertuples():
            lines.append(
                f"| {int(row.rank_by_duration)} | {int(row.bout_id)} | "
                f"{row.bout_start_s:.6f}-{row.bout_end_s:.6f} | "
                f"{row.bout_duration_s:.6f} | "
                f"{int(row.source_frame_start)}-{int(row.source_frame_end)} | "
                f"{row.exported_clip_duration_s:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Automatic validation",
            "",
            "| State | Bouts | Separators | Frames | Duration (s) | PTS monotonic | Gray exact | Status |",
            "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in validation.sort_values("state_id").itertuples():
        lines.append(
            f"| {int(row.state_id)} | {int(row.selected_bouts)} | "
            f"{int(row.separator_count)} | {int(row.output_frame_count)} | "
            f"{row.output_duration_s:.6f} | "
            f"{bool(row.pts_strictly_increasing)} | "
            f"{bool(row.gray_frames_exact)} | {row.status} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_longest_bout_montages(
    *,
    runs_root: Path,
    run_id: str | None,
    trial_id: str,
    encoder_condition: str | None,
    encoder_seed: int | None,
    downstream_algorithm: str,
    states: int,
    downstream_seed: int | None,
    top_n: int,
    output_dir: Path | None,
    ffmpeg: Path,
    ffprobe: Path,
    codec: str,
    preset: str,
    crf: int,
    seek_guard_s: float,
) -> Path:
    audit_root = (
        runs_root
        / "comparisons"
        / "temporal_and_state_consistency"
        / "temporal_alignment"
    )
    bout_audit_path = audit_root / "bout_interval_audit.csv"
    mapping_path = audit_root / "bout_video_mapping.csv"
    video_metadata_path = audit_root / "video_metadata.csv"
    for path in (
        bout_audit_path,
        mapping_path,
        video_metadata_path,
        ffmpeg,
        ffprobe,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    audit = pd.read_csv(bout_audit_path)
    resolved = resolve_run(
        audit,
        run_id=run_id,
        trial_id=trial_id,
        encoder_condition=encoder_condition,
        encoder_seed=encoder_seed,
        downstream_algorithm=downstream_algorithm,
        states=states,
        downstream_seed=downstream_seed,
    )
    actual_run_id = str(resolved["state_run_id"])
    usecols = [
        "trial_id",
        "encoder_condition",
        "encoder_seed",
        "downstream_seed",
        "state_run_id",
        "bout_id",
        "state",
        "bout_start_s",
        "bout_end_s",
        "bout_duration_s",
        "video_frame_start",
        "video_frame_end",
        "alignment_error_s",
        "alignment_status",
        "source_video",
        "video_time_offset_s",
        "left_truncated",
        "right_truncated",
    ]
    mapping = pd.read_csv(mapping_path, usecols=usecols, low_memory=False)
    mapping = mapping[
        mapping["state_run_id"].astype(str).eq(actual_run_id)
        & mapping["trial_id"].astype(str).eq(trial_id)
    ].copy()
    if mapping.empty:
        raise ValueError(
            f"{actual_run_id}: no rows in validated bout-video mapping"
        )
    bad_alignment = ~mapping["alignment_status"].eq(
        "mapped_from_packet_validated_pts_grid"
    )
    if bad_alignment.any():
        raise ValueError(
            f"{actual_run_id}: {int(bad_alignment.sum())} bouts lack validated video alignment"
        )
    if mapping["source_video"].nunique() != 1:
        raise ValueError(
            f"{actual_run_id}: expected one source video for {trial_id}"
        )

    videos = pd.read_csv(video_metadata_path)
    candidates = videos[videos["trial_id"].astype(str).eq(trial_id)]
    if len(candidates) != 1:
        raise ValueError(f"{trial_id}: expected one validated video metadata row")
    video = candidates.iloc[0]
    source_video = str(video["source_video"])
    if source_video != str(mapping["source_video"].iloc[0]):
        raise ValueError("source video differs between mapping and video metadata")
    if bool(video["variable_pts_intervals"]):
        raise ValueError(
            "this exporter currently requires the audit's validated CFR frame grid"
        )
    if not Path(source_video).exists():
        raise FileNotFoundError(source_video)

    selected = add_frame_timing(
        select_longest_bouts(mapping, top_n=top_n, states=states), video
    )
    occupied = sorted(selected["state"].astype(int).unique())
    warnings: list[str] = []
    for state_id in occupied:
        count = int(selected["state"].astype(int).eq(state_id).sum())
        if count < top_n:
            warnings.append(
                f"State {state_id} has only {count} valid bouts; requested {top_n}."
            )
    missing_states = sorted(set(range(states)).difference(occupied))
    if missing_states:
        warnings.append(
            f"Unoccupied run-local states were not exported: {missing_states}."
        )

    destination = output_dir or (
        runs_root / "video_review" / canonical_run_directory_name(actual_run_id)
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix="montage_", dir=destination
    ) as temporary:
        temp = Path(temporary)
        gray_path = temp / "separator_gray_15f.mp4"
        _make_gray_separator(
            ffmpeg=ffmpeg,
            width=int(video.width),
            height=int(video.height),
            output_path=gray_path,
            codec=codec,
            preset=preset,
            crf=crf,
        )
        gray_probe = _probe_video(ffprobe, gray_path)
        gray_stats = _signal_stats(
            ffmpeg, gray_path, [(0, SEPARATOR_FRAMES - 1)]
        )
        gray_rgb = _first_rgb_pixel(ffmpeg, gray_path)
        if (
            gray_probe["frame_count"] != SEPARATOR_FRAMES
            or len(gray_stats) != SEPARATOR_FRAMES
            or len(
                {
                    tuple(record[key] for key in SIGNAL_KEYS)
                    for record in gray_stats
                }
            )
            != 1
            or gray_rgb != SEPARATOR_RGB
        ):
            raise RuntimeError(
                "gray separator reference failed QC: "
                f"frames={gray_probe['frame_count']}, "
                f"stats={len(gray_stats)}, rgb={gray_rgb}"
            )
        gray_signature = tuple(gray_stats[0][key] for key in SIGNAL_KEYS)

        total_clips = len(selected)
        completed = 0
        for state_id, group in selected.groupby("state", sort=True):
            group = group.sort_values("rank_by_duration").copy()
            clip_paths: list[Path] = []
            clip_probes: list[dict[str, Any]] = []
            for row in group.itertuples(index=False):
                completed += 1
                clip = temp / (
                    f"state_{int(state_id)}_rank_{int(row.rank_by_duration)}_"
                    f"bout_{int(row.bout_id)}.mp4"
                )
                print(
                    f"[{completed:02d}/{total_clips:02d}] state {int(state_id)} "
                    f"rank {int(row.rank_by_duration)} bout {int(row.bout_id)}",
                    flush=True,
                )
                _extract_clip(
                    ffmpeg=ffmpeg,
                    source_video=source_video,
                    row=pd.Series(row._asdict()),
                    output_path=clip,
                    codec=codec,
                    preset=preset,
                    crf=crf,
                    seek_guard_s=seek_guard_s,
                )
                probe = _probe_video(ffprobe, clip)
                expected = int(row.source_frame_count)
                if (
                    probe["frame_count"] != expected
                    or probe["decoded_frame_count"] != expected
                ):
                    raise RuntimeError(
                        f"state {state_id} bout {row.bout_id}: expected "
                        f"{expected} frames, decoded {probe['decoded_frame_count']}"
                    )
                if not probe["pts_strictly_increasing"]:
                    raise RuntimeError(
                        f"state {state_id} bout {row.bout_id}: nonmonotonic clip PTS"
                    )
                clip_paths.append(clip)
                clip_probes.append(probe)

            frame_counts = [int(probe["frame_count"]) for probe in clip_probes]
            plan, separator_ranges = montage_segment_plan(frame_counts)
            segments: list[Path] = []
            for index, clip in enumerate(clip_paths):
                segments.append(clip)
                if index + 1 < len(clip_paths):
                    segments.append(gray_path)
            concat_file = temp / f"state_{int(state_id)}_concat.txt"
            _write_concat_file(concat_file, segments)
            output = destination / (
                f"state_{int(state_id)}_top{top_n}_longest.mp4"
            )
            _concat_segments(ffmpeg, concat_file, output)
            montage_probe = _probe_video(ffprobe, output)
            expected_frames = sum(frame_counts) + SEPARATOR_FRAMES * (
                len(frame_counts) - 1
            )
            output_gray_stats = _signal_stats(
                ffmpeg, output, separator_ranges
            )
            gray_exact = bool(
                len(output_gray_stats)
                == SEPARATOR_FRAMES * len(separator_ranges)
                and all(
                    tuple(record[key] for key in SIGNAL_KEYS) == gray_signature
                    for record in output_gray_stats
                )
            )
            expected_duration = expected_frames / OUTPUT_FPS
            duration_error = float(
                montage_probe["format_duration_s"] - expected_duration
            )
            checks = {
                "frame_count": montage_probe["frame_count"] == expected_frames,
                "decoded_frame_count": montage_probe["decoded_frame_count"]
                == expected_frames,
                "dimensions": montage_probe["width"] == int(video.width)
                and montage_probe["height"] == int(video.height),
                "fps": montage_probe["r_frame_rate"] == f"{OUTPUT_FPS}/1"
                and montage_probe["avg_frame_rate"] == f"{OUTPUT_FPS}/1",
                "pixel_format": montage_probe["pix_fmt"] == "yuv420p",
                "pts": montage_probe["pts_finite"]
                and montage_probe["pts_strictly_increasing"],
                "duration": abs(duration_error)
                <= (1.0 / OUTPUT_FPS + 1e-6),
                "gray": gray_exact,
                "separator_count": len(separator_ranges)
                == max(0, len(frame_counts) - 1),
                "segment_ends": plan[0]["kind"] == "clip"
                and plan[-1]["kind"] == "clip",
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            validation_rows.append(
                {
                    "run_id": actual_run_id,
                    "trial_id": trial_id,
                    "state_id": int(state_id),
                    "output_video": str(output.resolve()),
                    "selected_bouts": len(frame_counts),
                    "requested_bouts": top_n,
                    "separator_count": len(separator_ranges),
                    "separator_frames_each": SEPARATOR_FRAMES,
                    "gray_frames_expected": SEPARATOR_FRAMES
                    * len(separator_ranges),
                    "gray_frames_observed": len(output_gray_stats),
                    "gray_frames_exact": gray_exact,
                    "output_frame_count": montage_probe["frame_count"],
                    "expected_frame_count": expected_frames,
                    "output_duration_s": montage_probe["format_duration_s"],
                    "expected_duration_s": expected_duration,
                    "duration_error_s": duration_error,
                    "width": montage_probe["width"],
                    "height": montage_probe["height"],
                    "r_frame_rate": montage_probe["r_frame_rate"],
                    "pixel_format": montage_probe["pix_fmt"],
                    "pts_strictly_increasing": montage_probe[
                        "pts_strictly_increasing"
                    ],
                    "no_intro_separator": plan[0]["kind"] == "clip",
                    "no_trailing_separator": plan[-1]["kind"] == "clip",
                    "status": status,
                    "failed_checks": ";".join(
                        key for key, passed in checks.items() if not passed
                    ),
                }
            )

            cursor = 0
            for index, (row, probe) in enumerate(
                zip(group.itertuples(index=False), clip_probes)
            ):
                frames = int(probe["frame_count"])
                start_frame = cursor
                end_frame = cursor + frames - 1
                selected_rows.append(
                    {
                        "run_id": actual_run_id,
                        "trial_id": trial_id,
                        "state_id": int(state_id),
                        "rank_by_duration": int(row.rank_by_duration),
                        "bout_id": int(row.bout_id),
                        "bout_start_s": float(row.bout_start_s),
                        "bout_end_s": float(row.bout_end_s),
                        "bout_duration_s": float(row.bout_duration_s),
                        "source_video_path": source_video,
                        "source_frame_start": int(row.video_frame_start),
                        "source_frame_end": int(row.video_frame_end),
                        "source_frame_count": int(row.source_frame_count),
                        "source_first_pts_s": float(row.source_first_pts_s),
                        "source_end_exclusive_pts_s": float(
                            row.source_end_exclusive_pts_s
                        ),
                        "start_quantization_error_s": float(
                            row.start_quantization_error_s
                        ),
                        "end_quantization_error_s": float(
                            row.end_quantization_error_s
                        ),
                        "source_frame_interval_duration_s": float(
                            row.source_frame_interval_duration_s
                        ),
                        "frame_span_minus_bout_duration_s": float(
                            row.frame_span_minus_bout_duration_s
                        ),
                        "exported_frame_count": frames,
                        "exported_clip_duration_s": frames / OUTPUT_FPS,
                        "montage_frame_start": start_frame,
                        "montage_frame_end": end_frame,
                        "montage_start_s": start_frame / OUTPUT_FPS,
                        "montage_end_s": (end_frame + 1) / OUTPUT_FPS,
                        "separator_after_frames": SEPARATOR_FRAMES
                        if index + 1 < len(group)
                        else 0,
                        "alignment_status": str(row.alignment_status),
                    }
                )
                cursor = end_frame + 1
                if index + 1 < len(group):
                    cursor += SEPARATOR_FRAMES

    selected_manifest = pd.DataFrame(
        selected_rows, columns=_manifest_columns()
    )
    validation = pd.DataFrame(validation_rows)
    selected_manifest.to_csv(destination / "selected_bouts.csv", index=False)
    validation.to_csv(destination / "export_validation.csv", index=False)
    _write_report(
        destination / "export_report.md",
        run_id=actual_run_id,
        trial_id=trial_id,
        selected=selected_manifest,
        validation=validation,
        warnings=warnings,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[
            validation.status.ne("PASS"), ["state_id", "failed_checks"]
        ]
        raise RuntimeError(
            "montage QC failed:\n" + failed.to_string(index=False)
        )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--encoder-condition")
    parser.add_argument("--encoder-seed", type=int)
    parser.add_argument(
        "--downstream-algorithm", default="hmm", choices=("hmm",)
    )
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--downstream-seed", type=int)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--seek-guard-s", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = export_longest_bout_montages(
        runs_root=args.runs_root,
        run_id=args.run_id,
        trial_id=args.trial_id,
        encoder_condition=args.encoder_condition,
        encoder_seed=args.encoder_seed,
        downstream_algorithm=args.downstream_algorithm,
        states=args.states,
        downstream_seed=args.downstream_seed,
        top_n=args.top_n,
        output_dir=args.output_dir,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        seek_guard_s=args.seek_guard_s,
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
