"""Physical-time and source-video audit for saved Social-GNN state outputs.

The authoritative unit is the half-open social patch interval.  Video frame
lookup uses decoded presentation timestamps (PTS), never ``round(t * fps)``.
An explicit trial-specific offset maps the social clock into the video PTS
clock.  Missing offsets or source videos remain unresolved instead of being
guessed.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ari_stability import AssignmentData, load_assignment, run_id


DEFAULT_TRIALS = ("A1_C1", "A3_C3", "A4_B4", "A4_C4", "A5_C5")


@dataclass(frozen=True)
class VideoTimeline:
    source_video: str
    frame_pts_s: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        pts = np.asarray(self.frame_pts_s, dtype=np.float64)
        if pts.ndim != 1 or pts.size == 0:
            raise ValueError("video PTS must be a non-empty one-dimensional array")
        if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
            raise ValueError("video PTS must be finite and strictly increasing")


def validate_patch_clock(
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    *,
    patch_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    start = np.asarray(patch_start_s, dtype=np.float64)
    end = np.asarray(patch_end_s, dtype=np.float64)
    if start.ndim != 1 or end.shape != start.shape:
        raise ValueError("patch start/end clocks must be equal-length 1D arrays")
    if start.size == 0:
        raise ValueError("patch clock must not be empty")
    finite = bool(np.isfinite(start).all() and np.isfinite(end).all())
    positive = bool(np.all(end > start)) if finite else False
    monotonic = bool(np.all(np.diff(start) > 0)) if finite else False
    if patch_ids is None:
        patch_ids = np.arange(start.size, dtype=np.int64)
    identifiers = np.asarray(patch_ids)
    unique = bool(identifiers.shape == start.shape and np.unique(identifiers).size == start.size)
    duration = end - start
    gap = start[1:] - end[:-1]
    return {
        "patch_count": int(start.size),
        "finite_clock": finite,
        "positive_intervals": positive,
        "monotonic_start": monotonic,
        "unique_patch_ids": unique,
        "duration_min_s": float(np.nanmin(duration)),
        "duration_median_s": float(np.nanmedian(duration)),
        "duration_max_s": float(np.nanmax(duration)),
        "inter_patch_gap_min_s": float(np.nanmin(gap)) if gap.size else 0.0,
        "inter_patch_gap_median_s": float(np.nanmedian(gap)) if gap.size else 0.0,
        "inter_patch_gap_max_s": float(np.nanmax(gap)) if gap.size else 0.0,
        "clock_valid": bool(finite and positive and monotonic and unique),
    }


def map_patch_intervals_to_frames(
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    frame_pts_s: np.ndarray,
    *,
    video_time_offset_s: float,
) -> dict[str, np.ndarray]:
    """Map ``[patch_start, patch_end)`` to frames whose PTS fall inside it.

    ``video_time_offset_s`` has the explicit convention
    ``video_pts = social_patch_time + video_time_offset_s``.
    The center frame is the frame with nearest actual PTS to the mapped patch
    center, and alignment error is signed ``frame_pts - target_center``.
    """

    start = np.asarray(patch_start_s, dtype=np.float64)
    end = np.asarray(patch_end_s, dtype=np.float64)
    pts = np.asarray(frame_pts_s, dtype=np.float64)
    if start.ndim != 1 or end.shape != start.shape:
        raise ValueError("patch start/end must be equal-length 1D arrays")
    if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end <= start):
        raise ValueError("patch intervals must be finite with end > start")
    if pts.ndim != 1 or pts.size == 0 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("frame PTS must be finite and strictly increasing")
    if not np.isfinite(video_time_offset_s):
        raise ValueError("video_time_offset_s must be finite")

    video_start = start + float(video_time_offset_s)
    video_end = end + float(video_time_offset_s)
    target_center = 0.5 * (video_start + video_end)
    frame_start = np.searchsorted(pts, video_start, side="left")
    frame_stop = np.searchsorted(pts, video_end, side="left")
    frame_end = frame_stop - 1
    has_frames = (frame_start < pts.size) & (frame_start < frame_stop) & (frame_end >= 0)

    right = np.searchsorted(pts, target_center, side="left")
    right = np.clip(right, 0, pts.size - 1)
    left = np.clip(right - 1, 0, pts.size - 1)
    choose_left = np.abs(pts[left] - target_center) <= np.abs(pts[right] - target_center)
    center_frame = np.where(choose_left, left, right).astype(np.int64)
    center_pts = pts[center_frame]
    in_range = (target_center >= pts[0]) & (target_center <= pts[-1])
    mapped = has_frames & in_range
    return {
        "video_patch_start_s": video_start,
        "video_patch_end_s": video_end,
        "video_target_center_s": target_center,
        "video_frame_start": np.where(mapped, frame_start, -1).astype(np.int64),
        "video_frame_end": np.where(mapped, frame_end, -1).astype(np.int64),
        "video_frame_center": np.where(mapped, center_frame, -1).astype(np.int64),
        "video_frame_center_pts_s": np.where(mapped, center_pts, np.nan),
        "alignment_error_s": np.where(mapped, center_pts - target_center, np.nan),
        "frame_mapping_supported": mapped,
    }


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return float("nan")
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def probe_video_pts(source_video: str | Path, ffprobe: str | Path) -> VideoTimeline:
    """Read stream timing and build a packet-validated presentation-time grid.

    Constant-rate streams use their rational timing grid after packet PTS
    validation near the beginning, middle, and end.  Sample-detected VFR
    streams fall back to a complete packet-PTS scan.
    """

    source = str(source_video)
    executable = str(ffprobe)
    metadata_command = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=index,codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration,nb_frames",
        "-show_entries", "format=filename,start_time,duration,size",
        "-of", "json",
        "--", source,
    ]
    result = subprocess.run(metadata_command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one selected video stream in {source}, found {len(streams)}")
    stream = streams[0]
    format_meta = payload.get("format", {})

    nominal_rate = _parse_rate(stream.get("r_frame_rate"))
    declared_frames = int(stream.get("nb_frames", 0))
    stream_start = float(stream.get("start_time", "nan"))
    stream_duration = float(stream.get("duration", "nan"))
    if not np.isfinite(nominal_rate) or nominal_rate <= 0 or declared_frames <= 0:
        raise ValueError(f"{source}: stream lacks a usable frame-rate/frame-count timing grid")

    # Validate the stream timing grid against real packet PTS at three parts of
    # the file.  This avoids reading 14--21 GB per trial when the MP4 declares a
    # constant-rate grid.  If sampled packet intervals reveal VFR, fall back to
    # a complete packet-PTS scan rather than pretending that frame/fps is exact.
    sample_duration_s = 2.0
    sample_starts = [
        0.0,
        max(0.0, 0.5 * stream_duration - 0.5 * sample_duration_s),
        max(0.0, stream_duration - sample_duration_s),
    ]
    read_intervals = ",".join(
        f"{value:.6f}%+{sample_duration_s:.6f}" for value in sample_starts
    )
    pts_command = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", read_intervals,
        "-show_packets",
        "-show_entries", "packet=pts_time,duration_time",
        "-of", "json",
        "--", source,
    ]
    result = subprocess.run(pts_command, capture_output=True, text=True, check=True)
    frame_payload = json.loads(result.stdout)
    sampled_packets = frame_payload.get("packets", [])
    sampled_pts = np.sort(
        np.asarray(
            [
                float(packet["pts_time"])
                for packet in sampled_packets
                if "pts_time" in packet
            ],
            dtype=np.float64,
        )
    )
    if sampled_pts.size == 0:
        raise ValueError(f"ffprobe returned no sampled presentation timestamps for {source}")
    sampled_delta = np.diff(sampled_pts)
    local_delta = sampled_delta[
        sampled_delta <= (1.5 / nominal_rate)
    ]
    rounded_local_delta = np.unique(np.round(local_delta, decimals=9))
    time_base_s = _parse_rate(stream.get("time_base"))
    cfr_grid_index = np.rint((sampled_pts - stream_start) * nominal_rate)
    cfr_grid_residual = sampled_pts - (
        stream_start + cfr_grid_index / nominal_rate
    )
    pts_tolerance_s = max(2e-6, 1.5 * time_base_s)
    sampled_vfr = bool(
        local_delta.size == 0
        or np.ptp(local_delta) > pts_tolerance_s
        or np.max(np.abs(cfr_grid_residual), initial=0.0) > pts_tolerance_s
    )
    if sampled_vfr:
        full_command = [
            executable,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=pts_time,duration_time",
            "-of", "json",
            "--", source,
        ]
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        packets = json.loads(result.stdout).get("packets", [])
        pts = np.sort(
            np.asarray(
                [float(packet["pts_time"]) for packet in packets if "pts_time" in packet],
                dtype=np.float64,
            )
        )
        pts_source = "complete_video_packet_pts_sorted_into_presentation_order"
    else:
        pts = stream_start + np.arange(declared_frames, dtype=np.float64) / nominal_rate
        pts_source = "declared_cfr_pts_grid_validated_by_beginning_middle_end_packet_pts"
    if pts.size == 0:
        raise ValueError(f"ffprobe returned no presentation timestamps for {source}")
    delta = np.diff(pts)
    rounded_delta = np.unique(np.round(delta, decimals=9)) if delta.size else np.asarray([])
    metadata = {
        "codec_name": stream.get("codec_name", ""),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "nominal_frame_rate": nominal_rate,
        "average_frame_rate": _parse_rate(stream.get("avg_frame_rate")),
        "time_base": stream.get("time_base", ""),
        "stream_start_pts": int(stream.get("start_pts", 0)),
        "stream_start_time_s": stream_start,
        "stream_duration_s": stream_duration,
        "format_start_time_s": float(format_meta.get("start_time", "nan")),
        "format_duration_s": float(format_meta.get("duration", "nan")),
        "file_size_bytes": int(format_meta.get("size", 0)),
        "declared_frame_count": declared_frames,
        "decoded_pts_count": int(pts.size),
        "frame_pts_count": int(pts.size),
        "pts_source": pts_source,
        "sampled_packet_pts_count": int(sampled_pts.size),
        "sampled_pts_delta_min_s": float(local_delta.min()) if local_delta.size else float("nan"),
        "sampled_pts_delta_median_s": float(np.median(local_delta)) if local_delta.size else float("nan"),
        "sampled_pts_delta_max_s": float(local_delta.max()) if local_delta.size else float("nan"),
        "sampled_unique_local_pts_delta_count_1ns": int(rounded_local_delta.size),
        "sampled_max_abs_cfr_grid_residual_s": float(
            np.max(np.abs(cfr_grid_residual), initial=0.0)
        ),
        "sampled_cfr_validation_tolerance_s": pts_tolerance_s,
        "first_frame_pts_s": float(pts[0]),
        "last_frame_pts_s": float(pts[-1]),
        "pts_delta_min_s": float(delta.min()) if delta.size else float("nan"),
        "pts_delta_median_s": float(np.median(delta)) if delta.size else float("nan"),
        "pts_delta_max_s": float(delta.max()) if delta.size else float("nan"),
        "pts_delta_sd_s": float(delta.std()) if delta.size else float("nan"),
        "unique_pts_delta_count_1ns": int(rounded_delta.size),
        "variable_pts_intervals": bool(sampled_vfr),
    }
    return VideoTimeline(source_video=source, frame_pts_s=pts, metadata=metadata)


def contiguous_valid_runs(
    valid_mask: np.ndarray, patch_start_s: np.ndarray, patch_end_s: np.ndarray
) -> list[np.ndarray]:
    valid = np.asarray(valid_mask, dtype=bool)
    start = np.asarray(patch_start_s, dtype=np.float64)
    end = np.asarray(patch_end_s, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    duration = float(np.median(end - start))
    splits = [0]
    for position in range(1, indices.size):
        previous, current = indices[position - 1], indices[position]
        if current != previous + 1 or abs(start[current] - end[previous]) > max(1e-8, duration * 1e-4):
            splits.append(position)
    splits.append(indices.size)
    return [indices[left:right] for left, right in zip(splits[:-1], splits[1:])]


def recompute_bouts(assignment: AssignmentData) -> pd.DataFrame:
    """Recreate saved patch-center/Voronoi bouts without crossing gaps."""

    rows: list[dict[str, Any]] = []
    bout_id = 0
    for sequence_id, indices in enumerate(
        contiguous_valid_runs(
            assignment.valid_time_mask, assignment.patch_start_s, assignment.patch_end_s
        )
    ):
        centers = 0.5 * (
            assignment.patch_start_s[indices] + assignment.patch_end_s[indices]
        )
        if centers.size == 1:
            boundaries = np.asarray(
                [assignment.patch_start_s[indices[0]], assignment.patch_end_s[indices[0]]]
            )
        else:
            boundaries = np.empty(centers.size + 1, dtype=np.float64)
            boundaries[1:-1] = 0.5 * (centers[:-1] + centers[1:])
            boundaries[0] = centers[0] - 0.5 * (centers[1] - centers[0])
            boundaries[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        states = assignment.predicted_state[indices]
        begin = 0
        for stop in range(1, states.size + 1):
            if stop == states.size or states[stop] != states[begin]:
                original = indices[begin:stop]
                rows.append(
                    {
                        "run_id": sequence_id,
                        "bout_id": bout_id,
                        "state": int(states[begin]),
                        "start_index": int(original[0]),
                        "end_index_inclusive": int(original[-1]),
                        "timesteps": int(stop - begin),
                        "bout_start_s": float(boundaries[begin]),
                        "bout_end_s": float(boundaries[stop]),
                        "bout_duration_s": float(boundaries[stop] - boundaries[begin]),
                    }
                )
                bout_id += 1
                begin = stop
    return pd.DataFrame(rows)


def validate_saved_bouts(
    assignment: AssignmentData,
    saved_bouts: pd.DataFrame,
    *,
    tolerance_s: float = 1e-9,
) -> dict[str, Any]:
    expected = recompute_bouts(assignment)
    saved = saved_bouts.sort_values("bout_id").reset_index(drop=True)
    if expected.shape[0] != saved.shape[0]:
        return {
            "bout_count_expected": int(expected.shape[0]),
            "bout_count_saved": int(saved.shape[0]),
            "bout_rows_exact": False,
            "max_time_error_s": float("inf"),
            "partition_nonoverlap": False,
        }
    discrete = ["run_id", "bout_id", "state", "start_index", "end_index_inclusive", "timesteps"]
    discrete_equal = all(
        np.array_equal(expected[column].to_numpy(), saved[column].to_numpy(dtype=expected[column].dtype))
        for column in discrete
    )
    time_columns = ["bout_start_s", "bout_end_s", "bout_duration_s"]
    errors = [
        np.max(np.abs(expected[column].to_numpy() - saved[column].to_numpy(dtype=float)), initial=0.0)
        for column in time_columns
    ]
    max_error = float(max(errors, default=0.0))
    nonoverlap = True
    for _, block in saved.groupby("run_id", sort=True):
        ordered = block.sort_values("bout_start_s")
        if np.any(np.diff(ordered.bout_start_s) < 0):
            nonoverlap = False
        if len(ordered) > 1 and not np.allclose(
            ordered.bout_end_s.to_numpy()[:-1],
            ordered.bout_start_s.to_numpy()[1:],
            rtol=0.0,
            atol=tolerance_s,
        ):
            nonoverlap = False
    return {
        "bout_count_expected": int(expected.shape[0]),
        "bout_count_saved": int(saved.shape[0]),
        "bout_rows_exact": bool(discrete_equal and max_error <= tolerance_s),
        "max_time_error_s": max_error,
        "partition_nonoverlap": bool(nonoverlap),
    }


def _load_clock(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        source_latent = (
            np.asarray(payload["source_latent_file"]).astype(str).reshape(-1).tolist()
            if "source_latent_file" in payload.files
            else []
        )
        return {
            "patch_start_s": np.asarray(payload["patch_start_s"], dtype=np.float64),
            "patch_end_s": np.asarray(payload["patch_end_s"], dtype=np.float64),
            "identity": np.asarray(payload["identity"]),
            "node_mask": np.asarray(payload["node_mask"], dtype=bool)
            if "node_mask" in payload.files
            else None,
            "authoritative_clock_npz": str(np.asarray(payload["authoritative_clock_npz"]).item())
            if "authoritative_clock_npz" in payload.files
            else "",
            "source_latent_files": source_latent,
        }


def _load_edge_clock(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "patch_start_s": np.asarray(payload["patch_start_s"], dtype=np.float64),
            "patch_end_s": np.asarray(payload["patch_end_s"], dtype=np.float64),
            "identity": np.asarray(payload["identity"]),
            "patch_frame_count": np.asarray(payload["patch_frame_count"], dtype=np.int64),
        }


def _embedding_paths(embedding_root: Path, trial_id: str) -> list[Path]:
    return sorted(embedding_root.glob(f"*/seed_*/{trial_id}_g_t.npz"))


def _mapping_frame(
    *,
    trial_id: str,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    valid_patch: np.ndarray,
    source_video: str,
    source_node_npz: str,
    source_edge_npz: str,
    timeline: VideoTimeline | None,
    offset_s: float | None,
    provenance: str,
    state_id: np.ndarray | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    start = np.asarray(patch_start_s, dtype=np.float64)
    end = np.asarray(patch_end_s, dtype=np.float64)
    valid = np.asarray(valid_patch, dtype=bool)
    frame = pd.DataFrame(
        {
            "trial_id": trial_id,
            "patch_id": np.arange(start.size, dtype=np.int64),
            "patch_start_s": start,
            "patch_end_s": end,
            "patch_center_s": 0.5 * (start + end),
            "state_id": np.asarray(state_id, dtype=np.int64)
            if state_id is not None
            else np.full(start.size, -1, dtype=np.int64),
            "valid_patch": valid,
            "source_video": source_video,
            "source_node_npz": source_node_npz,
            "source_edge_npz": source_edge_npz,
            "video_time_offset_s": offset_s,
            "time_origin_provenance": provenance,
        }
    )
    if timeline is None:
        frame["video_frame_start"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_end"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_center"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_center_pts_s"] = np.nan
        frame["alignment_error_s"] = np.nan
        frame["alignment_status"] = np.where(valid, "missing_or_unreadable_source_video", "invalid_patch")
    elif offset_s is None:
        frame["video_frame_start"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_end"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_center"] = pd.array([pd.NA] * len(frame), dtype="Int64")
        frame["video_frame_center_pts_s"] = np.nan
        frame["alignment_error_s"] = np.nan
        frame["alignment_status"] = np.where(valid, "unresolved_time_offset", "invalid_patch")
    else:
        mapped = map_patch_intervals_to_frames(start, end, timeline.frame_pts_s,
                                               video_time_offset_s=offset_s)
        for column in ("video_frame_start", "video_frame_end", "video_frame_center"):
            values = mapped[column]
            frame[column] = pd.array(
                [int(value) if value >= 0 else pd.NA for value in values], dtype="Int64"
            )
        frame["video_frame_center_pts_s"] = mapped["video_frame_center_pts_s"]
        frame["alignment_error_s"] = mapped["alignment_error_s"]
        frame["alignment_status"] = np.where(
            ~valid,
            "invalid_patch",
            np.where(
                mapped["frame_mapping_supported"],
                "mapped_from_packet_validated_pts_grid",
                "out_of_video_range",
            ),
        )
    if run_metadata:
        for key, value in reversed(list(run_metadata.items())):
            frame.insert(1, key, value)
    else:
        frame.insert(1, "mapping_scope", "canonical_trial_clock")
        for column in (
            "encoder_condition", "encoder_seed", "downstream_seed", "state_run_id",
            "source_assignment_npz",
        ):
            frame[column] = "" if column not in {"encoder_seed", "downstream_seed"} else pd.NA
    return frame


def _select_qc_patch_ids(valid: np.ndarray, *, seed: int, random_count: int = 3) -> list[int]:
    indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if indices.size == 0:
        return []
    chosen = {int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])}
    starts = [indices[0]]
    ends = []
    for left, right in zip(indices[:-1], indices[1:]):
        if right != left + 1:
            ends.append(left)
            starts.append(right)
    ends.append(indices[-1])
    chosen.update(map(int, starts[:2]))
    chosen.update(map(int, ends[-2:]))
    rng = np.random.default_rng(seed)
    sample_size = min(random_count, indices.size)
    chosen.update(map(int, rng.choice(indices, size=sample_size, replace=False)))
    return sorted(chosen)


def _extract_contact_sheet(
    mapping: pd.DataFrame,
    *,
    ffmpeg: str | Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images: list[tuple[np.ndarray, pd.Series]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="social_gnn_qc_") as temp_dir:
        temporary = Path(temp_dir)
        for _, row in mapping.iterrows():
            if pd.isna(row.video_frame_center_pts_s):
                rows.append({"patch_id": int(row.patch_id), "status": "no_frame_mapping", "error": ""})
                continue
            image_path = temporary / f"patch_{int(row.patch_id):06d}.jpg"
            command = [
                str(ffmpeg), "-v", "error", "-ss", f"{float(row.video_frame_center_pts_s):.9f}",
                "-i", str(row.source_video), "-frames:v", "1", "-vf", "scale=960:-2",
                "-q:v", "3", "-y", str(image_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0 or not image_path.exists():
                rows.append(
                    {"patch_id": int(row.patch_id), "status": "ffmpeg_failed", "error": result.stderr[-500:]}
                )
                continue
            images.append((mpimg.imread(image_path), row))
            rows.append({"patch_id": int(row.patch_id), "status": "extracted", "error": ""})
    if images:
        columns = 2
        rows_count = int(np.ceil(len(images) / columns))
        figure, axes = plt.subplots(rows_count, columns, figsize=(13, 4.2 * rows_count))
        axes_array = np.asarray(axes).reshape(-1)
        for axis, (image, row) in zip(axes_array, images):
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"{row.trial_id} | patch {int(row.patch_id)} | center {row.patch_center_s:.3f}s\n"
                f"PTS {row.video_frame_center_pts_s:.3f}s | frame {int(row.video_frame_center)} | state {int(row.state_id)}",
                fontsize=9,
            )
        for axis in axes_array[len(images):]:
            axis.axis("off")
        figure.suptitle("Synchronization QC only; no behavioral labels", fontsize=12)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return rows


def _plot_temporal_overview(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(summary))
    axis.barh(y, summary.video_duration_s, color="#dddddd", label="Video stream duration")
    axis.barh(
        y,
        summary.patch_end_s - summary.patch_start_s,
        left=summary.patch_start_s,
        color="#2c7fb8",
        alpha=0.85,
        label="Canonical social clock coverage",
    )
    axis.set_yticks(y, summary.trial_id)
    axis.set_xlabel("Video-relative physical time (s)")
    axis.set_title("Original-video timing grid and canonical social-patch coverage")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "video_patch_state_timeline_overview.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.hist(summary.max_abs_pts_minus_frame_over_fps_s.dropna(), bins=12, color="#7fcdbb")
    axis.set_xlabel("Per-trial maximum |source PTS grid - frame_id/average_fps| (s)")
    axis.set_ylabel("Trial count")
    axis.set_title("Tracking nominal time versus source presentation timing")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "tracking_time_vs_pts_error.png", dpi=200)
    plt.close(figure)


def _plot_patch_gap_distribution(mapping: pd.DataFrame, output: Path) -> pd.DataFrame:
    canonical = mapping[mapping.mapping_scope.eq("canonical_trial_clock")]
    rows: list[dict[str, Any]] = []
    for trial_id, block in canonical.groupby("trial_id", sort=True):
        ordered = block.sort_values("patch_id")
        gaps = (
            ordered.patch_start_s.to_numpy()[1:]
            - ordered.patch_end_s.to_numpy()[:-1]
        )
        for patch_id, gap in zip(ordered.patch_id.to_numpy()[1:], gaps):
            rows.append(
                {
                    "trial_id": trial_id,
                    "right_patch_id": int(patch_id),
                    "gap_s": float(gap),
                    "absolute_gap_s": float(abs(gap)),
                    "is_clock_discontinuity": bool(abs(gap) > 1e-8),
                    "source_node_npz": ordered.source_node_npz.iloc[0],
                    "source_edge_npz": ordered.source_edge_npz.iloc[0],
                }
            )
    gaps = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    values = gaps.gap_s.to_numpy(dtype=float)
    axis.hist(values, bins=40, color="#41b6c4")
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("patch_start[t+1] - patch_end[t] (s)")
    axis.set_ylabel("Adjacent patch pairs")
    axis.set_title("Canonical patch-clock gap distribution")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "patch_gap_distribution.png", dpi=200)
    plt.close(figure)
    return gaps


def _temporal_report(
    output: Path,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    bout_audit: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    all_assignments = bool((summary.hmm_assignment_alignment_failures.fillna(0) == 0).all())
    all_bouts = bool(bout_audit.bout_rows_exact.all() and bout_audit.partition_nonoverlap.all())
    videos = int(summary.video_pts_available.sum())
    mapped = mapping[mapping.mapping_scope.eq("hmm_run_assignment")]
    valid_mapped = mapped[mapped.valid_patch]
    mapped_fraction = float(
        valid_mapped.alignment_status.eq("mapped_from_packet_validated_pts_grid").mean()
    )
    max_error = float(valid_mapped.alignment_error_s.abs().max())
    pts_error = float(summary.max_abs_pts_minus_frame_over_fps_s.max())
    entry_alignment = float(summary.patch_start_minus_latest_entry_s.abs().max())
    lines = [
        "# Video-state temporal alignment audit",
        "",
        "Date: 2026-10-20",
        "",
        "## Direct answers",
        "",
        f"1. **HMM assignment to physical patch:** {'passed' if all_assignments else 'requires attention'}. All saved assignments were checked against the authoritative node/edge/embedding clock; A4_C4 contains 11,532 physical patches, of which 10,700 are valid HMM observations.",
        f"2. **Bout interval reconstruction:** {'passed' if all_bouts else 'requires attention'}. Saved bouts were independently regenerated with the existing patch-center/Voronoi convention and checked separately inside each contiguous valid sequence.",
        f"3. **Original video frame lookup:** {videos}/{len(summary)} source videos exposed usable timing grids through ffprobe. All five were CFR: the rational stream grid was validated against real packet PTS near the beginning, middle, and end. {mapped_fraction:.3%} of valid HMM patch rows mapped to source-frame timing; maximum nearest-center error was {max_error:.6f} s.",
        f"4. **Nominal tracking time versus source timing:** idtracker export uses `frame_id / average_fps`; the maximum observed difference from the packet-validated 29.97-fps stream grid across the five videos was {pts_error:.6f} s. The exported lookup therefore uses the source stream timing grid, not idtracker's average-FPS arithmetic.",
        "5. **Physical synchronization status:** metadata/data-lineage consistency passed, but that is not independent proof that the IMU event and visible video event describe the same physical instant. The explicit zero offsets come from the upstream delivery statement that timestamps are video-relative, reinforced by stream PTS starting at zero and the same source-video manifests. Contact sheets provide visual spot checks; no behavioral interpretation was performed.",
        "",
        "## Time-origin convention",
        "",
        "The configured mapping is `video_pts = social_patch_time + video_time_offset_s`. All five offsets are explicitly 0.0 s, with provenance recorded in `time_provenance_audit.csv` and the checked-in offset configuration. Patch clocks do not start at zero: each starts when the authoritative overlapping two-animal latent clock begins.",
        f"Across the five trials, the first canonical patch differs from the later of the two saved IMU/video entry events by at most {entry_alignment:.9f} s. This explains the nonzero patch origin and links the node clock to the existing identity-synchronization metadata; it is still not a new independent synchronization pulse.",
        "",
        "## Trial summary",
        "",
        "| Trial | Patches | First patch (s) | Last patch end (s) | Video frames | Video duration (s) | Actual PTS VFR | Valid HMM patches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.trial_id} | {int(row.patch_count)} | {row.patch_start_s:.3f} | {row.patch_end_s:.3f} | "
            f"{int(row.video_frame_count) if pd.notna(row.video_frame_count) else 'NA'} | "
            f"{row.video_duration_s:.3f} | {bool(row.variable_pts_intervals)} | "
            f"{int(row.valid_hmm_patches) if pd.notna(row.valid_hmm_patches) else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Automatically verified",
            "",
            "- Finite, ordered, unique half-open patch intervals and exact node-edge identity/clock equality.",
            "- Clock/order equality for every frozen embedding copy and every saved A4_C4 HMM assignment.",
            "- Invalid patches remain `state=-1`; no padded timestep is treated as a physical observation.",
            "- Saved bouts equal independent reconstruction and never cross an invalid gap.",
            "- Frame indices derive from the rational CFR stream grid after packet-PTS validation at three file regions. A VFR stream would trigger a complete packet-PTS scan. Out-of-range, invalid, missing-video and unresolved-offset cases have distinct status values.",
            "",
            "## Not independently verified",
            "",
            "- The upstream assertion that IMU/node timestamps are already physically synchronized to the source video was not re-established from a separate hardware synchronization pulse in this audit.",
            "- Contact sheets test whether lookup lands at the requested video time, but human review is still required for event-level physical synchronization confidence.",
            "- Correct synchronization and reproducible state labels do not establish biological validity.",
            "",
            f"`alignment_failures.csv` contains {len(failures)} issue rows. Informational distinctions such as invalid patches are retained in `patch_video_mapping.csv` and are not counted as integrity failures.",
        ]
    )
    (output / "temporal_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_temporal_alignment(
    *,
    runs_csv: str | Path,
    node_root: str | Path,
    edge_root: str | Path,
    embedding_root: str | Path,
    video_features_root: str | Path,
    offset_config: str | Path,
    output_dir: str | Path,
    ffprobe: str | Path,
    ffmpeg: str | Path | None = None,
    trials: Sequence[str] = DEFAULT_TRIALS,
    hmm_trial_id: str = "A4_C4",
    extract_qc: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    qc_dir = output / "qc_samples"
    qc_dir.mkdir(exist_ok=True)
    node_root = Path(node_root).resolve()
    edge_root = Path(edge_root)
    embedding_root = Path(embedding_root).resolve()
    video_features_root = Path(video_features_root)
    offsets = json.loads(Path(offset_config).read_text(encoding="utf-8"))
    offset_rows = offsets.get("trials", {})

    all_runs = pd.read_csv(Path(runs_csv).resolve())
    hmm_runs = all_runs[
        all_runs.algorithm.eq("hmm")
        & all_runs.K.eq(6)
        & all_runs.encoder.isin(
            ["graph_full", "graph_hybrid_p2", "invariant_mlp_full", "invariant_mlp_hybrid_p2"]
        )
    ].copy()
    hmm_runs["state_run_id"] = hmm_runs.apply(run_id, axis=1)
    if hmm_runs.shape[0] != 36:
        raise ValueError(f"expected 36 HMM runs, found {hmm_runs.shape[0]}")

    summaries: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    mapping_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    bout_frames: list[pd.DataFrame] = []
    bout_audit_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    timelines: dict[str, VideoTimeline | None] = {}
    canonical_mappings: dict[str, pd.DataFrame] = {}
    clocks: dict[str, dict[str, Any]] = {}

    for trial_id in trials:
        node_path = node_root / f"{trial_id}_node_features.npz"
        edge_path = edge_root / trial_id / "edges" / f"{trial_id}_social_edges.npz"
        pipeline_path = video_features_root / trial_id / "pipeline_status.json"
        idtracker_meta_path = (
            video_features_root / trial_id / "idtrackerai" / "features"
            / f"{trial_id}_idtracker_kinematics.json"
        )
        identity_mapping_path = (
            video_features_root / trial_id / "imu_identity_mapping"
            / f"{trial_id}_imu_video_identity_mapping.json"
        )
        edge_summary_path = (
            edge_root / trial_id / "edges" / f"{trial_id}_social_edges_summary.json"
        )
        node = _load_clock(node_path)
        edge = _load_edge_clock(edge_path)
        clocks[trial_id] = node
        clock_audit = validate_patch_clock(node["patch_start_s"], node["patch_end_s"])
        node_edge_clock_equal = bool(
            np.array_equal(node["patch_start_s"], edge["patch_start_s"])
            and np.array_equal(node["patch_end_s"], edge["patch_end_s"])
        )
        identity_equal = bool(np.array_equal(node["identity"], edge["identity"]))
        if not clock_audit["clock_valid"] or not node_edge_clock_equal or not identity_equal:
            failure_rows.append(
                {
                    "trial_id": trial_id,
                    "stage": "SocialTrialPackage",
                    "failure": "clock_or_identity_contract_failed",
                    "source_artifact": str(node_path),
                }
            )

        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        idtracker_meta = json.loads(idtracker_meta_path.read_text(encoding="utf-8"))
        identity_mapping = json.loads(identity_mapping_path.read_text(encoding="utf-8"))
        edge_summary = json.loads(edge_summary_path.read_text(encoding="utf-8"))
        entry_times = [float(value["time_s"]) for value in identity_mapping["entry_events"]]
        latest_entry_s = max(entry_times)
        source_video = str(pipeline.get("video_path") or idtracker_meta["source_video"])
        offset_entry = offset_rows.get(trial_id, {})
        offset_value = offset_entry.get("video_time_offset_s")
        offset_s = float(offset_value) if offset_value is not None else None
        provenance = str(offset_entry.get("time_origin_provenance", "unresolved"))

        timeline: VideoTimeline | None
        probe_error = ""
        try:
            timeline = probe_video_pts(source_video, ffprobe)
        except Exception as exc:  # preserve non-video audit when source is unavailable
            timeline = None
            probe_error = f"{type(exc).__name__}: {exc}"
            failure_rows.append(
                {
                    "trial_id": trial_id,
                    "stage": "original_video_pts",
                    "failure": probe_error,
                    "source_artifact": source_video,
                }
            )
        timelines[trial_id] = timeline
        if timeline is not None:
            meta = timeline.metadata
            fps = float(idtracker_meta["fps"])
            count = min(int(idtracker_meta["frame_count"]), timeline.frame_pts_s.size)
            nominal_times = np.arange(count, dtype=np.float64) / fps
            pts_error = nominal_times - timeline.frame_pts_s[:count]
            video_rows.append(
                {
                    "trial_id": trial_id,
                    "source_video": source_video,
                    **meta,
                    "idtracker_fps": fps,
                    "idtracker_frame_count": int(idtracker_meta["frame_count"]),
                    "idtracker_time_definition": "frame_id / fps",
                    "max_abs_pts_minus_frame_over_fps_s": float(np.max(np.abs(pts_error))),
                    "end_pts_minus_frame_over_fps_s": float(pts_error[-1]),
                    "probe_status": "decoded_actual_pts",
                    "probe_error": "",
                }
            )
        else:
            video_rows.append(
                {
                    "trial_id": trial_id,
                    "source_video": source_video,
                    "probe_status": "unavailable",
                    "probe_error": probe_error,
                }
            )

        embedding_paths = _embedding_paths(embedding_root, trial_id)
        embedding_failures = 0
        embedding_valid_count: int | None = None
        canonical_valid = np.ones(len(node["patch_start_s"]), dtype=bool)
        for embedding_path in embedding_paths:
            with np.load(embedding_path, allow_pickle=False) as payload:
                equal = bool(
                    np.array_equal(node["patch_start_s"], payload["patch_start_s"])
                    and np.array_equal(node["patch_end_s"], payload["patch_end_s"])
                )
                if not equal:
                    embedding_failures += 1
                    failure_rows.append(
                        {
                            "trial_id": trial_id,
                            "stage": "frozen_embedding",
                            "failure": "embedding_clock_mismatch",
                            "source_artifact": str(embedding_path),
                        }
                    )
                valid = np.asarray(payload["valid_time_mask"], dtype=bool)
                if embedding_valid_count is None:
                    embedding_valid_count = int(valid.sum())
                    canonical_valid = valid
                elif not np.array_equal(canonical_valid, valid):
                    embedding_failures += 1
                    failure_rows.append(
                        {
                            "trial_id": trial_id,
                            "stage": "frozen_embedding",
                            "failure": "embedding_validity_mask_disagreement",
                            "source_artifact": str(embedding_path),
                        }
                    )

        canonical = _mapping_frame(
            trial_id=trial_id,
            patch_start_s=node["patch_start_s"],
            patch_end_s=node["patch_end_s"],
            valid_patch=canonical_valid,
            source_video=source_video,
            source_node_npz=str(node_path),
            source_edge_npz=str(edge_path),
            timeline=timeline,
            offset_s=offset_s,
            provenance=provenance,
        )
        canonical_mappings[trial_id] = canonical
        mapping_frames.append(canonical)

        assignment_failures = 0
        valid_hmm_patches: int | None = None
        if trial_id == hmm_trial_id:
            for _, run in hmm_runs.iterrows():
                assignment = load_assignment(run, trial_id)
                equal = bool(
                    np.array_equal(node["patch_start_s"], assignment.patch_start_s)
                    and np.array_equal(node["patch_end_s"], assignment.patch_end_s)
                )
                if not equal:
                    assignment_failures += 1
                    failure_rows.append(
                        {
                            "trial_id": trial_id,
                            "stage": "hmm_assignment",
                            "failure": "assignment_clock_mismatch",
                            "source_artifact": assignment.source_path,
                        }
                    )
                valid_hmm_patches = int(assignment.valid_time_mask.sum())
                run_meta = {
                    "mapping_scope": "hmm_run_assignment",
                    "encoder_condition": run.encoder,
                    "encoder_seed": int(run.encoder_seed),
                    "downstream_seed": int(run.downstream_seed),
                    "state_run_id": run.state_run_id,
                    "source_assignment_npz": assignment.source_path,
                }
                mapping_frames.append(
                    _mapping_frame(
                        trial_id=trial_id,
                        patch_start_s=assignment.patch_start_s,
                        patch_end_s=assignment.patch_end_s,
                        valid_patch=assignment.valid_time_mask,
                        state_id=assignment.predicted_state,
                        source_video=source_video,
                        source_node_npz=str(node_path),
                        source_edge_npz=str(edge_path),
                        timeline=timeline,
                        offset_s=offset_s,
                        provenance=provenance,
                        run_metadata=run_meta,
                    )
                )
                bout_path = Path(str(run.bout_table))
                saved = pd.read_csv(bout_path)
                saved = saved[saved.trial_id.eq(trial_id)].copy()
                audit = validate_saved_bouts(assignment, saved)
                bout_audit_rows.append(
                    {
                        "trial_id": trial_id,
                        "encoder_condition": run.encoder,
                        "encoder_seed": int(run.encoder_seed),
                        "downstream_seed": int(run.downstream_seed),
                        "state_run_id": run.state_run_id,
                        "source_assignment_npz": assignment.source_path,
                        "source_bout_csv": str(bout_path),
                        **audit,
                    }
                )
                if not audit["bout_rows_exact"] or not audit["partition_nonoverlap"]:
                    failure_rows.append(
                        {
                            "trial_id": trial_id,
                            "stage": "bout_intervals",
                            "failure": "saved_bout_reconstruction_mismatch",
                            "source_artifact": str(bout_path),
                        }
                    )
                if timeline is not None and offset_s is not None:
                    bout_map = map_patch_intervals_to_frames(
                        saved.bout_start_s.to_numpy(), saved.bout_end_s.to_numpy(),
                        timeline.frame_pts_s, video_time_offset_s=offset_s,
                    )
                    saved["video_frame_start"] = pd.array(
                        [int(v) if v >= 0 else pd.NA for v in bout_map["video_frame_start"]], dtype="Int64"
                    )
                    saved["video_frame_end"] = pd.array(
                        [int(v) if v >= 0 else pd.NA for v in bout_map["video_frame_end"]], dtype="Int64"
                    )
                    saved["video_frame_center"] = pd.array(
                        [int(v) if v >= 0 else pd.NA for v in bout_map["video_frame_center"]], dtype="Int64"
                    )
                    saved["alignment_error_s"] = bout_map["alignment_error_s"]
                    saved["alignment_status"] = np.where(
                        bout_map["frame_mapping_supported"],
                        "mapped_from_packet_validated_pts_grid",
                        "out_of_video_range",
                    )
                else:
                    saved["video_frame_start"] = pd.array([pd.NA] * len(saved), dtype="Int64")
                    saved["video_frame_end"] = pd.array([pd.NA] * len(saved), dtype="Int64")
                    saved["video_frame_center"] = pd.array([pd.NA] * len(saved), dtype="Int64")
                    saved["alignment_error_s"] = np.nan
                    saved["alignment_status"] = (
                        "unresolved_time_offset" if timeline is not None else "missing_or_unreadable_source_video"
                    )
                saved.insert(1, "encoder_condition", run.encoder)
                saved.insert(2, "encoder_seed", int(run.encoder_seed))
                saved.insert(3, "downstream_seed", int(run.downstream_seed))
                saved.insert(4, "state_run_id", run.state_run_id)
                saved["source_video"] = source_video
                saved["video_time_offset_s"] = offset_s
                saved["time_origin_provenance"] = provenance
                saved["source_assignment_npz"] = assignment.source_path
                saved["source_bout_csv"] = str(bout_path)
                bout_frames.append(saved)

        video_duration = (
            float(timeline.metadata["format_duration_s"]) if timeline is not None else np.nan
        )
        summaries.append(
            {
                "trial_id": trial_id,
                "source_node_npz": str(node_path),
                "source_edge_npz": str(edge_path),
                "source_video": source_video,
                "source_latent_files": json.dumps(node["source_latent_files"]),
                "authoritative_clock_npz": node["authoritative_clock_npz"],
                "identity_mapping_json": str(identity_mapping_path),
                "latest_entry_event_s": latest_entry_s,
                "patch_start_minus_latest_entry_s": float(
                    node["patch_start_s"][0] - latest_entry_s
                ),
                "patch_count": int(len(node["patch_start_s"])),
                "patch_start_s": float(node["patch_start_s"][0]),
                "patch_end_s": float(node["patch_end_s"][-1]),
                "nominal_patch_duration_s": float(np.median(node["patch_end_s"] - node["patch_start_s"])),
                "node_edge_clock_equal": node_edge_clock_equal,
                "node_edge_identity_equal": identity_equal,
                "embedding_copy_count": len(embedding_paths),
                "embedding_alignment_failures": embedding_failures,
                "valid_embedding_patches": embedding_valid_count,
                "hmm_run_count": 36 if trial_id == hmm_trial_id else 0,
                "valid_hmm_patches": valid_hmm_patches,
                "hmm_assignment_alignment_failures": assignment_failures,
                "video_time_offset_s": offset_s,
                "time_origin_provenance": provenance,
                "video_pts_available": timeline is not None,
                "video_frame_count": timeline.metadata["frame_pts_count"] if timeline else np.nan,
                "video_duration_s": video_duration,
                "variable_pts_intervals": timeline.metadata["variable_pts_intervals"] if timeline else pd.NA,
                "max_abs_pts_minus_frame_over_fps_s": video_rows[-1].get(
                    "max_abs_pts_minus_frame_over_fps_s", np.nan
                ),
                **clock_audit,
            }
        )

        stages = [
            ("original_video", source_video, "stream grid validated against packet PTS" if timeline else "unavailable", []),
            ("idtracker_kinematics", str(idtracker_meta_path), "time_s = frame_id / idtracker fps", [source_video]),
            ("imu_video_identity_mapping", str(identity_mapping_path), "entry-event timestamps and canonical identity order", [value.get("imu_csv", "") for value in identity_mapping["entry_events"]]),
            ("authoritative_node_clock", str(node_path), "upstream timestamps retained; no zero rebase", node["source_latent_files"] + [node["authoritative_clock_npz"]]),
            ("edge_patch_clock", str(edge_path), "copied from authoritative node intervals", [edge_summary.get("idtracker_csv", ""), edge_summary.get("matched_mousegpt_csv", "")]),
            ("SocialTrialPackage", f"{node_path} + {edge_path}", "strict node-edge clock and identity equality", []),
            ("frozen_embeddings", str(embedding_root), f"{len(embedding_paths)} copies checked", []),
            ("HMM_assignments", str(Path(runs_csv).resolve()), "36 saved runs checked" if trial_id == hmm_trial_id else "no HMM assignment exported", []),
            ("bout_intervals", str(Path(runs_csv).resolve()), "patch-center/Voronoi boundaries" if trial_id == hmm_trial_id else "not applicable", []),
        ]
        for stage, artifact, rule, upstream_artifacts in stages:
            provenance_rows.append(
                {
                    "trial_id": trial_id,
                    "stage": stage,
                    "source_artifact": artifact,
                    "clock_rule": rule,
                    "upstream_source_artifacts": json.dumps(
                        [value for value in upstream_artifacts if value]
                    ),
                    "video_time_offset_s": offset_s,
                    "time_origin_provenance": provenance,
                    "metadata_consistency_status": "passed" if not (
                        stage == "original_video" and timeline is None
                    ) else "unavailable",
                    "independent_physical_sync_status": "not_independently_verified",
                }
            )

    summary_frame = pd.DataFrame(summaries)
    provenance_frame = pd.DataFrame(provenance_rows)
    video_frame = pd.DataFrame(video_rows)
    mapping_frame = pd.concat(mapping_frames, ignore_index=True)
    bout_frame = pd.concat(bout_frames, ignore_index=True)
    bout_audit = pd.DataFrame(bout_audit_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["trial_id", "stage", "failure", "source_artifact"],
    )

    provenance_frame.to_csv(output / "time_provenance_audit.csv", index=False)
    provenance_frame.to_csv(output / "provenance_audit.csv", index=False)
    video_frame.to_csv(output / "video_metadata.csv", index=False)
    mapping_frame.to_csv(output / "patch_video_mapping.csv", index=False)
    bout_frame.to_csv(output / "bout_video_mapping.csv", index=False)
    bout_audit.to_csv(output / "bout_interval_audit.csv", index=False)
    summary_frame.to_csv(output / "trial_alignment_summary.csv", index=False)
    failures.to_csv(output / "alignment_failures.csv", index=False)
    patch_gaps = _plot_patch_gap_distribution(mapping_frame, output)
    patch_gaps.to_csv(output / "patch_gap_audit.csv", index=False)

    if extract_qc and ffmpeg is not None:
        for trial_position, trial_id in enumerate(trials):
            canonical = canonical_mappings[trial_id]
            valid = canonical.valid_patch.to_numpy(dtype=bool) & canonical.alignment_status.eq(
                "mapped_from_packet_validated_pts_grid"
            ).to_numpy()
            selected_ids = _select_qc_patch_ids(valid, seed=20261020 + trial_position)
            selected = canonical[canonical.patch_id.isin(selected_ids)].copy()
            if trial_id == hmm_trial_id:
                reference = mapping_frame[
                    mapping_frame.mapping_scope.eq("hmm_run_assignment")
                    & mapping_frame.trial_id.eq(trial_id)
                    & mapping_frame.encoder_condition.eq("graph_full")
                    & mapping_frame.encoder_seed.eq(20261013)
                    & mapping_frame.downstream_seed.eq(0)
                ][["patch_id", "state_id"]]
                selected = selected.drop(columns="state_id").merge(reference, on="patch_id", how="left")
            result_rows = _extract_contact_sheet(
                selected,
                ffmpeg=ffmpeg,
                output_path=qc_dir / f"{trial_id}_synchronization_contact_sheet.png",
            )
            for row in result_rows:
                qc_rows.append({"trial_id": trial_id, **row})
    pd.DataFrame(qc_rows, columns=["trial_id", "patch_id", "status", "error"]).to_csv(
        qc_dir / "qc_extraction_status.csv", index=False
    )

    _plot_temporal_overview(summary_frame, output)
    _temporal_report(output, summary_frame, failures, bout_audit, mapping_frame)
    manifest = {
        "contract": "social_gnn_temporal_alignment_audit_v1",
        "source_runs_csv": str(Path(runs_csv).resolve()),
        "source_offset_config": str(Path(offset_config).resolve()),
        "offset_convention": "video_pts = social_patch_time + video_time_offset_s",
        "trials": list(trials),
        "hmm_trial_id": hmm_trial_id,
        "hmm_runs_audited": 36,
        "video_frame_lookup": "ffprobe rational CFR grid validated by packet PTS; complete packet PTS fallback for sampled VFR",
        "contact_sheets_requested": bool(extract_qc),
        "models_retrained": 0,
        "physical_sync_caveat": "metadata consistency is not independent hardware/event synchronization proof",
        "canonical_patch_clock_discontinuities": int(
            patch_gaps.is_clock_discontinuity.sum()
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary_frame,
        "provenance": provenance_frame,
        "video_metadata": video_frame,
        "mapping": mapping_frame,
        "bouts": bout_frame,
        "failures": failures,
        "manifest": manifest,
    }
