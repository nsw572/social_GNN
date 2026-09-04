"""Run MouseGPT and idtracker.ai video feature extraction as parallel branches.

The branch outputs deliberately retain their native identity labels. Cross-tool
identity reconciliation is a later, independently testable processing stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


REUSABLE_IDTRACKER_KEYS = (
    "number_of_animals",
    "intensity_ths",
    "area_ths",
    "resolution_reduction",
    "roi_list",
    "use_bkg",
    "background_subtraction_stat",
    "number_of_frames_for_background",
    "check_segmentation",
    "track_wo_identities",
    "exclusive_rois",
    "id_image_size",
    "number_of_parallel_workers",
    "frames_per_episode",
    "bounding_box_images_in_ram",
    "knowledge_transfer_folder",
)


@dataclass(frozen=True)
class VideoJob:
    video_id: str
    video_path: Path
    output_dir: Path
    social_patch_count: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _expand_strings(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_strings(item) for key, item in value.items()}
    return value


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required by the orchestrator. Run this script in the deepof environment."
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline config must contain a YAML mapping at its root")
    return _expand_strings(payload)


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError(f"Could not derive a safe video id from {value!r}")
    return cleaned


def build_jobs(
    config: dict[str, Any], replacement_videos: Iterable[str] | None = None
) -> list[VideoJob]:
    output_root = Path(config["output_root"]).resolve()
    raw_videos: list[Any]
    if replacement_videos:
        raw_videos = list(replacement_videos)
    else:
        raw_videos = config.get("videos") or []
    if not raw_videos:
        raise ValueError("No videos configured")

    jobs: list[VideoJob] = []
    seen_ids: set[str] = set()
    for item in raw_videos:
        if isinstance(item, str):
            video_path = Path(item).resolve()
            video_id = _safe_id(video_path.stem)
        elif isinstance(item, dict) and "path" in item:
            video_path = Path(item["path"]).resolve()
            video_id = _safe_id(str(item.get("id") or video_path.stem))
        else:
            raise ValueError(f"Invalid video entry: {item!r}")
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if video_id in seen_ids:
            raise ValueError(
                f"Duplicate video id {video_id!r}; assign explicit unique ids in the YAML"
            )
        seen_ids.add(video_id)
        social_patch_count = (
            int(item["social_patch_count"])
            if isinstance(item, dict) and item.get("social_patch_count") is not None
            else None
        )
        if social_patch_count is not None and social_patch_count < 1:
            raise ValueError("social_patch_count must be >= 1")
        jobs.append(
            VideoJob(
                video_id,
                video_path,
                output_root / video_id,
                social_patch_count=social_patch_count,
            )
        )
    return jobs


def _require_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    mouse = config.get("mousegpt", {})
    if mouse.get("enabled", True):
        _require_file(mouse["python"], "MouseGPT Python")
        _require_dir(mouse["workdir"], "MouseGPT workdir")
        _require_file(mouse["inferencer_script"], "MouseGPT inferencer")
        _require_file(mouse["pose_config"], "MouseGPT pose config")
        _require_file(mouse["pose_weights"], "MouseGPT pose weights")
        _require_file(mouse["det_config"], "MouseGPT detector config")
        _require_file(mouse["det_weights"], "MouseGPT detector weights")
        post = mouse.get("postprocess", {})
        if post.get("enabled", True):
            _require_file(post["python"], "MouseGPT postprocess Python")
            _require_file(post["script"], "MouseGPT postprocess script")

    tracker = config.get("idtrackerai", {})
    if tracker.get("enabled", True):
        _require_file(tracker["python"], "idtracker.ai Python")
        if tracker.get("source_dir"):
            _require_dir(tracker["source_dir"], "idtracker.ai source directory")
        exporter = tracker.get("export", {})
        if exporter.get("enabled", True):
            _require_file(exporter["script"], "idtracker.ai export script")
        smoothing = tracker.get("smoothing", {})
        if smoothing.get("enabled", False):
            if not exporter.get("enabled", True):
                raise ValueError("idtracker.ai smoothing requires kinematics export")
            if int(tracker.get("number_of_animals", 2)) != 2:
                raise ValueError(
                    "idtracker.ai extra smoothing currently supports exactly two animals"
                )
            _require_file(smoothing["python"], "idtracker.ai smoothing Python")
            _require_file(smoothing["script"], "idtracker.ai smoothing script")
        policy = tracker.get("gui_policy", "first_video")
        if policy not in {"first_video", "every_video", "never"}:
            raise ValueError(
                "idtrackerai.gui_policy must be first_video, every_video, or never"
            )
        bootstrap = tracker.get("bootstrap_parameters_file")
        if bootstrap:
            _require_file(bootstrap, "idtracker.ai bootstrap parameter profile")

    matching = config.get("identity_matching", {})
    if matching.get("enabled", False):
        if not mouse.get("enabled", True) or not mouse.get("postprocess", {}).get(
            "enabled", True
        ):
            raise ValueError("identity matching requires MouseGPT postprocessing")
        if not tracker.get("enabled", True) or not tracker.get("export", {}).get(
            "enabled", True
        ):
            raise ValueError("identity matching requires idtracker.ai kinematics export")
        _require_file(matching["python"], "identity matching Python")
        _require_file(matching["script"], "identity matching script")
        visualization = matching.get("visualization", {})
        if visualization.get("enabled", False):
            _require_file(
                visualization["script"], "identity matching visualization script"
            )

    edge_extraction = config.get("edge_extraction", {})
    if edge_extraction.get("enabled", False):
        if not matching.get("enabled", False):
            raise ValueError("edge extraction requires identity matching")
        _require_file(edge_extraction["python"], "edge extraction Python")
        _require_file(edge_extraction["script"], "edge extraction script")
        patch_length_s = edge_extraction.get("patch_length_s")
        if patch_length_s is None or float(patch_length_s) <= 0:
            raise ValueError(
                "edge_extraction.patch_length_s must be set to the minimum "
                "upstream node patch duration before enabling edge extraction"
            )


def idtracker_manual_for_job(
    *,
    policy: str,
    job_index: int,
    reusable_profile_exists: bool,
    force_recalibrate: bool,
) -> bool:
    if policy == "every_video":
        return True
    if policy == "never":
        return False
    if policy != "first_video":
        raise ValueError(f"Unknown GUI policy: {policy}")
    return job_index == 0 and (force_recalibrate or not reusable_profile_exists)


def _append_cli_value(command: list[str], option: str, value: Any) -> None:
    if value is None:
        return
    command.append(option)
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    elif isinstance(value, bool):
        command.append("true" if value else "false")
    else:
        command.append(str(value))


def build_mousegpt_command(
    config: dict[str, Any], job: VideoJob, prediction_dir: Path, vis_dir: Path
) -> list[str]:
    mouse = config["mousegpt"]
    command = [
        str(Path(mouse["python"]).resolve()),
        str(Path(mouse["inferencer_script"]).resolve()),
        str(job.video_path),
        "--pose2d",
        str(Path(mouse["pose_config"]).resolve()),
        "--pose2d-weights",
        str(Path(mouse["pose_weights"]).resolve()),
        "--det-model",
        str(Path(mouse["det_config"]).resolve()),
        "--det-weights",
        str(Path(mouse["det_weights"]).resolve()),
        "--vis-out-dir",
        str(vis_dir) if mouse.get("save_visualization", True) else "",
        "--pred-out-dir",
        str(prediction_dir),
        "--bbox-thr",
        str(mouse.get("bbox_threshold", 0.6)),
        "--nms-thr",
        str(mouse.get("nms_threshold", 0.1)),
        "--kpt-thr",
        str(mouse.get("keypoint_threshold", 0.3)),
    ]
    if mouse.get("device"):
        command.extend(["--device", str(mouse["device"])])
    if mouse.get("draw_bbox", True):
        command.append("--draw-bbox")
    command.extend(str(item) for item in mouse.get("extra_args", []))
    return command


def build_mousegpt_postprocess_command(
    config: dict[str, Any], job: VideoJob, prediction_json: Path, output_dir: Path
) -> list[str]:
    post = config["mousegpt"].get("postprocess", {})
    command = [
        str(Path(post["python"]).resolve()),
        str(Path(post["script"]).resolve()),
        "--video",
        str(job.video_path),
        "--predictions-json",
        str(prediction_json),
        "--outdir",
        str(output_dir),
        "--score-threshold",
        str(post.get("score_threshold", 0.3)),
        "--max-match-distance",
        str(post.get("max_match_distance", 700.0)),
        "--max-track-gap-frames",
        str(post.get("max_track_gap_frames", 30)),
        "--max-tracks",
        str(post.get("max_tracks", 2)),
    ]
    if post.get("skip_videos", True):
        command.append("--skip-videos")
    command.extend(str(item) for item in post.get("extra_args", []))
    return command


def build_idtracker_command(
    config: dict[str, Any],
    job: VideoJob,
    output_dir: Path,
    *,
    manual_gui: bool,
    parameter_profile: Path | None,
) -> list[str]:
    tracker = config["idtrackerai"]
    command = [str(Path(tracker["python"]).resolve()), "-m", "idtrackerai.start"]
    if parameter_profile is not None:
        command.extend(["--load", str(parameter_profile)])
    if not manual_gui:
        command.append("--track")
    _append_cli_value(command, "--video_paths", [job.video_path])
    _append_cli_value(command, "--name", job.video_id)
    _append_cli_value(command, "--output_dir", output_dir)
    _append_cli_value(
        command, "--number_of_animals", tracker.get("number_of_animals", 2)
    )
    _append_cli_value(
        command,
        "--trajectories_formats",
        tracker.get("trajectories_formats", ["npy", "csv_tidy"]),
    )
    _append_cli_value(command, "--data_policy", tracker.get("data_policy", "all"))
    _append_cli_value(command, "--device", tracker.get("device"))
    _append_cli_value(
        command,
        "--number_of_parallel_workers",
        tracker.get("number_of_parallel_workers"),
    )
    command.extend(str(item) for item in tracker.get("extra_args", []))
    return command


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_process(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    printable = _format_command(command)
    print(f"[{label}] {printable}")
    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "cwd": str(cwd),
            "log": str(log_path),
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"started: {_now_iso()}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {printable}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"\nfinished: {_now_iso()}\nexit_code: {completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}; see {log_path}"
        )
    return {
        "status": "completed",
        "command": command,
        "cwd": str(cwd),
        "log": str(log_path),
        "exit_code": completed.returncode,
    }


def _resolve_prediction_json(prediction_dir: Path, video_stem: str) -> Path:
    exact = prediction_dir / f"{video_stem}.json"
    if exact.is_file():
        return exact
    matches = sorted(prediction_dir.rglob(f"{video_stem}.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"MouseGPT prediction JSON for {video_stem!r} not found under {prediction_dir}"
        )
    raise RuntimeError(
        f"Multiple MouseGPT prediction JSON files found for {video_stem!r}: {matches}"
    )


def _session_complete(session_dir: Path) -> bool:
    trajectories_dir = session_dir / "trajectories"
    return (session_dir / "session.json").is_file() and trajectories_dir.is_dir() and any(
        path.is_file() for path in trajectories_dir.rglob("*")
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


def extract_reusable_idtracker_parameters(
    payload: dict[str, Any]
) -> dict[str, Any]:
    """Select and validate video-independent parameters from a session payload."""
    reusable = {
        key: payload[key]
        for key in REUSABLE_IDTRACKER_KEYS
        if key in payload and payload[key] is not None
    }
    required = {"intensity_ths", "area_ths", "number_of_animals"}
    missing = sorted(required - reusable.keys())
    if missing:
        raise ValueError(
            f"Completed idtracker.ai session lacks reusable parameters: {missing}"
        )
    return reusable


def snapshot_reusable_idtracker_parameters(
    session_json: Path, destination: Path
) -> dict[str, Any]:
    """Write video-independent GUI parameters from a completed session to TOML."""
    payload = json.loads(session_json.read_text(encoding="utf-8"))
    reusable = extract_reusable_idtracker_parameters(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated from a completed idtracker.ai GUI session.",
        "# Video path, session name, output directory, and tracking interval are intentionally excluded.",
    ]
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in reusable.items())
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reusable


def _idtracker_environment(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    source = config["idtrackerai"].get("source_dir")
    if source:
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(Path(source).resolve()) + (
            os.pathsep + current if current else ""
        )
    return env


def run_mousegpt_branch(
    config: dict[str, Any], job: VideoJob, *, dry_run: bool, resume: bool
) -> dict[str, Any]:
    mouse = config["mousegpt"]
    root = job.output_dir / "mousegpt"
    prediction_dir = root / "raw_predictions"
    vis_dir = root / "visualizations"
    feature_dir = root / "features"
    log_dir = job.output_dir / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if mouse.get("save_visualization", True):
        vis_dir.mkdir(parents=True, exist_ok=True)

    expected_json = prediction_dir / f"{job.video_path.stem}.json"
    inference_result: dict[str, Any]
    if resume and expected_json.is_file():
        inference_result = {
            "status": "skipped_existing",
            "prediction_json": str(expected_json),
        }
    else:
        inference_result = run_process(
            label=f"{job.video_id}:mousegpt",
            command=build_mousegpt_command(
                config, job, prediction_dir, vis_dir
            ),
            cwd=Path(mouse["workdir"]).resolve(),
            log_path=log_dir / "mousegpt_inference.log",
            env=None,
            dry_run=dry_run,
        )

    prediction_json = expected_json if dry_run else _resolve_prediction_json(
        prediction_dir, job.video_path.stem
    )
    post = mouse.get("postprocess", {})
    post_result: dict[str, Any] | None = None
    expected_csv = feature_dir / "all_mousegpt_keypoints_and_head_direction.csv"
    if post.get("enabled", True):
        if resume and expected_csv.is_file():
            post_result = {
                "status": "skipped_existing",
                "features_csv": str(expected_csv),
            }
        else:
            feature_dir.mkdir(parents=True, exist_ok=True)
            post_result = run_process(
                label=f"{job.video_id}:mousegpt-postprocess",
                command=build_mousegpt_postprocess_command(
                    config, job, prediction_json, feature_dir
                ),
                cwd=Path(post["script"]).resolve().parent,
                log_path=log_dir / "mousegpt_postprocess.log",
                env=None,
                dry_run=dry_run,
            )
            if not dry_run and not expected_csv.is_file():
                raise FileNotFoundError(
                    f"MouseGPT postprocess did not create {expected_csv}"
                )

    return {
        "status": "completed" if not dry_run else "dry_run",
        "prediction_json": str(prediction_json),
        "features_csv": str(expected_csv) if post.get("enabled", True) else None,
        "inference": inference_result,
        "postprocess": post_result,
    }


def run_idtracker_branch(
    config: dict[str, Any],
    job: VideoJob,
    *,
    manual_gui: bool,
    reusable_profile: Path,
    bootstrap_profile: Path | None,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    tracker = config["idtrackerai"]
    root = job.output_dir / "idtrackerai"
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / f"session_{job.video_id}"
    log_dir = job.output_dir / "logs"
    complete = _session_complete(session_dir)

    if complete and resume:
        tracking_result: dict[str, Any] = {"status": "skipped_existing"}
    else:
        if session_dir.exists():
            raise FileExistsError(
                f"Incomplete idtracker.ai session already exists: {session_dir}. "
                "Move it aside or finish/remove it explicitly before rerunning."
            )
        if manual_gui:
            parameter_profile = (
                reusable_profile
                if reusable_profile.is_file()
                else bootstrap_profile
            )
        else:
            parameter_profile = reusable_profile
            if not dry_run and not parameter_profile.is_file():
                raise FileNotFoundError(
                    "Headless idtracker.ai run requires a reusable parameter profile: "
                    f"{parameter_profile}"
                )
        tracking_result = run_process(
            label=f"{job.video_id}:idtrackerai",
            command=build_idtracker_command(
                config,
                job,
                root,
                manual_gui=manual_gui,
                parameter_profile=parameter_profile,
            ),
            cwd=root,
            log_path=log_dir / "idtrackerai.log",
            env=_idtracker_environment(config),
            dry_run=dry_run,
        )
        if not dry_run and not _session_complete(session_dir):
            raise RuntimeError(
                "idtracker.ai exited without a completed trajectory session. "
                "The GUI may have been closed without selecting 'Close and track video'."
            )

    parameters_used = root / "parameters_used.toml"
    if not dry_run:
        snapshot_reusable_idtracker_parameters(
            session_dir / "session.json", parameters_used
        )
        if tracker.get("gui_policy", "first_video") == "first_video" and manual_gui:
            snapshot_reusable_idtracker_parameters(
                session_dir / "session.json", reusable_profile
            )

    export_cfg = tracker.get("export", {})
    export_result: dict[str, Any] | None = None
    feature_dir = root / "features"
    output_stem = job.video_id
    expected_csv = feature_dir / f"{output_stem}_idtracker_kinematics.csv"
    expected_npz = feature_dir / f"{output_stem}_idtracker_kinematics.npz"
    if export_cfg.get("enabled", True):
        if resume and expected_csv.is_file() and expected_npz.is_file():
            export_result = {"status": "skipped_existing"}
        else:
            feature_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(Path(tracker["python"]).resolve()),
                str(Path(export_cfg["script"]).resolve()),
                "--session-dir",
                str(session_dir),
                "--video",
                str(job.video_path),
                "--outdir",
                str(feature_dir),
                "--output-stem",
                output_stem,
                "--velocity-method",
                str(export_cfg.get("velocity_method", "backward")),
            ]
            export_result = run_process(
                label=f"{job.video_id}:idtracker-export",
                command=command,
                cwd=Path(export_cfg["script"]).resolve().parent,
                log_path=log_dir / "idtracker_export.log",
                env=_idtracker_environment(config),
                dry_run=dry_run,
            )
            if not dry_run and not (expected_csv.is_file() and expected_npz.is_file()):
                raise FileNotFoundError(
                    "idtracker.ai export did not create its expected CSV/NPZ outputs"
                )

    smoothing_cfg = tracker.get("smoothing", {})
    smoothing_result: dict[str, Any] | None = None
    smoothed_csv = feature_dir / f"{output_stem}_idtracker_kinematics_smoothed.csv"
    smoothed_npz = feature_dir / f"{output_stem}_idtracker_kinematics_smoothed.npz"
    smoothing_events = feature_dir / f"{output_stem}_idtracker_smoothing_events.csv"
    smoothing_summary = feature_dir / f"{output_stem}_idtracker_smoothing_summary.json"
    smoothing_enabled = bool(smoothing_cfg.get("enabled", False))
    if smoothing_enabled:
        smoothing_outputs = (
            smoothed_csv,
            smoothed_npz,
            smoothing_events,
            smoothing_summary,
        )
        if resume and all(path.is_file() for path in smoothing_outputs):
            smoothing_result = {"status": "skipped_existing"}
        else:
            smoothing_result = run_process(
                label=f"{job.video_id}:idtracker-smoothing",
                command=build_idtracker_smoothing_command(
                    config,
                    job,
                    session_dir=session_dir,
                    input_npz=expected_npz,
                    output_dir=feature_dir,
                ),
                cwd=Path(smoothing_cfg["script"]).resolve().parent,
                log_path=log_dir / "idtracker_smoothing.log",
                env=None,
                dry_run=dry_run,
            )
            if not dry_run and not all(path.is_file() for path in smoothing_outputs):
                raise FileNotFoundError(
                    "idtracker.ai smoothing did not create its expected outputs"
                )

    active_csv = smoothed_csv if smoothing_enabled else expected_csv
    active_npz = smoothed_npz if smoothing_enabled else expected_npz

    return {
        "status": "completed" if not dry_run else "dry_run",
        "manual_gui": manual_gui,
        "session_dir": str(session_dir),
        "parameters_used": str(parameters_used),
        "features_csv": str(active_csv) if export_cfg.get("enabled", True) else None,
        "features_npz": str(active_npz) if export_cfg.get("enabled", True) else None,
        "raw_features_csv": str(expected_csv) if export_cfg.get("enabled", True) else None,
        "raw_features_npz": str(expected_npz) if export_cfg.get("enabled", True) else None,
        "extra_smoothing_enabled": smoothing_enabled,
        "tracking": tracking_result,
        "export": export_result,
        "smoothing": smoothing_result,
    }


def build_idtracker_smoothing_command(
    config: dict[str, Any],
    job: VideoJob,
    *,
    session_dir: Path,
    input_npz: Path,
    output_dir: Path,
) -> list[str]:
    """Build the optional CPU-only idtracker.ai postprocessing command."""
    tracker = config["idtrackerai"]
    smoothing = tracker["smoothing"]
    command = [
        str(Path(smoothing["python"]).resolve()),
        str(Path(smoothing["script"]).resolve()),
        "--input-npz",
        str(input_npz),
        "--session-dir",
        str(session_dir),
        "--outdir",
        str(output_dir),
        "--output-stem",
        job.video_id,
        "--velocity-method",
        str(tracker.get("export", {}).get("velocity_method", "backward")),
    ]
    option_map = {
        "speed_multiplier": "--speed-multiplier",
        "robust_mean_upper_quantile": "--robust-mean-upper-quantile",
        "minimum_speed_similarity_ratio": "--minimum-speed-similarity-ratio",
        "maximum_displacement_cosine": "--maximum-displacement-cosine",
        "contact_distance_body_lengths": "--contact-distance-body-lengths",
        "contact_window_frames": "--contact-window-frames",
        "max_seed_gap_frames": "--max-seed-gap-frames",
        "identity_contact_distance_body_lengths": (
            "--identity-contact-distance-body-lengths"
        ),
        "maximum_cross_identity_endpoint_ratio": (
            "--maximum-cross-identity-endpoint-ratio"
        ),
        "maximum_swap_pair_gap_seconds": "--maximum-swap-pair-gap-seconds",
        "transition_speed_cap_multiplier": "--transition-speed-cap-multiplier",
        "maximum_transition_expansion_seconds": (
            "--maximum-transition-expansion-seconds"
        ),
        "interpolation_confidence": "--interpolation-confidence",
    }
    for key, option in option_map.items():
        if key in smoothing:
            command.extend([option, str(smoothing[key])])
    return command


def active_idtracker_features_csv(config: dict[str, Any], job: VideoJob) -> Path:
    suffix = (
        "_idtracker_kinematics_smoothed.csv"
        if config.get("idtrackerai", {}).get("smoothing", {}).get("enabled", False)
        else "_idtracker_kinematics.csv"
    )
    return job.output_dir / "idtrackerai" / "features" / f"{job.video_id}{suffix}"


def _matching_outputs_are_current(summary_json: Path, idtracker_csv: Path) -> bool:
    if not summary_json.is_file():
        return False
    try:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        recorded = Path(summary["idtracker_csv"]).resolve()
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return recorded == idtracker_csv.resolve()


def build_identity_matching_command(
    config: dict[str, Any],
    job: VideoJob,
    mousegpt_csv: Path,
    idtracker_csv: Path,
    output_dir: Path,
) -> list[str]:
    matching = config["identity_matching"]
    return [
        str(Path(matching["python"]).resolve()),
        str(Path(matching["script"]).resolve()),
        "--mousegpt-csv",
        str(mousegpt_csv),
        "--idtracker-csv",
        str(idtracker_csv),
        "--outdir",
        str(output_dir),
        "--output-stem",
        job.video_id,
        "--max-distance-px",
        str(matching.get("max_distance_px", 200.0)),
        "--min-assignment-margin-px",
        str(matching.get("min_assignment_margin_px", 100.0)),
        "--distance-confidence-scale-px",
        str(matching.get("distance_confidence_scale_px", 100.0)),
        "--margin-confidence-scale-px",
        str(matching.get("margin_confidence_scale_px", 300.0)),
        "--min-anchor-confidence",
        str(matching.get("min_anchor_confidence", 0.30)),
    ]


def build_identity_visualization_command(
    config: dict[str, Any],
    job: VideoJob,
    enriched_mousegpt_csv: Path,
    idtracker_csv: Path,
    output_video: Path,
) -> list[str]:
    matching = config["identity_matching"]
    visualization = matching.get("visualization", {})
    command = [
        str(Path(matching["python"]).resolve()),
        str(Path(visualization["script"]).resolve()),
        "--video",
        str(job.video_path),
        "--enriched-mousegpt-csv",
        str(enriched_mousegpt_csv),
        "--idtracker-csv",
        str(idtracker_csv),
        "--output",
        str(output_video),
        "--frame-start",
        str(visualization.get("frame_start", 0)),
        "--max-width",
        str(visualization.get("max_width", 1920)),
        "--max-height",
        str(visualization.get("max_height", 1080)),
        "--codec",
        str(visualization.get("codec", "mp4v")),
    ]
    if visualization.get("frame_end") is not None:
        command.extend(["--frame-end", str(visualization["frame_end"])])
    if not visualization.get("draw_keypoints", True):
        command.append("--no-keypoints")
    if not visualization.get("draw_bboxes", True):
        command.append("--no-bboxes")
    if not visualization.get("draw_head_direction", True):
        command.append("--no-head-direction")
    if visualization.get("opencv_threads") is not None:
        command.extend(
            ["--opencv-threads", str(visualization["opencv_threads"])]
        )
    if visualization.get("low_priority", False):
        command.append("--low-priority")
    return command


def run_identity_matching_stage(
    config: dict[str, Any], job: VideoJob, *, dry_run: bool, resume: bool
) -> dict[str, Any]:
    matching = config["identity_matching"]
    output_dir = job.output_dir / "identity_matching"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = job.output_dir / "logs"
    mousegpt_csv = (
        job.output_dir
        / "mousegpt"
        / "features"
        / "all_mousegpt_keypoints_and_head_direction.csv"
    )
    idtracker_csv = active_idtracker_features_csv(config, job)
    matches_csv = output_dir / f"{job.video_id}_identity_matches.csv"
    enriched_csv = (
        output_dir / f"{job.video_id}_mousegpt_with_idtracker_identity.csv"
    )
    frames_csv = output_dir / f"{job.video_id}_identity_match_frames.csv"
    summary_json = output_dir / f"{job.video_id}_identity_match_summary.json"

    matching_reused = resume and all(
        path.is_file()
        for path in (matches_csv, enriched_csv, frames_csv, summary_json)
    ) and _matching_outputs_are_current(summary_json, idtracker_csv)
    if matching_reused:
        matching_result: dict[str, Any] = {"status": "skipped_existing"}
    else:
        matching_result = run_process(
            label=f"{job.video_id}:identity-matching",
            command=build_identity_matching_command(
                config, job, mousegpt_csv, idtracker_csv, output_dir
            ),
            cwd=Path(matching["script"]).resolve().parent,
            log_path=log_dir / "identity_matching.log",
            env=None,
            dry_run=dry_run,
        )
        if not dry_run and not all(
            path.is_file()
            for path in (matches_csv, enriched_csv, frames_csv, summary_json)
        ):
            raise FileNotFoundError("Identity matcher did not create expected outputs")

    visualization = matching.get("visualization", {})
    visualization_result: dict[str, Any] | None = None
    output_video = output_dir / f"{job.video_id}_identity_match_overlay.mp4"
    if visualization.get("enabled", False):
        if (
            matching_reused
            and output_video.is_file()
            and output_video.stat().st_size > 0
        ):
            visualization_result = {"status": "skipped_existing"}
        else:
            visualization_result = run_process(
                label=f"{job.video_id}:identity-visualization",
                command=build_identity_visualization_command(
                    config, job, enriched_csv, idtracker_csv, output_video
                ),
                cwd=Path(visualization["script"]).resolve().parent,
                log_path=log_dir / "identity_match_visualization.log",
                env=None,
                dry_run=dry_run,
            )
            if not dry_run and (
                not output_video.is_file() or output_video.stat().st_size == 0
            ):
                raise FileNotFoundError(
                    "Identity match visualizer did not create its output video"
                )

    return {
        "status": "dry_run" if dry_run else "completed",
        "matches_csv": str(matches_csv),
        "enriched_mousegpt_csv": str(enriched_csv),
        "frame_summary_csv": str(frames_csv),
        "summary_json": str(summary_json),
        "visualization_video": (
            str(output_video) if visualization.get("enabled", False) else None
        ),
        "matching": matching_result,
        "visualization": visualization_result,
    }


def build_edge_extraction_command(
    config: dict[str, Any],
    job: VideoJob,
    *,
    idtracker_csv: Path,
    matched_mousegpt_csv: Path,
    output_dir: Path,
) -> list[str]:
    edge_cfg = config["edge_extraction"]
    command = [
        str(Path(edge_cfg["python"]).resolve()),
        str(Path(edge_cfg["script"]).resolve()),
        "--idtracker-csv",
        str(idtracker_csv),
        "--matched-mousegpt-csv",
        str(matched_mousegpt_csv),
        "--outdir",
        str(output_dir),
        "--output-stem",
        job.video_id,
        "--patch-length-s",
        str(edge_cfg["patch_length_s"]),
        "--clock-start-s",
        str(edge_cfg.get("clock_start_s", 0.0)),
        "--minimum-movement-speed-px-s",
        str(edge_cfg.get("minimum_movement_speed_px_s", 1.0)),
    ]
    patch_count = (
        job.social_patch_count
        if job.social_patch_count is not None
        else edge_cfg.get("patch_count")
    )
    if patch_count is not None:
        command.extend(["--patch-count", str(patch_count)])
    if not edge_cfg.get("save_frame_level", True):
        command.append("--no-frame-level")
    return command


def _edge_outputs_are_current(
    summary_json: Path,
    *,
    idtracker_csv: Path,
    matched_mousegpt_csv: Path,
    patch_length_s: float,
    patch_count: int | None,
) -> bool:
    if not summary_json.is_file():
        return False
    try:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        parameters = summary["parameters"]
        recorded_count = parameters.get("patch_count")
        source_matches = (
            Path(summary["idtracker_csv"]).resolve() == idtracker_csv.resolve()
            and Path(summary["matched_mousegpt_csv"]).resolve()
            == matched_mousegpt_csv.resolve()
        )
        duration_matches = math.isclose(
            float(parameters["patch_length_s"]),
            float(patch_length_s),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        count_matches = patch_count is None or int(recorded_count) == int(patch_count)
        return source_matches and duration_matches and count_matches
    except (OSError, ValueError, TypeError, KeyError):
        return False


def run_edge_extraction_stage(
    config: dict[str, Any], job: VideoJob, *, dry_run: bool, resume: bool
) -> dict[str, Any]:
    edge_cfg = config["edge_extraction"]
    output_dir = job.output_dir / "edges"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = job.output_dir / "logs"
    idtracker_csv = active_idtracker_features_csv(config, job)
    matched_mousegpt_csv = (
        job.output_dir
        / "identity_matching"
        / f"{job.video_id}_mousegpt_with_idtracker_identity.csv"
    )
    output_csv = output_dir / f"{job.video_id}_social_edges.csv"
    output_npz = output_dir / f"{job.video_id}_social_edges.npz"
    summary_json = output_dir / f"{job.video_id}_social_edges_summary.json"
    patch_count = (
        job.social_patch_count
        if job.social_patch_count is not None
        else edge_cfg.get("patch_count")
    )
    outputs_exist = all(path.is_file() for path in (output_csv, output_npz, summary_json))
    current = outputs_exist and _edge_outputs_are_current(
        summary_json,
        idtracker_csv=idtracker_csv,
        matched_mousegpt_csv=matched_mousegpt_csv,
        patch_length_s=float(edge_cfg["patch_length_s"]),
        patch_count=int(patch_count) if patch_count is not None else None,
    )
    if resume and current:
        extraction_result: dict[str, Any] = {"status": "skipped_existing"}
    else:
        extraction_result = run_process(
            label=f"{job.video_id}:edge-extraction",
            command=build_edge_extraction_command(
                config,
                job,
                idtracker_csv=idtracker_csv,
                matched_mousegpt_csv=matched_mousegpt_csv,
                output_dir=output_dir,
            ),
            cwd=Path(edge_cfg["script"]).resolve().parent,
            log_path=log_dir / "edge_extraction.log",
            env=None,
            dry_run=dry_run,
        )
        if not dry_run and not all(
            path.is_file() for path in (output_csv, output_npz, summary_json)
        ):
            raise FileNotFoundError(
                "Edge extractor did not create its expected CSV/NPZ/JSON outputs"
            )
    return {
        "status": "dry_run" if dry_run else "completed",
        "patch_length_s": float(edge_cfg["patch_length_s"]),
        "patch_count": int(patch_count) if patch_count is not None else None,
        "edge_csv": str(output_csv),
        "edge_npz": str(output_npz),
        "summary_json": str(summary_json),
        "extraction": extraction_result,
    }


def _safe_branch(
    name: str, function: Callable[[], dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    try:
        return name, function()
    except Exception as exc:  # keep the other long-running branch alive for diagnosis
        return name, {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_job(
    config: dict[str, Any],
    job: VideoJob,
    *,
    job_index: int,
    reusable_profile: Path,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "contract_version": "video_feature_pipeline_v0",
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "started_at": _now_iso(),
        "branches": {},
    }
    pipeline_cfg = config.get("pipeline", {})
    tracker = config.get("idtrackerai", {})
    bootstrap_value = tracker.get("bootstrap_parameters_file")
    bootstrap_profile = Path(bootstrap_value).resolve() if bootstrap_value else None
    manual_gui = idtracker_manual_for_job(
        policy=tracker.get("gui_policy", "first_video"),
        job_index=job_index,
        reusable_profile_exists=reusable_profile.is_file(),
        force_recalibrate=bool(tracker.get("force_recalibrate", False)),
    )

    branches: dict[str, Callable[[], dict[str, Any]]] = {}
    if config.get("mousegpt", {}).get("enabled", True):
        branches["mousegpt"] = lambda: run_mousegpt_branch(
            config, job, dry_run=dry_run, resume=resume
        )
    if tracker.get("enabled", True):
        branches["idtrackerai"] = lambda: run_idtracker_branch(
            config,
            job,
            manual_gui=manual_gui,
            reusable_profile=reusable_profile,
            bootstrap_profile=bootstrap_profile,
            dry_run=dry_run,
            resume=resume,
        )

    if pipeline_cfg.get("parallel_branches", True) and len(branches) > 1:
        with ThreadPoolExecutor(max_workers=len(branches)) as pool:
            futures = {
                pool.submit(_safe_branch, name, function): name
                for name, function in branches.items()
            }
            for future in as_completed(futures):
                name, result = future.result()
                status["branches"][name] = result
    else:
        for name, function in branches.items():
            branch_name, result = _safe_branch(name, function)
            status["branches"][branch_name] = result

    failed_branches = [
        name
        for name, result in status["branches"].items()
        if result.get("status") == "failed"
    ]
    matching_cfg = config.get("identity_matching", {})
    if matching_cfg.get("enabled", False) and not failed_branches:
        _stage_name, matching_result = _safe_branch(
            "identity_matching",
            lambda: run_identity_matching_stage(
                config, job, dry_run=dry_run, resume=resume
            ),
        )
        status["identity_matching"] = matching_result

    failed_stages = []
    if status.get("identity_matching", {}).get("status") == "failed":
        failed_stages.append("identity_matching")
    edge_cfg = config.get("edge_extraction", {})
    if edge_cfg.get("enabled", False) and not failed_branches and not failed_stages:
        _stage_name, edge_result = _safe_branch(
            "edge_extraction",
            lambda: run_edge_extraction_stage(
                config, job, dry_run=dry_run, resume=resume
            ),
        )
        status["edge_extraction"] = edge_result
    if status.get("edge_extraction", {}).get("status") == "failed":
        failed_stages.append("edge_extraction")
    failed = failed_branches + failed_stages
    status["status"] = "failed" if failed else ("dry_run" if dry_run else "completed")
    status["failed_branches"] = failed_branches
    status["failed_stages"] = failed_stages
    status["finished_at"] = _now_iso()
    status_path = job.output_dir / "pipeline_status.json"
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def run_pipeline(
    config: dict[str, Any],
    *,
    replacement_videos: Iterable[str] | None = None,
    dry_run_override: bool | None = None,
    gui_policy_override: str | None = None,
) -> dict[str, Any]:
    if gui_policy_override:
        config.setdefault("idtrackerai", {})["gui_policy"] = gui_policy_override
    validate_config(config)
    jobs = build_jobs(config, replacement_videos)
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = config.get("pipeline", {})
    dry_run = (
        bool(pipeline_cfg.get("dry_run", False))
        if dry_run_override is None
        else dry_run_override
    )
    resume = bool(pipeline_cfg.get("resume", True))
    tracker = config.get("idtrackerai", {})
    profile_value = tracker.get("reusable_parameters_file")
    reusable_profile = (
        Path(profile_value).resolve()
        if profile_value
        else output_root / "_idtrackerai" / "reusable_parameters.toml"
    )

    manifest: dict[str, Any] = {
        "contract_version": "video_feature_pipeline_v0",
        "started_at": _now_iso(),
        "output_root": str(output_root),
        "gui_policy": tracker.get("gui_policy", "first_video"),
        "parallel_branches": bool(pipeline_cfg.get("parallel_branches", True)),
        "jobs": [],
    }
    for index, job in enumerate(jobs):
        print(f"\n=== Video {index + 1}/{len(jobs)}: {job.video_id} ===")
        result = run_job(
            config,
            job,
            job_index=index,
            reusable_profile=reusable_profile,
            dry_run=dry_run,
            resume=resume,
        )
        manifest["jobs"].append(result)
        if result["status"] == "failed" and pipeline_cfg.get("fail_fast", True):
            break

    manifest["finished_at"] = _now_iso()
    manifest["status"] = (
        "failed"
        if any(job["status"] == "failed" for job in manifest["jobs"])
        else ("dry_run" if dry_run else "completed")
    )
    manifest_path = output_root / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPipeline manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel idtracker.ai and MouseGPT video feature extraction."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--video",
        action="append",
        help="Replace config videos; repeat this option for multiple videos.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gui-policy", choices=("first_video", "every_video", "never")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config.resolve())
    manifest = run_pipeline(
        config,
        replacement_videos=args.video,
        dry_run_override=True if args.dry_run else None,
        gui_policy_override=args.gui_policy,
    )
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
