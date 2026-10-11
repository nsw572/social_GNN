"""Open per-trial idtracker.ai GUIs that save parameters without tracking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .video_feature_pipeline import (
        _idtracker_environment,
        build_idtracker_command,
        build_jobs,
        load_yaml_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from video_feature_pipeline import (
        _idtracker_environment,
        build_idtracker_command,
        build_jobs,
        load_yaml_config,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def launch_calibration_guis(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path.resolve())
    jobs = build_jobs(config)
    tracker = config["idtrackerai"]
    base_profile = Path(tracker["reusable_parameters_file"]).resolve()
    if not base_profile.is_file():
        raise FileNotFoundError(
            f"Initial idtracker parameter profile not found: {base_profile}"
        )

    output_root = Path(config["output_root"]).resolve()
    calibration_root = output_root / "_idtracker_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    launched: list[tuple[Any, subprocess.Popen, Any]] = []
    manifest: dict[str, Any] = {
        "contract_version": "idtracker_calibration_v0",
        "started_at": _now_iso(),
        "config": str(config_path.resolve()),
        "jobs": [],
    }

    for job in jobs:
        destination = job.idtracker_parameters_file
        if destination is None:
            raise ValueError(
                f"Video {job.video_id!r} has no idtracker_parameters_file"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        initial_profile = destination if destination.is_file() else base_profile
        job_root = calibration_root / job.video_id
        job_root.mkdir(parents=True, exist_ok=True)
        log_path = job_root / "calibration_gui.log"
        log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        command = build_idtracker_command(
            config,
            job,
            job.output_dir / "idtrackerai",
            manual_gui=True,
            parameter_profile=initial_profile,
        )
        env = _idtracker_environment(config)
        env["IDTRACKERAI_CALIBRATION_OUTPUT"] = str(destination)
        env["IDTRACKERAI_CALIBRATION_LABEL"] = job.video_id
        creationflags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        process = subprocess.Popen(
            command,
            cwd=str(job_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        launched.append((job, process, log_handle))
        manifest["jobs"].append(
            {
                "video_id": job.video_id,
                "pid": process.pid,
                "parameter_file": str(destination),
                "initial_profile": str(initial_profile),
                "log": str(log_path),
                "status": "open",
            }
        )

    manifest_path = calibration_root / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Opened {len(launched)} idtracker calibration GUIs")
    print(f"Manifest: {manifest_path}")

    failed = False
    for index, (job, process, log_handle) in enumerate(launched):
        return_code = process.wait()
        log_handle.close()
        destination = job.idtracker_parameters_file
        saved = bool(destination and destination.is_file())
        status = "saved" if return_code == 0 and saved else "failed"
        manifest["jobs"][index].update(
            {"return_code": return_code, "saved": saved, "status": status}
        )
        failed |= status == "failed"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest["finished_at"] = _now_iso()
    manifest["status"] = "failed" if failed else "complete"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open all configured idtracker.ai calibration GUIs. Closing a GUI "
            "saves that trial's parameters and does not start tracking."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = launch_calibration_guis(args.config)
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
