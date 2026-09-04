"""Export idtracker.ai trajectories into a stable, frame-aligned contract.

This script is intentionally executed with the idtracker.ai Python interpreter.
It does not perform MouseGPT/idtracker.ai identity matching; identity values are
the original one-based idtracker.ai identities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_velocity_px_s(
    trajectories: np.ndarray, fps: float, method: str = "backward"
) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity vectors and their validity mask.

    ``backward`` matches the convention used by the existing speed notebook:
    velocity at frame ``t`` is ``(p[t] - p[t-1]) * fps``. ``central`` uses a
    centred difference for interior frames and one-sided differences at the
    two boundaries. No missing trajectory value is interpolated.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [frames, animals, 2]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if method not in {"backward", "central"}:
        raise ValueError("velocity method must be 'backward' or 'central'")

    velocity = np.full_like(trajectories, np.nan, dtype=float)
    valid = np.zeros(trajectories.shape[:2], dtype=bool)
    n_frames = trajectories.shape[0]
    if n_frames < 2:
        return velocity, valid

    if method == "backward":
        pairs_valid = np.isfinite(trajectories[1:]).all(axis=-1) & np.isfinite(
            trajectories[:-1]
        ).all(axis=-1)
        delta = (trajectories[1:] - trajectories[:-1]) * fps
        velocity[1:][pairs_valid] = delta[pairs_valid]
        valid[1:] = pairs_valid
        return velocity, valid

    first_valid = np.isfinite(trajectories[0]).all(axis=-1) & np.isfinite(
        trajectories[1]
    ).all(axis=-1)
    last_valid = np.isfinite(trajectories[-2]).all(axis=-1) & np.isfinite(
        trajectories[-1]
    ).all(axis=-1)
    velocity[0][first_valid] = (
        trajectories[1][first_valid] - trajectories[0][first_valid]
    ) * fps
    velocity[-1][last_valid] = (
        trajectories[-1][last_valid] - trajectories[-2][last_valid]
    ) * fps
    valid[0] = first_valid
    valid[-1] = last_valid

    if n_frames > 2:
        central_valid = np.isfinite(trajectories[2:]).all(axis=-1) & np.isfinite(
            trajectories[:-2]
        ).all(axis=-1)
        delta = (trajectories[2:] - trajectories[:-2]) * (fps / 2.0)
        velocity[1:-1][central_valid] = delta[central_valid]
        valid[1:-1] = central_valid
    return velocity, valid


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def _probe_video(path: Path) -> dict[str, float | int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    result = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return result


def export_idtracker_kinematics(
    *,
    session_dir: Path,
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    velocity_method: str = "backward",
) -> dict[str, Any]:
    """Export long-form CSV, compact NPZ, and a JSON metadata summary."""
    from idtrackerai.utils import load_trajectories

    session_dir = session_dir.resolve()
    video_path = video_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_json = session_dir / "session.json"
    if not session_json.is_file():
        raise FileNotFoundError(session_json)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    session_meta = json.loads(session_json.read_text(encoding="utf-8"))
    trajectory_payload = load_trajectories(session_dir)
    trajectories = np.asarray(trajectory_payload["trajectories"], dtype=float)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(
            f"Unexpected idtracker.ai trajectory shape: {trajectories.shape}"
        )

    fps = float(
        trajectory_payload.get("frames_per_second")
        or session_meta.get("frames_per_second")
        or 0.0
    )
    velocity, velocity_valid = compute_velocity_px_s(
        trajectories, fps, velocity_method
    )
    speed = np.linalg.norm(velocity, axis=-1)
    centroid_valid = np.isfinite(trajectories).all(axis=-1)
    centroid_confidence_v0 = centroid_valid.astype(np.float32)
    velocity_confidence_v0 = velocity_valid.astype(np.float32)

    n_frames, n_animals, _ = trajectories.shape
    frame_ids = np.arange(n_frames, dtype=np.int64)
    identities = np.arange(1, n_animals + 1, dtype=np.int64)
    video_meta = _probe_video(video_path)
    warnings: list[str] = []
    if video_meta["frame_count"] != n_frames:
        warnings.append(
            "Source video frame count does not match idtracker.ai trajectory length: "
            f"{video_meta['frame_count']} != {n_frames}"
        )
    if video_meta["fps"] > 0 and abs(float(video_meta["fps"]) - fps) > 1e-3:
        warnings.append(
            "Source video FPS does not match idtracker.ai FPS: "
            f"{video_meta['fps']} != {fps}"
        )

    csv_path = output_dir / f"{output_stem}_idtracker_kinematics.csv"
    npz_path = output_dir / f"{output_stem}_idtracker_kinematics.npz"
    summary_path = output_dir / f"{output_stem}_idtracker_kinematics.json"

    fieldnames = [
        "frame_id",
        "time_s",
        "identity",
        "centroid_x_px",
        "centroid_y_px",
        "centroid_valid",
        "centroid_confidence_v0",
        "velocity_x_px_s",
        "velocity_y_px_s",
        "speed_px_s",
        "velocity_valid",
        "velocity_confidence_v0",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id in range(n_frames):
            for animal_index, identity in enumerate(identities):
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "time_s": frame_id / fps,
                        "identity": int(identity),
                        "centroid_x_px": _finite_or_blank(
                            trajectories[frame_id, animal_index, 0]
                        ),
                        "centroid_y_px": _finite_or_blank(
                            trajectories[frame_id, animal_index, 1]
                        ),
                        "centroid_valid": int(
                            centroid_valid[frame_id, animal_index]
                        ),
                        "centroid_confidence_v0": float(
                            centroid_confidence_v0[frame_id, animal_index]
                        ),
                        "velocity_x_px_s": _finite_or_blank(
                            velocity[frame_id, animal_index, 0]
                        ),
                        "velocity_y_px_s": _finite_or_blank(
                            velocity[frame_id, animal_index, 1]
                        ),
                        "speed_px_s": _finite_or_blank(
                            speed[frame_id, animal_index]
                        ),
                        "velocity_valid": int(
                            velocity_valid[frame_id, animal_index]
                        ),
                        "velocity_confidence_v0": float(
                            velocity_confidence_v0[frame_id, animal_index]
                        ),
                    }
                )

    np.savez_compressed(
        npz_path,
        frame_id=frame_ids,
        identity=identities,
        fps=np.asarray(fps, dtype=np.float64),
        centroids_px=trajectories,
        centroid_valid=centroid_valid,
        centroid_confidence_v0=centroid_confidence_v0,
        velocity_px_s=velocity,
        speed_px_s=speed,
        velocity_valid=velocity_valid,
        velocity_confidence_v0=velocity_confidence_v0,
    )

    summary: dict[str, Any] = {
        "contract_version": "idtracker_kinematics_v0",
        "source_video": str(video_path),
        "session_dir": str(session_dir),
        "fps": fps,
        "frame_count": n_frames,
        "number_of_animals": n_animals,
        "velocity_method": velocity_method,
        "video_metadata": video_meta,
        "centroid_missing_per_identity": (
            (~centroid_valid).sum(axis=0).astype(int).tolist()
        ),
        "velocity_missing_per_identity": (
            (~velocity_valid).sum(axis=0).astype(int).tolist()
        ),
        "confidence_note": (
            "idtracker.ai does not expose a calibrated per-frame centroid probability "
            "in the trajectory file. V0 confidence is therefore the binary observed mask."
        ),
        "warnings": warnings,
        "outputs": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frame-aligned idtracker.ai centroids and velocity vectors."
    )
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output-stem")
    parser.add_argument(
        "--velocity-method", choices=("backward", "central"), default="backward"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_idtracker_kinematics(
        session_dir=args.session_dir,
        video_path=args.video,
        output_dir=args.outdir,
        output_stem=args.output_stem or args.video.stem,
        velocity_method=args.velocity_method,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
