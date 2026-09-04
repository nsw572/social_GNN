"""Optional visual QA overlay for MouseGPT/idtracker.ai identity matching."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


IDENTITY_COLORS = {
    1: (255, 80, 40),   # blue in BGR
    2: (45, 45, 255),   # red in BGR
    3: (60, 200, 60),
    4: (220, 100, 220),
}
UNMATCHED_COLOR = (0, 190, 255)
KEYPOINT_NAMES = ("EarL", "EarR", "Snout", "SpineF", "SpineG", "SpineH", "Hip")


def _set_below_normal_priority() -> bool:
    """Lower this process priority on Windows without adding dependencies."""
    if os.name != "nt":
        return False
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        # Keep a dependency-free fallback for lean deployment environments.
        pass
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.windll.kernel32
    return bool(
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), below_normal_priority_class
        )
    )


def _color(identity: Any) -> tuple[int, int, int]:
    try:
        if pd.isna(identity):
            return UNMATCHED_COLOR
        value = int(identity)
    except (TypeError, ValueError):
        return UNMATCHED_COLOR
    return IDENTITY_COLORS.get(value, (180, 180, 180))


def _point(x: Any, y: Any, sx: float, sy: float) -> tuple[int, int] | None:
    try:
        x_float, y_float = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_float) or not np.isfinite(y_float):
        return None
    return int(round(x_float * sx)), int(round(y_float * sy))


def _scaled_output_size(
    width: int, height: int, max_width: int, max_height: int
) -> tuple[int, int, float, float]:
    scale = min(1.0, max_width / width, max_height / height)
    out_width = max(2, int(round(width * scale / 2.0)) * 2)
    out_height = max(2, int(round(height * scale / 2.0)) * 2)
    return out_width, out_height, out_width / width, out_height / height


def _draw_identity_centroids(
    frame: np.ndarray,
    rows: pd.DataFrame,
    sx: float,
    sy: float,
) -> dict[int, tuple[int, int]]:
    points: dict[int, tuple[int, int]] = {}
    for row in rows.itertuples(index=False):
        if not bool(row.centroid_valid):
            continue
        center = _point(row.centroid_x_px, row.centroid_y_px, sx, sy)
        if center is None:
            continue
        identity = int(row.identity)
        points[identity] = center
        color = _color(identity)
        cv2.circle(frame, center, 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, center, 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"IDT {identity}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return points


def _draw_mousegpt_row(
    frame: np.ndarray,
    row: Any,
    centroid_points: dict[int, tuple[int, int]],
    sx: float,
    sy: float,
    *,
    draw_keypoints: bool,
    draw_bboxes: bool,
    draw_head_direction: bool,
) -> None:
    valid = bool(row.association_valid)
    proposed = row.proposed_idtracker_identity
    final_identity = row.idtracker_identity if valid else proposed
    color = _color(final_identity) if valid else UNMATCHED_COLOR
    anchor = _point(row.anchor_x_px, row.anchor_y_px, sx, sy)

    if draw_bboxes:
        top_left = _point(row.bbox_x1, row.bbox_y1, sx, sy)
        bottom_right = _point(row.bbox_x2, row.bbox_y2, sx, sy)
        if top_left is not None and bottom_right is not None:
            cv2.rectangle(frame, top_left, bottom_right, color, 2, cv2.LINE_AA)

    if draw_keypoints:
        for name in KEYPOINT_NAMES:
            point = _point(
                getattr(row, f"{name}_x"), getattr(row, f"{name}_y"), sx, sy
            )
            if point is not None:
                cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)

    if draw_head_direction:
        origin = _point(row.origin_x, row.origin_y, sx, sy)
        try:
            dx, dy = float(row.direction_x), float(row.direction_y)
        except (TypeError, ValueError):
            dx = dy = float("nan")
        if origin is not None and np.isfinite(dx) and np.isfinite(dy):
            length = 90
            end = (
                int(round(origin[0] + dx * length)),
                int(round(origin[1] + dy * length)),
            )
            cv2.arrowedLine(frame, origin, end, color, 2, cv2.LINE_AA, tipLength=0.25)

    if anchor is None:
        return
    cv2.drawMarker(
        frame,
        anchor,
        color,
        markerType=cv2.MARKER_DIAMOND,
        markerSize=16,
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    if not pd.isna(proposed) and int(proposed) in centroid_points:
        target = centroid_points[int(proposed)]
        line_color = _color(proposed) if valid else (120, 120, 120)
        cv2.line(frame, anchor, target, line_color, 2, cv2.LINE_AA)

    track = int(row.mousegpt_track_id) if not pd.isna(row.mousegpt_track_id) else "?"
    identity_text = int(proposed) if not pd.isna(proposed) else "?"
    distance = (
        f"{float(row.association_distance_px):.0f}px"
        if np.isfinite(float(row.association_distance_px))
        else "NA"
    )
    confidence = float(row.association_confidence)
    label = (
        f"MG {track} -> ID {identity_text}  d={distance} q={confidence:.2f}"
        if valid
        else f"MG {track} REJECT {row.association_status}"
    )
    cv2.putText(
        frame,
        label,
        (anchor[0] + 10, max(24, anchor[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_panel(
    frame: np.ndarray,
    frame_id: int,
    rows: pd.DataFrame,
    tracker_rows: pd.DataFrame,
) -> None:
    valid_count = int(rows["association_valid"].sum()) if len(rows) else 0
    rejected = len(rows) - valid_count
    if len(tracker_rows) and "centroid_valid" in tracker_rows:
        tracker_count = int(
            pd.to_numeric(tracker_rows["centroid_valid"], errors="coerce")
            .fillna(0)
            .astype(bool)
            .sum()
        )
    else:
        tracker_count = 0
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (600, 142), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    lines = [
        f"frame {frame_id}",
        f"MouseGPT detections: {len(rows)}   idtracker valid IDs: {tracker_count}",
        f"accepted matches: {valid_count}   rejected MouseGPT: {rejected}",
        "circle = idtracker centroid   diamond = MouseGPT anchor",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (26, 36 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def render_identity_match_overlay(
    *,
    video_path: Path,
    enriched_mousegpt_csv: Path,
    idtracker_csv: Path,
    output_path: Path,
    frame_start: int = 0,
    frame_end: int | None = None,
    max_width: int = 1920,
    max_height: int = 1080,
    codec: str = "mp4v",
    draw_keypoints: bool = True,
    draw_bboxes: bool = True,
    draw_head_direction: bool = True,
    opencv_threads: int | None = None,
    low_priority: bool = False,
) -> dict[str, Any]:
    if opencv_threads is not None:
        if opencv_threads < 1:
            raise ValueError("opencv_threads must be at least 1")
        cv2.setNumThreads(opencv_threads)
    below_normal_applied = _set_below_normal_priority() if low_priority else False

    mouse = pd.read_csv(enriched_mousegpt_csv)
    tracker = pd.read_csv(idtracker_csv)
    mouse_groups = mouse.groupby("frame_id", sort=False).indices
    tracker_groups = tracker.groupby("frame_id", sort=False).indices

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read source video metadata")

    frame_start = max(0, int(frame_start))
    frame_end = total_frames if frame_end is None else min(int(frame_end), total_frames)
    if frame_start >= frame_end:
        cap.release()
        raise ValueError(f"Invalid frame range [{frame_start}, {frame_end})")
    out_width, out_height, sx, sy = _scaled_output_size(
        width, height, max_width, max_height
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    written = 0
    try:
        for frame_id in range(frame_start, frame_end):
            ok, source = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
            frame = cv2.resize(source, (out_width, out_height), interpolation=cv2.INTER_AREA)
            tracker_indices = tracker_groups.get(frame_id, [])
            tracker_rows = tracker.iloc[tracker_indices] if len(tracker_indices) else tracker.iloc[0:0]
            centroid_points = _draw_identity_centroids(frame, tracker_rows, sx, sy)

            mouse_indices = mouse_groups.get(frame_id, [])
            mouse_rows = mouse.iloc[mouse_indices] if len(mouse_indices) else mouse.iloc[0:0]
            for row in mouse_rows.itertuples(index=False):
                _draw_mousegpt_row(
                    frame,
                    row,
                    centroid_points,
                    sx,
                    sy,
                    draw_keypoints=draw_keypoints,
                    draw_bboxes=draw_bboxes,
                    draw_head_direction=draw_head_direction,
                )
            _draw_panel(frame, frame_id, mouse_rows, tracker_rows)
            writer.write(frame)
            written += 1
            if written % 1000 == 0:
                print(f"Rendered {written}/{frame_end - frame_start} frames")
    finally:
        cap.release()
        writer.release()

    summary = {
        "contract_version": "identity_match_visualization_v0",
        "video": str(video_path.resolve()),
        "enriched_mousegpt_csv": str(enriched_mousegpt_csv.resolve()),
        "idtracker_csv": str(idtracker_csv.resolve()),
        "output_video": str(output_path.resolve()),
        "fps": fps,
        "source_size": [width, height],
        "output_size": [out_width, out_height],
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "frames_written": written,
        "resource_limits": {
            "opencv_threads": int(cv2.getNumThreads()),
            "below_normal_priority_requested": bool(low_priority),
            "below_normal_priority_applied": bool(below_normal_applied),
            "gpu_used": False,
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render identity matching QA video.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--enriched-mousegpt-csv", required=True, type=Path)
    parser.add_argument("--idtracker-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--codec", default="mp4v", choices=("mp4v", "avc1"))
    parser.add_argument("--no-keypoints", action="store_true")
    parser.add_argument("--no-bboxes", action="store_true")
    parser.add_argument("--no-head-direction", action="store_true")
    parser.add_argument(
        "--opencv-threads",
        type=int,
        help="Limit OpenCV CPU worker threads (for example, 1 for low impact).",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="Run at Windows BELOW_NORMAL process priority.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = render_identity_match_overlay(
        video_path=args.video,
        enriched_mousegpt_csv=args.enriched_mousegpt_csv,
        idtracker_csv=args.idtracker_csv,
        output_path=args.output,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_width=args.max_width,
        max_height=args.max_height,
        codec=args.codec,
        draw_keypoints=not args.no_keypoints,
        draw_bboxes=not args.no_bboxes,
        draw_head_direction=not args.no_head_direction,
        opencv_threads=args.opencv_threads,
        low_priority=args.low_priority,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
