import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from social_gnn.idtracker_export import compute_velocity_px_s
from social_gnn.identity_matching import match_mousegpt_to_idtracker
from social_gnn.video_feature_pipeline import (
    VideoJob,
    active_idtracker_features_csv,
    build_edge_extraction_command,
    build_idtracker_command,
    build_idtracker_smoothing_command,
    build_mousegpt_command,
    extract_reusable_idtracker_parameters,
    idtracker_manual_for_job,
)


class VideoFeaturePipelineTest(unittest.TestCase):
    def test_gui_policy(self):
        self.assertTrue(
            idtracker_manual_for_job(
                policy="first_video",
                job_index=0,
                reusable_profile_exists=False,
                force_recalibrate=False,
            )
        )
        self.assertFalse(
            idtracker_manual_for_job(
                policy="first_video",
                job_index=1,
                reusable_profile_exists=True,
                force_recalibrate=False,
            )
        )
        self.assertTrue(
            idtracker_manual_for_job(
                policy="every_video",
                job_index=3,
                reusable_profile_exists=True,
                force_recalibrate=False,
            )
        )
        self.assertFalse(
            idtracker_manual_for_job(
                policy="never",
                job_index=0,
                reusable_profile_exists=True,
                force_recalibrate=False,
            )
        )

    def test_backward_velocity_preserves_missing_values(self):
        points = np.asarray(
            [
                [[0.0, 0.0], [5.0, 5.0]],
                [[1.0, 2.0], [np.nan, np.nan]],
                [[3.0, 2.0], [8.0, 5.0]],
            ]
        )
        velocity, valid = compute_velocity_px_s(points, fps=10.0)
        np.testing.assert_allclose(velocity[1, 0], [10.0, 20.0])
        np.testing.assert_allclose(velocity[2, 0], [20.0, 0.0])
        self.assertFalse(valid[0].any())
        self.assertFalse(valid[:, 1].any())
        self.assertTrue(np.isnan(velocity[:, 1]).all())

    def test_parameter_snapshot_excludes_video_specific_fields(self):
        reusable = extract_reusable_idtracker_parameters(
            {
                "video_paths": ["old.mp4"],
                "name": "old",
                "tracking_intervals": [[0, 100]],
                "number_of_animals": 2,
                "intensity_ths": [0, 100],
                "area_ths": [120, 9000],
                "roi_list": ["+ Polygon [[0, 0], [1, 0], [1, 1]]"],
                "use_bkg": True,
            }
        )
        self.assertEqual(reusable["number_of_animals"], 2)
        self.assertTrue(reusable["use_bkg"])
        self.assertNotIn("video_paths", reusable)
        self.assertNotIn("tracking_intervals", reusable)

    def test_commands_are_argument_lists_without_shell_escaping(self):
        job = VideoJob("sample", Path("C:/video sample.mp4"), Path("C:/out/sample"))
        config = {
            "mousegpt": {
                "python": "C:/mouse/python.exe",
                "inferencer_script": "C:/mouse/inferencer_demo.py",
                "pose_config": "C:/mouse/pose.py",
                "pose_weights": "C:/mouse/pose.pth",
                "det_config": "C:/mouse/det.py",
                "det_weights": "C:/mouse/det.pth",
                "draw_bbox": True,
            },
            "idtrackerai": {
                "python": "C:/id/python.exe",
                "number_of_animals": 2,
                "trajectories_formats": ["npy"],
                "data_policy": "all",
            },
        }
        mouse = build_mousegpt_command(
            config, job, Path("C:/pred"), Path("C:/vis")
        )
        tracker = build_idtracker_command(
            config,
            job,
            Path("C:/idout"),
            manual_gui=False,
            parameter_profile=None,
        )
        self.assertIn("--draw-bbox", mouse)
        self.assertIn("C:\\video sample.mp4", mouse)
        self.assertIn("--track", tracker)
        self.assertNotIn("`", "".join(mouse + tracker))

    def test_smoothing_switch_selects_repaired_contract(self):
        job = VideoJob("sample", Path("C:/video.mp4"), Path("C:/out/sample"))
        config = {
            "idtrackerai": {
                "export": {"velocity_method": "backward"},
                "smoothing": {
                    "enabled": True,
                    "python": "C:/deepof/python.exe",
                    "script": "C:/repo/social_gnn/idtracker_identity_smoothing.py",
                    "speed_multiplier": 5.0,
                },
            }
        }
        active = active_idtracker_features_csv(config, job)
        self.assertTrue(active.name.endswith("_idtracker_kinematics_smoothed.csv"))
        command = build_idtracker_smoothing_command(
            config,
            job,
            session_dir=Path("C:/out/sample/idtrackerai/session_sample"),
            input_npz=Path("C:/out/sample/idtrackerai/features/raw.npz"),
            output_dir=Path("C:/out/sample/idtrackerai/features"),
        )
        self.assertIn("--speed-multiplier", command)
        self.assertIn("5.0", command)
        self.assertNotIn("`", "".join(command))

    def test_edge_command_uses_fixed_unknown_patch_parameter(self):
        job = VideoJob(
            "sample",
            Path("C:/video.mp4"),
            Path("C:/out/sample"),
            social_patch_count=17,
        )
        config = {
            "edge_extraction": {
                "python": "C:/deepof/python.exe",
                "script": "C:/repo/social_gnn/edge_extraction.py",
                "patch_length_s": 0.125,
                "clock_start_s": 0.0,
            }
        }
        command = build_edge_extraction_command(
            config,
            job,
            idtracker_csv=Path("C:/out/id.csv"),
            matched_mousegpt_csv=Path("C:/out/mouse.csv"),
            output_dir=Path("C:/out/edges"),
        )
        self.assertEqual(command[command.index("--patch-length-s") + 1], "0.125")
        self.assertEqual(command[command.index("--patch-count") + 1], "17")

    @staticmethod
    def _mouse_row(frame, track, x, y):
        row = {"frame_id": frame, "track_id": track, "bbox_score": 0.95}
        for name in ("SpineF", "SpineG", "SpineH", "Hip"):
            row[f"{name}_x"] = x
            row[f"{name}_y"] = y
            row[f"{name}_score"] = 0.9
        row.update(
            {
                "bbox_x1": x - 5,
                "bbox_y1": y - 5,
                "bbox_x2": x + 5,
                "bbox_y2": y + 5,
            }
        )
        return row

    @staticmethod
    def _id_row(frame, identity, x, y, confidence=1.0):
        return {
            "frame_id": frame,
            "identity": identity,
            "centroid_x_px": x,
            "centroid_y_px": y,
            "centroid_valid": 1,
            "centroid_confidence_v0": confidence,
        }

    def test_matching_is_framewise_and_ignores_mousegpt_track_identity(self):
        mouse = pd.DataFrame(
            [
                self._mouse_row(0, 1, 1, 0),
                self._mouse_row(0, 2, 99, 0),
                # MouseGPT track labels swap sides on the next frame.
                self._mouse_row(1, 1, 99, 0),
                self._mouse_row(1, 2, 1, 0),
            ]
        )
        tracker = pd.DataFrame(
            [
                self._id_row(0, 1, 0, 0),
                self._id_row(0, 2, 100, 0),
                self._id_row(1, 1, 0, 0),
                self._id_row(1, 2, 100, 0),
            ]
        )
        _matches, enriched, _frames = match_mousegpt_to_idtracker(
            mouse,
            tracker,
            max_distance_px=20,
            min_assignment_margin_px=10,
            distance_confidence_scale_px=10,
            margin_confidence_scale_px=20,
        )
        track_one = enriched[enriched["track_id"] == 1].sort_values("frame_id")
        self.assertEqual(track_one["idtracker_identity"].astype(int).tolist(), [1, 2])
        self.assertTrue(enriched["association_valid"].all())

    def test_ambiguous_assignment_is_rejected(self):
        mouse = pd.DataFrame(
            [
                self._mouse_row(0, 1, 49, 0),
                self._mouse_row(0, 2, 51, 0),
            ]
        )
        tracker = pd.DataFrame(
            [self._id_row(0, 1, 0, 0), self._id_row(0, 2, 100, 0)]
        )
        _matches, enriched, frames = match_mousegpt_to_idtracker(
            mouse,
            tracker,
            max_distance_px=100,
            min_assignment_margin_px=10,
            distance_confidence_scale_px=100,
            margin_confidence_scale_px=20,
        )
        self.assertFalse(enriched["association_valid"].any())
        self.assertTrue(
            enriched["association_status"].str.contains("ambiguous").all()
        )
        self.assertLess(float(frames.loc[0, "assignment_margin_px"]), 10)

    def test_smoothed_centroid_confidence_caps_mousegpt_association(self):
        mouse = pd.DataFrame(
            [self._mouse_row(0, 1, 1, 0), self._mouse_row(0, 2, 99, 0)]
        )
        tracker = pd.DataFrame(
            [
                self._id_row(0, 1, 0, 0, confidence=0.05),
                self._id_row(0, 2, 100, 0, confidence=1.0),
            ]
        )
        _matches, enriched, _frames = match_mousegpt_to_idtracker(
            mouse,
            tracker,
            max_distance_px=20,
            min_assignment_margin_px=10,
            distance_confidence_scale_px=10,
            margin_confidence_scale_px=20,
        )
        identity_one = enriched[enriched["idtracker_identity"] == 1].iloc[0]
        self.assertTrue(identity_one["association_valid"])
        self.assertAlmostEqual(identity_one["association_confidence"], 0.05)
        self.assertAlmostEqual(identity_one["idtracker_centroid_confidence"], 0.05)


if __name__ == "__main__":
    unittest.main()
