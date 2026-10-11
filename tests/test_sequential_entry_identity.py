import shutil
import unittest
import uuid
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from social_gnn.sequential_entry_identity import (
    IMUAnimalSource,
    IMUEntryEvent,
    SequentialEntryIdentityError,
    apply_sequential_entry_mapping,
    detect_entry_event_from_imu,
    infer_sequential_entry_mapping,
    run_sequential_entry_identity_mapping,
)
from social_gnn.video_feature_pipeline import _matching_outputs_are_current


class SequentialEntryIdentityTest(unittest.TestCase):
    @staticmethod
    def _idtracker_table(*, ambiguous=False):
        rows = []
        for frame_id in range(200):
            time_s = frame_id / 10.0
            for identity in (1, 2):
                if ambiguous:
                    valid = time_s >= 2.0
                else:
                    valid = (identity == 2 and time_s >= 2.0) or (
                        identity == 1 and time_s >= 12.0
                    )
                rows.append(
                    {
                        "frame_id": frame_id,
                        "time_s": time_s,
                        "identity": identity,
                        "centroid_x_px": frame_id + identity if valid else np.nan,
                        "centroid_y_px": 20.0 * identity if valid else np.nan,
                        "centroid_valid": int(valid),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _events():
        return (
            IMUEntryEvent("A4_1c", 2.0, "manual_confirmed", "a.csv", None),
            IMUEntryEvent("B4_ab", 12.0, "manual_confirmed", "b.csv", None),
        )

    def test_single_animal_window_recovers_swapped_idtracker_number(self):
        source = self._idtracker_table()
        mapping = infer_sequential_entry_mapping(
            source,
            self._events(),
            trial_id="A4_B4",
            after_first_entry_margin_s=1.0,
            before_second_entry_margin_s=1.0,
        )
        self.assertEqual(mapping.source_to_canonical, {2: 1, 1: 2})
        self.assertEqual(mapping.source_to_label, {2: "A4_1c", 1: "B4_ab"})
        self.assertEqual(mapping.confidence_v0, 1.0)

        canonical = apply_sequential_entry_mapping(source, mapping)
        id2 = canonical[canonical["source_idtracker_identity"] == 2]
        self.assertTrue((id2["identity"] == 1).all())
        self.assertTrue((id2["canonical_identity_label"] == "A4_1c").all())
        self.assertTrue((source["identity"].to_numpy() == [1, 2] * 200).all())

    def test_ambiguous_two_identity_window_is_rejected(self):
        with self.assertRaisesRegex(
            SequentialEntryIdentityError, "single-valid-identity fraction"
        ):
            infer_sequential_entry_mapping(
                self._idtracker_table(ambiguous=True),
                self._events(),
                trial_id="ambiguous",
                after_first_entry_margin_s=1.0,
                before_second_entry_margin_s=1.0,
            )

    def test_two_and_a_half_second_entry_gap_is_sufficient(self):
        events = (
            self._events()[0],
            IMUEntryEvent("B4_ab", 4.5, "manual_confirmed", "b.csv", None),
        )
        mapping = infer_sequential_entry_mapping(
            self._idtracker_table(), events, trial_id="rapid_entry"
        )
        self.assertAlmostEqual(mapping.anchor_start_s, 2.5)
        self.assertAlmostEqual(mapping.anchor_end_s, 3.5)
        self.assertEqual(mapping.anchor_frame_count, 10)
        self.assertEqual(mapping.selected_source_identity, 2)

    def test_incomplete_or_gapped_tracking_cannot_claim_full_coverage(self):
        source = self._idtracker_table()
        for broken in (
            source[source.time_s >= 4.0],
            source[source.frame_id != 50],
            source.drop(index=100),
            source.assign(frame_id=source.frame_id + 0.1),
        ):
            with self.subTest(rows=len(broken)):
                with self.assertRaisesRegex(SequentialEntryIdentityError, "trial 'incomplete'"):
                    infer_sequential_entry_mapping(
                        broken, self._events(), trial_id="incomplete",
                        after_first_entry_margin_s=1.0,
                        before_second_entry_margin_s=1.0,
                    )

    def test_too_close_entries_are_rejected(self):
        events = (
            self._events()[0],
            IMUEntryEvent("B4_ab", 3.0, "manual_confirmed", "b.csv", None),
        )
        with self.assertRaisesRegex(SequentialEntryIdentityError, "shorter than"):
            infer_sequential_entry_mapping(self._idtracker_table(), events, trial_id="close")

    def test_dynamic_acceleration_peak_detects_entry(self):
        time_s = np.arange(0.0, 10.0, 0.01)
        acceleration = np.zeros((len(time_s), 3), dtype=float)
        acceleration[:, 2] = 1.0
        acceleration[400, 2] = 8.0
        imu = pd.DataFrame(
            {
                "time": time_s,
                "ax": acceleration[:, 0],
                "ay": acceleration[:, 1],
                "az": acceleration[:, 2],
                "interp_mask": 0,
            }
        )
        event_time, peak_z = detect_entry_event_from_imu(
            imu,
            label="mouse",
            search_end_s=8.0,
            threshold_z=6.0,
        )
        self.assertAlmostEqual(event_time, 4.0, places=6)
        self.assertGreater(peak_z, 6.0)

    def test_file_runner_writes_auditable_non_destructive_contract(self):
        root = Path.cwd() / "artifacts" / f"test_entry_identity_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            idtracker_path = root / "idtracker.csv"
            imu_a_path = root / "a.csv"
            imu_b_path = root / "b.csv"
            self._idtracker_table().to_csv(idtracker_path, index=False)
            imu = pd.DataFrame(
                {
                    "time": [0.0, 0.1, 0.2],
                    "ax": [0.0, 0.0, 0.0],
                    "ay": [0.0, 0.0, 0.0],
                    "az": [1.0, 1.0, 1.0],
                }
            )
            imu.to_csv(imu_a_path, index=False)
            imu.to_csv(imu_b_path, index=False)
            summary = run_sequential_entry_identity_mapping(
                idtracker_csv=idtracker_path,
                animals=(
                    IMUAnimalSource("A4_1c", imu_a_path, 2.0),
                    IMUAnimalSource("B4_ab", imu_b_path, 12.0),
                ),
                output_dir=root / "out",
                output_stem="A4_B4",
                after_first_entry_margin_s=1.0,
                before_second_entry_margin_s=1.0,
            )
            canonical_path = Path(summary["outputs"]["canonical_idtracker_csv"])
            mapping_path = Path(summary["outputs"]["mapping_csv"])
            self.assertTrue(canonical_path.is_file())
            self.assertTrue(mapping_path.is_file())
            self.assertTrue(Path(summary["outputs"]["summary_json"]).is_file())
            canonical = pd.read_csv(canonical_path)
            self.assertIn("source_idtracker_identity", canonical)
            self.assertIn("canonical_identity_label", canonical)
            self.assertEqual(summary["source_to_canonical"], {"2": 1, "1": 2})
            self.assertEqual(summary["canonical_label_order"], ["A4_1c", "B4_ab"])
            # A new identity permutation at the same CSV path must invalidate
            # old MouseGPT matching, which subsequently invalidates edges.
            cached_summary = root / "match.json"
            cached_summary.write_text(json.dumps({"idtracker_csv": str(canonical_path)}))
            os.utime(cached_summary, ns=(2_000_000_000, 2_000_000_000))
            os.utime(canonical_path, ns=(1_000_000_000, 1_000_000_000))
            self.assertTrue(_matching_outputs_are_current(cached_summary, canonical_path))
            os.utime(canonical_path, ns=(3_000_000_000, 3_000_000_000))
            self.assertFalse(_matching_outputs_are_current(cached_summary, canonical_path))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
