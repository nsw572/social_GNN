import unittest

import numpy as np
import pandas as pd

from social_gnn.downstream.video_montage import (
    SEPARATOR_FRAMES,
    add_frame_timing,
    canonical_run_directory_name,
    montage_segment_plan,
    resolve_run,
    select_longest_bouts,
)


class VideoMontageTests(unittest.TestCase):
    def test_run_resolution_is_exact(self):
        audit = pd.DataFrame(
            {
                "trial_id": ["A4_C4", "A4_C4"],
                "encoder_condition": ["graph_full", "graph_full"],
                "encoder_seed": [20261029, 20261029],
                "downstream_seed": [0, 1],
                "state_run_id": [
                    "graph_full|e20261029|hmm|k6|d0",
                    "graph_full|e20261029|hmm|k6|d1",
                ],
                "bout_rows_exact": [True, True],
                "partition_nonoverlap": [True, True],
            }
        )
        row = resolve_run(
            audit,
            run_id=None,
            trial_id="A4_C4",
            encoder_condition="graph_full",
            encoder_seed=20261029,
            downstream_algorithm="hmm",
            states=6,
            downstream_seed=0,
        )
        self.assertEqual(row.state_run_id, "graph_full|e20261029|hmm|k6|d0")
        self.assertEqual(
            canonical_run_directory_name(row.state_run_id),
            "graph_full_e20261029_hmm_k6_d0",
        )

    def test_longest_selection_uses_earliest_start_for_ties(self):
        rows = []
        for bout, state, start, duration in (
            (0, 0, 10.0, 2.0),
            (1, 0, 5.0, 2.0),
            (2, 0, 1.0, 1.0),
            (3, 1, 4.0, 3.0),
        ):
            rows.append(
                {
                    "state_run_id": "r",
                    "trial_id": "t",
                    "state": state,
                    "bout_id": bout,
                    "bout_start_s": start,
                    "bout_end_s": start + duration,
                    "bout_duration_s": duration,
                    "video_frame_start": bout * 10,
                    "video_frame_end": bout * 10 + 4,
                    "alignment_status": "mapped_from_packet_validated_pts_grid",
                    "source_video": "video.mp4",
                    "video_time_offset_s": 0.0,
                }
            )
        selected = select_longest_bouts(
            pd.DataFrame(rows), top_n=2, states=6
        )
        state_zero = selected[selected.state.eq(0)].sort_values(
            "rank_by_duration"
        )
        self.assertEqual(state_zero.bout_id.tolist(), [1, 0])
        self.assertEqual(state_zero.rank_by_duration.tolist(), [1, 2])

    def test_frame_timing_uses_validated_grid_and_half_open_end(self):
        selected = pd.DataFrame(
            {
                "video_frame_start": [10],
                "video_frame_end": [14],
                "bout_start_s": [1.01],
                "bout_end_s": [1.51],
                "bout_duration_s": [0.5],
                "video_time_offset_s": [0.0],
            }
        )
        video = pd.Series(
            {
                "nominal_frame_rate": 10.0,
                "stream_start_time_s": 0.0,
                "frame_pts_count": 100,
            }
        )
        timed = add_frame_timing(selected, video).iloc[0]
        self.assertEqual(timed.source_frame_count, 5)
        self.assertAlmostEqual(timed.source_first_pts_s, 1.0)
        self.assertAlmostEqual(timed.source_end_exclusive_pts_s, 1.5)
        self.assertAlmostEqual(timed.start_quantization_error_s, -0.01)
        self.assertAlmostEqual(timed.end_quantization_error_s, -0.01)

    def test_segment_plan_has_only_internal_exact_separators(self):
        plan, ranges = montage_segment_plan([3, 4, 2])
        self.assertEqual(plan[0]["kind"], "clip")
        self.assertEqual(plan[-1]["kind"], "clip")
        self.assertEqual(
            ranges,
            [
                (3, 3 + SEPARATOR_FRAMES - 1),
                (22, 22 + SEPARATOR_FRAMES - 1),
            ],
        )
        separator_lengths = [end - start + 1 for start, end in ranges]
        np.testing.assert_array_equal(separator_lengths, [15, 15])


if __name__ == "__main__":
    unittest.main()
