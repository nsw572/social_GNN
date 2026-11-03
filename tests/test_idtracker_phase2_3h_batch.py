from pathlib import Path

from roughanalysis.idtracker_phase2_3h_batch import (
    MAX_CONCURRENT_TRACKING_TASKS,
    GroupPlan,
    VideoInfo,
    build_idtracker_command,
    choose_closest_subset,
    discover_phase2_groups,
)


def test_choose_closest_subset_uses_nearest_sum():
    videos = (
        VideoInfo("a.mp4", 4000.0),
        VideoInfo("b.mp4", 5000.0),
        VideoInfo("c.mp4", 6100.0),
    )

    selected = choose_closest_subset(videos, 10800.0)

    assert [item.path for item in selected] == ["b.mp4", "c.mp4"]


def test_discover_groups_by_day_and_channel(tmp_path: Path):
    expected = tmp_path / "260819_pair" / "D01" / "Phase2-D01-20260819_120000.mp4"
    expected.parent.mkdir(parents=True)
    expected.touch()
    ignored = expected.parent / "Phase3-D01-20260819_230000.mp4"
    ignored.touch()

    groups = discover_phase2_groups(tmp_path)

    assert groups == {("260819_pair", "D01"): [expected.resolve()]}


def test_tracking_command_contains_all_videos_and_track_flag(tmp_path: Path):
    parameter_file = tmp_path / "parameters.toml"
    parameter_file.write_text("number_of_animals = 2\n", encoding="utf-8")
    output_dir = tmp_path / "tracking"
    output_dir.mkdir()
    plan = GroupPlan(
        group_id="260819_pair_D01_Phase2_3h",
        experiment_day="260819_pair",
        channel="D01",
        target_seconds=10800.0,
        total_seconds=10790.0,
        difference_seconds=-10.0,
        videos=(VideoInfo("a.mp4", 5000), VideoInfo("b.mp4", 5790)),
        parameter_file=str(parameter_file),
        output_dir=str(output_dir),
    )

    command = build_idtracker_command(
        plan, Path("python.exe"), track=True, internal_workers=3
    )

    assert "--track" in command
    assert command[command.index("--video_paths") + 1 : command.index("--name")] == [
        "a.mp4",
        "b.mp4",
    ]
    assert command[command.index("--number_of_parallel_workers") + 1] == "3"


def test_tracking_concurrency_is_hard_capped_at_two():
    assert MAX_CONCURRENT_TRACKING_TASKS == 2
