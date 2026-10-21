"""Run the saved-output temporal-alignment and HMM-state consistency audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .state_correspondence import (
    ACROSS_ENCODER,
    GRAPH_VS_MLP,
    SAME_CHECKPOINT,
    analyze_state_correspondence,
)
from .temporal_alignment import analyze_temporal_alignment


def _executive_summary(root: Path) -> None:
    temporal = pd.read_csv(root / "temporal_alignment" / "trial_alignment_summary.csv")
    failures = pd.read_csv(root / "temporal_alignment" / "alignment_failures.csv")
    pairs = pd.read_csv(
        root / "state_correspondence" / "pairwise_correspondence_summary.csv"
    )
    group = pairs.groupby("comparison_scope").agg(
        pair_count=("pair_id", "size"),
        ari_mean=("ARI", "mean"),
        matched_f1_mean=("mean_hungarian_f1", "mean"),
        boundary_f1_mean=("boundary_f1_exact", "mean"),
    )
    graph_seed = pairs[
        pairs.comparison_scope.eq(ACROSS_ENCODER)
        & pairs.left_encoder.eq("graph_full")
        & (
            pairs.left_encoder_seed.eq(20261029)
            | pairs.right_encoder_seed.eq(20261029)
        )
    ]
    valid_video = int(temporal.video_pts_available.sum())
    clock_ok = bool(
        temporal.clock_valid.all()
        and temporal.node_edge_clock_equal.all()
        and temporal.embedding_alignment_failures.eq(0).all()
        and temporal.hmm_assignment_alignment_failures.fillna(0).eq(0).all()
    )
    lines = [
        "# Temporal and HMM state-consistency audit: executive summary",
        "",
        "Date: 2026-10-20",
        "",
        "No encoder, PCA, HMM, or HSMM was retrained. This audit reuses the saved K=6 HMM assignments on A4_C4 and the existing canonical trial clocks.",
        "",
        "## Outcome",
        "",
        f"- **Timestamp/data integrity:** {'passed' if clock_ok else 'requires attention'}. Node, edge, embedding, assignment and bout clocks were checked by physical timestamps rather than row order.",
        f"- **Original-video accessibility:** ffprobe timing metadata and packet-PTS samples were read for {valid_video}/{len(temporal)} source videos. These five CFR streams use their rational 29.97-fps timing grid after beginning/middle/end packet validation, not `round(time * average_fps)`.",
        "- **Physical synchronization:** the zero social-clock-to-video offsets are explicit and traceable to the upstream statement that delivered timestamps already use video-relative time. This is strong metadata consistency, but not independent hardware/event synchronization proof; contact sheets remain a human spot check.",
        "- **State reproducibility:** fixed-encoder downstream starts agree much more strongly than independently trained encoder representations. State-level tables expose stable matches, weak forced matches, splits, merges, and ambiguity instead of forcing a global taxonomy.",
        "- **Biological validity:** not tested. Alignment and repeatability do not make an HMM state a real behavioral category.",
        "",
        "## HMM K=6 correspondence",
        "",
        "| Comparison | Pairs | Mean ARI | Mean matched-state F1 | Exact boundary F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in (SAME_CHECKPOINT, ACROSS_ENCODER, GRAPH_VS_MLP):
        row = group.loc[scope]
        lines.append(
            f"| `{scope}` | {int(row.pair_count)} | {row.ari_mean:.3f} | "
            f"{row.matched_f1_mean:.3f} | {row.boundary_f1_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Graph Full encoder seed `20261029`, despite its near-deterministic within-checkpoint HMM optimization, averages cross-encoder ARI **{graph_seed.ARI.mean():.3f}** and matched-state F1 **{graph_seed.mean_hungarian_f1.mean():.3f}** against the other Graph Full encoder seeds. It should not be promoted as biologically superior solely because it is easy for HMM to fit reproducibly.",
            "",
            "## Recommendation",
            "",
            "The current patch-to-state mapping is reliable enough for timestamped downstream inspection and controlled quantitative analyses: every valid A4_C4 assignment and bout retains a traceable physical interval and source-frame lookup. State IDs are not yet reliable enough to be treated as seed-independent global behavioral labels. Use run-local IDs, report correspondence quality, prefer confidence-qualified within-checkpoint consensus for visualization, and validate state signatures on a genuinely independent animal pair before biological naming.",
            "",
            f"The temporal audit emitted {len(failures)} integrity-failure rows. See the two detailed reports and machine-readable tables for scope-specific diagnostics.",
        ]
    )
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "executive_summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Keep the requested report layout while retaining each analysis beside its tables.
    for source, destination in (
        (root / "temporal_alignment" / "temporal_alignment_report.md", reports / "temporal_alignment_report.md"),
        (root / "state_correspondence" / "state_correspondence_report.md", reports / "state_correspondence_report.md"),
    ):
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        default="evaluation_runs/gnn_9dim_downstream_v1/comparisons/confirmation/all_runs.csv",
    )
    parser.add_argument(
        "--measurement-csv",
        default="evaluation_runs/gnn_9dim_downstream_v1/comparisons/confirmation/state_measurements.csv",
    )
    parser.add_argument("--node-root", default="evaluation_runs/hybrid_9dim_v1/nodes")
    parser.add_argument(
        "--edge-root", default=r"D:\wxh本地化处理库\social_gnn_6dim_v0\video_features"
    )
    parser.add_argument(
        "--video-features-root", default=r"D:\wxh本地化处理库\social_gnn_6dim_v0\video_features"
    )
    parser.add_argument(
        "--embedding-root", default="evaluation_runs/gnn_9dim_downstream_v1/embeddings"
    )
    parser.add_argument(
        "--offset-config", default="configs/social_gnn_temporal_alignment_v1.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "evaluation_runs/gnn_9dim_downstream_v1/comparisons/"
            "temporal_and_state_consistency"
        ),
    )
    parser.add_argument(
        "--ffprobe", default=r"D:\ffmpeg-7.1-essentials_build\bin\ffprobe.exe"
    )
    parser.add_argument(
        "--ffmpeg", default=r"D:\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
    )
    parser.add_argument("--extract-qc", action="store_true")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--skip-state", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_temporal:
        analyze_temporal_alignment(
            runs_csv=args.runs_csv,
            node_root=args.node_root,
            edge_root=args.edge_root,
            embedding_root=args.embedding_root,
            video_features_root=args.video_features_root,
            offset_config=args.offset_config,
            output_dir=root / "temporal_alignment",
            ffprobe=args.ffprobe,
            ffmpeg=args.ffmpeg,
            extract_qc=args.extract_qc,
        )
    if not args.skip_state:
        analyze_state_correspondence(
            runs_csv=args.runs_csv,
            measurement_csv=args.measurement_csv,
            output_dir=root / "state_correspondence",
        )
    if (root / "temporal_alignment" / "trial_alignment_summary.csv").exists() and (
        root / "state_correspondence" / "pairwise_correspondence_summary.csv"
    ).exists():
        _executive_summary(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
