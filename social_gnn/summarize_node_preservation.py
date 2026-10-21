"""Build one comparison table from frozen-probe, role, and geometry artifacts.

Each ``--probe`` value is ``LABEL=MODEL=probe_metrics.csv``.  Repeating a
label supplies independent training seeds; means and sample standard deviations
are then reported without treating patches as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluation import DIRECTED_TRAITS, TRAIT_NAMES, write_csv, write_json


def _spec(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("artifact specification must be LABEL=MODEL=PATH")
    path = Path(parts[2])
    if not path.is_file():
        raise FileNotFoundError(path)
    return parts[0], parts[1], path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _mean(rows: Sequence[dict[str, str]], predicate) -> float:
    values = [float(row["r2"]) for row in rows if predicate(row) and np.isfinite(float(row["r2"]))]
    return float(np.mean(values)) if values else float("nan")


def _probe_record(model: str, path: Path) -> dict[str, float]:
    rows = [
        row for row in _read(path)
        if row.get("model") == model
        and row.get("probe") == "linear"
        and row.get("input_mode", "native") == "native"
        and row.get("subset") == "confidence_weighted_all"
        and row.get("split") in {"test", "evaluation"}
    ]
    nodes = [row for row in rows if row.get("target_kind") == "node_invariant"]
    edges = [row for row in rows if row.get("target_kind") == "edge"]
    invariants = [row for row in rows if row.get("target_kind") == "edge_invariant"]
    directed_names = {TRAIT_NAMES[index] for index in DIRECTED_TRAITS}
    return {
        "node_mean_r2": _mean(nodes, lambda row: row["target"].startswith("node_mean_")),
        "node_difference_r2": _mean(nodes, lambda row: row["target"].startswith("node_abs_difference_")),
        "node_combined_r2": _mean(nodes, lambda _row: True),
        "edge_r2": _mean(edges, lambda _row: True),
        "invariant_edge_r2": _mean(invariants, lambda _row: True),
        "directed_edge_r2": _mean(edges, lambda row: row.get("base_trait") in directed_names),
    }


def summarize(
    *,
    probe_specs: Sequence[str],
    role_models: Sequence[str],
    geometry_specs: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    records: dict[str, list[dict[str, float]]] = defaultdict(list)
    for value in probe_specs:
        label, model, path = _spec(value)
        records[label].append(_probe_record(model, path))

    role_by_label: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for value in role_models:
        label, model, path = _spec(value)
        role_by_label[label].append((model, path))
    geometry_by_label: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for value in geometry_specs:
        label, model, path = _spec(value)
        geometry_by_label[label].append((model, path))

    rows: list[dict[str, Any]] = []
    for label, seed_records in records.items():
        row: dict[str, Any] = {"experiment": label, "seed_count": len(seed_records)}
        for metric in seed_records[0]:
            values = np.asarray([item[metric] for item in seed_records], dtype=np.float64)
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
        if label in role_by_label:
            primary_values: list[float] = []
            mean_values: list[float] = []
            for role_model, role_path in role_by_label[label]:
                role_rows = [
                    item for item in _read(role_path)
                    if item.get("model") == role_model
                    and item.get("probe") == "mlp"
                    and item.get("split") == "evaluation"
                    and item.get("subset") == "confidence_weighted_all"
                ]
                primary = [item for item in role_rows if item.get("target") == "facing_x_radial_approach"]
                if primary:
                    primary_values.append(float(primary[0]["r2"]))
                mean_values.append(_mean(role_rows, lambda _item: True))
            row["primary_role_binding_r2"] = float(np.nanmean(primary_values))
            row["primary_role_binding_r2_sd"] = (
                float(np.nanstd(primary_values, ddof=1)) if len(primary_values) > 1 else np.nan
            )
            row["mean_role_binding_r2"] = float(np.nanmean(mean_values))
            row["mean_role_binding_r2_sd"] = (
                float(np.nanstd(mean_values, ddof=1)) if len(mean_values) > 1 else np.nan
            )
        else:
            row["primary_role_binding_r2"] = np.nan
            row["primary_role_binding_r2_sd"] = np.nan
            row["mean_role_binding_r2"] = np.nan
            row["mean_role_binding_r2_sd"] = np.nan
        if label in geometry_by_label:
            geometry_records: list[dict[str, str]] = []
            for geometry_model, geometry_path in geometry_by_label[label]:
                matches = [
                    item
                    for item in _read(geometry_path)
                    if item.get("model") == geometry_model
                    and item.get("trial_id") == "A4_C4"
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one A4_C4 geometry row for {geometry_model!r} "
                        f"in {geometry_path}, found {len(matches)}"
                    )
                geometry_records.append(matches[0])
            parameter_counts = {
                int(item["parameter_count"]) for item in geometry_records
            }
            if len(parameter_counts) != 1:
                raise ValueError(
                    f"geometry parameter counts differ across seeds for {label!r}: "
                    f"{sorted(parameter_counts)}"
                )
            row["parameter_count"] = parameter_counts.pop()
            for output_name, source_name in (
                ("total_embedding_variance", "total_variance"),
                ("effective_rank", "entropy_effective_rank"),
                ("permutation_error", "mean_permutation_error"),
            ):
                values = np.asarray(
                    [float(item[source_name]) for item in geometry_records],
                    dtype=np.float64,
                )
                row[output_name] = float(np.nanmean(values))
                row[f"{output_name}_sd"] = (
                    float(np.nanstd(values, ddof=1))
                    if len(values) > 1
                    else np.nan
                )
        rows.append(row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "unified_comparison.csv", rows)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_node_preservation_unified_comparison_v1",
        "model_selection_trial": "A4_C4",
        "excluded_from_selection": ["A5_C5"],
        "seed_aggregation": "mean and sample standard deviation across independent encoder seeds",
        "notes": [
            "Frozen linear probes are the primary common comparison; role binding uses the small nonlinear probe.",
            "A4_C4 is a model-selection validation trial, not an untouched test set.",
            "NaN standard deviation means only one trained seed was evaluated.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", required=True, help="LABEL=MODEL=PATH")
    parser.add_argument("--role-model", action="append", default=[], help="LABEL=MODEL=PATH")
    parser.add_argument("--geometry", action="append", default=[], help="LABEL=MODEL=PATH")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = summarize(
        probe_specs=args.probe,
        role_models=args.role_model,
        geometry_specs=args.geometry,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
