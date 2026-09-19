"""Convert an upstream H ZIP into a canonical Social-GNN node NPZ."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


NODE_CONTRACT_VERSION = "social_node_features_v0"
REQUIRED_FILES = ("latents.npz", "patches.csv", "meta.json")
REQUIRED_COLUMNS = {
    "mouse_id",
    "pair_row",
    "t_start",
    "t_end",
    "interp_frac",
    "drop_flag",
}


class HZipConversionError(ValueError):
    """Raised when an H delivery violates the node contract."""


def _fail(trial_id: str, message: str) -> None:
    raise HZipConversionError(f"Trial {trial_id!r}: {message}")


def _member(root: str, name: str) -> str:
    return f"{root}/{name}" if root else name


def _find_root(names: Sequence[str], requested: str | None) -> str:
    names = tuple(name.replace("\\", "/") for name in names)
    if requested is not None:
        candidates = [requested.strip("/\\")]
    else:
        candidates = sorted(
            {
                name[: -len("/meta.json")] if name.endswith("/meta.json") else ""
                for name in names
                if name == "meta.json" or name.endswith("/meta.json")
            }
        )
    complete = [
        root
        for root in candidates
        if all(_member(root, name) in names for name in REQUIRED_FILES)
    ]
    if len(complete) != 1:
        raise HZipConversionError(
            "Expected exactly one ZIP directory containing latents.npz, "
            f"patches.csv and meta.json; found {complete}"
        )
    return complete[0]


def _number(value: Any, kind: type, trial_id: str, field: str, row: int) -> Any:
    try:
        parsed = kind(value)
    except (TypeError, ValueError):
        _fail(trial_id, f"{field} at CSV row {row} is invalid: {value!r}")
    if kind is float and not np.isfinite(parsed):
        _fail(trial_id, f"{field} at CSV row {row} is not finite")
    return parsed


def _boolean(value: Any, trial_id: str, field: str, row: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    _fail(trial_id, f"{field} at CSV row {row} is not boolean: {value!r}")
    raise AssertionError("unreachable")


def _first_mismatch(left: np.ndarray, right: np.ndarray) -> int | None:
    if left.shape != right.shape:
        return 0
    indices = np.flatnonzero(left != right)
    return int(indices[0]) if indices.size else None


def _clock_summary(starts: np.ndarray, ends: np.ndarray) -> dict[str, Any]:
    durations = ends - starts
    strides = np.diff(starts)
    return {
        "t_start_s": float(starts[0]),
        "t_end_s": float(ends[-1]),
        "duration_s_min": float(durations.min()),
        "duration_s_max": float(durations.max()),
        "stride_s_min": float(strides.min()) if strides.size else None,
        "stride_s_max": float(strides.max()) if strides.size else None,
        "overlapping": bool(strides.size and np.any(starts[1:] < ends[:-1])),
    }


def convert_h_zip_to_node_npz(
    h_zip_path: str | Path,
    output_npz: str | Path,
    *,
    trial_id: str | None = None,
    member_root: str | None = None,
    identity_order: Sequence[str] | None = None,
    output_identity: Sequence[Any] | None = None,
    allow_duplicate_nodes: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """Read H contents in memory and preserve the authoritative patch intervals."""
    h_zip_path = Path(h_zip_path)
    output_npz = Path(output_npz)
    provisional_id = str(trial_id or h_zip_path.stem)
    try:
        with zipfile.ZipFile(h_zip_path) as archive:
            root = _find_root(archive.namelist(), member_root)
            meta = json.loads(
                archive.read(_member(root, "meta.json")).decode("utf-8-sig")
            )
            rows = list(
                csv.DictReader(
                    io.StringIO(
                        archive.read(_member(root, "patches.csv")).decode(
                            "utf-8-sig"
                        )
                    )
                )
            )
            latent_bytes = archive.read(_member(root, "latents.npz"))
    except HZipConversionError:
        raise
    except (
        OSError,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        _fail(provisional_id, f"could not read H ZIP: {exc}")

    if not rows:
        _fail(provisional_id, "patches.csv contains no rows")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        _fail(provisional_id, f"patches.csv is missing columns {sorted(missing)}")
    pair_ids = {
        str(row.get("pair_id", "")).strip()
        for row in rows
        if str(row.get("pair_id", "")).strip()
    }
    if len(pair_ids) > 1:
        _fail(provisional_id, f"patches.csv has multiple pair_id values {pair_ids}")
    meta_id = str(meta.get("pair_id", "")).strip() or None
    csv_id = next(iter(pair_ids), None)
    resolved_id = str(trial_id or meta_id or csv_id or provisional_id)
    for source, value in (("meta.json", meta_id), ("patches.csv", csv_id)):
        if value is not None and value != resolved_id:
            _fail(resolved_id, f"{source} pair_id {value!r} does not match")

    try:
        latent_npz = np.load(io.BytesIO(latent_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        _fail(resolved_id, f"could not read latents.npz: {exc}")
    try:
        latent_keys = {}
        for key in latent_npz.files:
            match = re.fullmatch(r"mouse_(.+)_latent", key)
            if match:
                latent_keys[match.group(1)] = key
        if not latent_keys:
            _fail(resolved_id, "latents.npz has no mouse_<identity>_latent arrays")
        identities = (
            [str(item) for item in identity_order]
            if identity_order is not None
            else sorted(latent_keys)
        )
        if set(identities) != set(latent_keys) or len(identities) != len(latent_keys):
            _fail(
                resolved_id,
                f"identity_order {identities} does not match {sorted(latent_keys)}",
            )
        grouped = {identity: [] for identity in identities}
        for row in rows:
            mouse_id = str(row["mouse_id"])
            if mouse_id not in grouped:
                _fail(resolved_id, f"unexpected mouse_id {mouse_id!r}")
            grouped[mouse_id].append(row)

        latents = []
        starts_by_mouse = []
        ends_by_mouse = []
        masks = []
        interp_fractions = []
        codes = []
        source_files = []
        for identity in identities:
            indexed = sorted(
                (
                    _number(row["pair_row"], int, resolved_id, "pair_row", i + 2),
                    row,
                )
                for i, row in enumerate(grouped[identity])
            )
            pair_rows = np.asarray([item[0] for item in indexed], dtype=np.int64)
            if not np.array_equal(pair_rows, np.arange(len(indexed))):
                _fail(
                    resolved_id,
                    f"identity {identity!r} pair_row must be exactly 0..T-1",
                )
            ordered = [item[1] for item in indexed]
            latent = np.asarray(latent_npz[latent_keys[identity]])
            if latent.ndim != 2 or latent.shape[0] != len(ordered):
                _fail(
                    resolved_id,
                    f"{latent_keys[identity]} shape {latent.shape} does not match "
                    f"{len(ordered)} patch rows",
                )
            if not np.issubdtype(latent.dtype, np.number) or not np.isfinite(latent).all():
                _fail(resolved_id, f"{latent_keys[identity]} is not finite numeric data")
            latents.append(latent.astype(np.float32, copy=False))
            starts_by_mouse.append(
                np.asarray(
                    [
                        _number(row["t_start"], float, resolved_id, "t_start", i + 2)
                        for i, row in enumerate(ordered)
                    ],
                    dtype=np.float64,
                )
            )
            ends_by_mouse.append(
                np.asarray(
                    [
                        _number(row["t_end"], float, resolved_id, "t_end", i + 2)
                        for i, row in enumerate(ordered)
                    ],
                    dtype=np.float64,
                )
            )
            drop = np.asarray(
                [
                    _boolean(row["drop_flag"], resolved_id, "drop_flag", i + 2)
                    for i, row in enumerate(ordered)
                ],
                dtype=bool,
            )
            masks.append(~drop)
            interp_fractions.append(
                np.asarray(
                    [
                        _number(
                            row["interp_frac"],
                            float,
                            resolved_id,
                            "interp_frac",
                            i + 2,
                        )
                        for i, row in enumerate(ordered)
                    ],
                    dtype=np.float32,
                )
            )
            if "code" in ordered[0]:
                csv_codes = np.asarray(
                    [
                        _number(row["code"], int, resolved_id, "code", i + 2)
                        for i, row in enumerate(ordered)
                    ],
                    dtype=np.int64,
                )
                code_key = f"codes_{identity}"
                if code_key in latent_npz.files and not np.array_equal(
                    csv_codes, np.asarray(latent_npz[code_key])
                ):
                    _fail(resolved_id, f"patches.csv code differs from {code_key}")
                codes.append(csv_codes)
            sources = {str(row.get("source_file", "")) for row in ordered}
            source_files.append(next(iter(sources)) if len(sources) == 1 else "")

        starts = starts_by_mouse[0]
        ends = ends_by_mouse[0]
        for identity, other_start, other_end in zip(
            identities[1:], starts_by_mouse[1:], ends_by_mouse[1:]
        ):
            mismatch = _first_mismatch(starts, other_start)
            if mismatch is not None:
                _fail(
                    resolved_id,
                    f"t_start differs for identity {identity!r} at pair_row {mismatch}",
                )
            mismatch = _first_mismatch(ends, other_end)
            if mismatch is not None:
                _fail(
                    resolved_id,
                    f"t_end differs for identity {identity!r} at pair_row {mismatch}",
                )
        if np.any(ends <= starts):
            index = int(np.flatnonzero(ends <= starts)[0])
            _fail(resolved_id, f"patch {index} has non-positive duration")
        if len(starts) > 1 and np.any(np.diff(starts) <= 0):
            index = int(np.flatnonzero(np.diff(starts) <= 0)[0] + 1)
            _fail(resolved_id, f"patch_start_s is not increasing at {index}")

        latent_shapes = {array.shape for array in latents}
        if len(latent_shapes) != 1:
            _fail(
                resolved_id,
                f"all identity latent arrays must share [T,D], got {sorted(latent_shapes)}",
            )
        node_features = np.stack(latents, axis=1)
        node_mask = np.stack(masks, axis=1)
        duplicate_pairs = []
        pair_diagnostics = []
        for left in range(len(identities)):
            for right in range(left + 1, len(identities)):
                exact = bool(
                    np.array_equal(node_features[:, left], node_features[:, right])
                )
                pair_diagnostics.append(
                    {
                        "left": identities[left],
                        "right": identities[right],
                        "exact_duplicate": exact,
                        "mean_absolute_difference": float(
                            np.mean(
                                np.abs(
                                    node_features[:, left] - node_features[:, right]
                                )
                            )
                        ),
                    }
                )
                if exact:
                    duplicate_pairs.append([identities[left], identities[right]])
        if duplicate_pairs and not allow_duplicate_nodes:
            _fail(
                resolved_id,
                f"exact duplicate node streams {duplicate_pairs}; possible upstream "
                "recombination failure. Use --allow-duplicate-nodes only for "
                "interface testing.",
            )

        labels = list(output_identity) if output_identity is not None else identities
        if len(labels) != len(identities) or len(
            {str(item) for item in labels}
        ) != len(labels):
            _fail(
                resolved_id,
                f"output_identity must contain {len(identities)} unique labels",
            )
        identity_array = np.asarray(labels)
        training_eligible = not duplicate_pairs
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract_version": np.asarray(NODE_CONTRACT_VERSION),
            "trial_id": np.asarray(resolved_id),
            "node_features": node_features,
            "node_mask": node_mask,
            "patch_start_s": starts,
            "patch_end_s": ends,
            "identity": identity_array,
            "source_identity": np.asarray(identities),
            "social_step": np.arange(len(starts), dtype=np.int64),
            "interp_fraction": np.stack(interp_fractions, axis=1),
            "drop_flag": ~node_mask,
            "source_file": np.asarray(source_files),
            "training_eligible": np.asarray(training_eligible),
            "upstream_meta_json": np.asarray(
                json.dumps(meta, ensure_ascii=False, sort_keys=True)
            ),
        }
        if len(codes) == len(identities):
            payload["codes"] = np.stack(codes, axis=1)
        np.savez_compressed(output_npz, **payload)

        summary_path = (
            Path(summary_json)
            if summary_json is not None
            else output_npz.with_name(f"{output_npz.stem}_summary.json")
        )
        summary = {
            "contract_version": NODE_CONTRACT_VERSION,
            "trial_id": resolved_id,
            "source_h_zip": str(h_zip_path.resolve()),
            "member_root": root,
            "output_node_npz": str(output_npz.resolve()),
            "identity": [item.item() if isinstance(item, np.generic) else item for item in identity_array],
            "source_identity": identities,
            "shape": list(node_features.shape),
            "valid_node_count": node_mask.sum(axis=0).astype(int).tolist(),
            "clock": _clock_summary(starts, ends),
            "qc": {
                "training_eligible": training_eligible,
                "duplicate_node_pairs": duplicate_pairs,
                "pair_diagnostics": pair_diagnostics,
                "duplicate_override_used": bool(
                    duplicate_pairs and allow_duplicate_nodes
                ),
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["summary_json"] = str(summary_path.resolve())
        return summary
    finally:
        latent_npz.close()


def _identity_label(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        return parsed
    raise HZipConversionError(f"Identity label must be a scalar, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an upstream H ZIP to a canonical Social-GNN node NPZ."
    )
    parser.add_argument("--h-zip", required=True, type=Path)
    parser.add_argument("--output-node-npz", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--trial-id")
    parser.add_argument("--member-root")
    parser.add_argument("--identity-order", nargs="+")
    parser.add_argument("--output-identity", nargs="+")
    parser.add_argument("--allow-duplicate-nodes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = (
        [_identity_label(value) for value in args.output_identity]
        if args.output_identity is not None
        else None
    )
    try:
        summary = convert_h_zip_to_node_npz(
            args.h_zip,
            args.output_node_npz,
            trial_id=args.trial_id,
            member_root=args.member_root,
            identity_order=args.identity_order,
            output_identity=labels,
            allow_duplicate_nodes=args.allow_duplicate_nodes,
            summary_json=args.summary_json,
        )
    except HZipConversionError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
