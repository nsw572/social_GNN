"""Synthetic recovery checks for HMM and explicit-duration HSMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .gaussian_hmm import GaussianHMM
from .gaussian_hsmm import GaussianHSMM


def generate_semimarkov(
    *,
    lengths: list[int],
    means: np.ndarray,
    standard_deviation: float,
    duration_rates: list[float],
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for length in lengths:
        state = int(rng.integers(0, len(duration_rates)))
        labels: list[int] = []
        while len(labels) < length:
            duration = 1 + int(rng.poisson(duration_rates[state]))
            labels.extend([state] * min(duration, length - len(labels)))
            choices = [value for value in range(len(duration_rates)) if value != state]
            state = int(rng.choice(choices))
        label = np.asarray(labels, dtype=np.int64)
        value = means[label] + rng.normal(
            scale=standard_deviation, size=(length, means.shape[1])
        )
        observations.append(value)
        states.append(label)
    return observations, states


def run_synthetic_checks(output_path: str | Path, seeds: list[int]) -> dict:
    scenarios = {
        "well_separated": {
            "means": np.asarray([[-3.0, 0.0], [3.0, 0.0]]),
            "standard_deviation": 0.55,
            "duration_rates": [7.0, 13.0],
        },
        "overlapping_non_geometric_duration": {
            "means": np.asarray([[-0.70, 0.0], [0.70, 0.0]]),
            "standard_deviation": 1.0,
            "duration_rates": [7.0, 15.0],
        },
        "extreme_overlap_stress": {
            "means": np.asarray([[-0.35, 0.0], [0.35, 0.0]]),
            "standard_deviation": 1.0,
            "duration_rates": [7.0, 15.0],
        },
    }
    records = []
    for scenario, settings in scenarios.items():
        train_x, _train_z = generate_semimarkov(
            lengths=[400, 550, 700], seed=101, **settings
        )
        validation_x, validation_z = generate_semimarkov(
            lengths=[475], seed=202, **settings
        )
        target = validation_z[0]
        for seed in seeds:
            hmm = GaussianHMM(2, seed=seed).fit(train_x, max_iter=30)
            hsmm = GaussianHSMM(2, max_duration=40, seed=seed).fit(
                train_x, max_iter=20
            )
            for name, model in (("hmm", hmm), ("hsmm", hsmm)):
                prediction = model.predict(validation_x[0])
                records.append(
                    {
                        "scenario": scenario,
                        "model": name,
                        "seed": seed,
                        "ari": float(adjusted_rand_score(target, prediction)),
                        "validation_ll_per_timestep": float(
                            model.score(validation_x) / validation_x[0].shape[0]
                        ),
                        "minimum_variance": float(model.variances_.min()),
                    }
                )
    summary = []
    for scenario in scenarios:
        for model in ("hmm", "hsmm"):
            selected = [
                row for row in records if row["scenario"] == scenario and row["model"] == model
            ]
            summary.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "mean_ari": float(np.mean([row["ari"] for row in selected])),
                    "mean_validation_ll_per_timestep": float(
                        np.mean([row["validation_ll_per_timestep"] for row in selected])
                    ),
                    "minimum_fitted_variance": float(
                        np.min([row["minimum_variance"] for row in selected])
                    ),
                }
            )
    result = {
        "contract": "social_temporal_synthetic_checks_v1",
        "seeds": seeds,
        "records": records,
        "summary": summary,
        "note": "Synthetic recovery validates implementation behavior, not biological validity.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    print(json.dumps(run_synthetic_checks(args.output, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
