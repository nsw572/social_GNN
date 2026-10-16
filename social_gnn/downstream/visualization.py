"""Static diagnostics for provisional temporal states."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import DecodedTrial, bout_rows, normalized_transition_matrix, transition_counts


def save_diagnostic_plots(
    decoded_trials: Iterable[DecodedTrial],
    output_dir: str | Path,
    n_states: int,
) -> list[str]:
    decoded = list(decoded_trials)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for trial in decoded:
        figure, axis = plt.subplots(figsize=(13, 2.4))
        valid = trial.predicted_state >= 0
        axis.scatter(
            trial.trial.patch_center_s[valid] / 60.0,
            trial.predicted_state[valid],
            c=trial.predicted_state[valid],
            s=3,
            cmap="tab20",
            vmin=0,
            vmax=max(n_states - 1, 1),
        )
        axis.set(xlabel="Time (min)", ylabel="State", title=f"{trial.trial.trial_id} state timeline")
        axis.set_yticks(range(n_states))
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = output / f"{trial.trial.trial_id}_timeline.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path.resolve()))

    matrix = normalized_transition_matrix(transition_counts(decoded, n_states))
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, vmin=0, vmax=max(float(matrix.max()), 1e-6), cmap="viridis")
    axis.set(xlabel="Target state", ylabel="Source state", title="Empirical transition probabilities")
    axis.set_xticks(range(n_states))
    axis.set_yticks(range(n_states))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path = output / "transition_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path.resolve()))

    bouts = [row for trial in decoded for row in bout_rows(trial)]
    figure, axis = plt.subplots(figsize=(9, 5))
    for state in range(n_states):
        duration = [row["bout_duration_s"] for row in bouts if row["state"] == state]
        if duration:
            axis.hist(duration, bins=30, histtype="step", linewidth=1.3, label=f"state {state}")
    axis.set(xlabel="Bout duration (s)", ylabel="Count", title="State-duration distributions")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output / "duration_histograms.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path.resolve()))

    embedding = np.concatenate(
        [trial.trial.graph_embeddings[trial.trial.valid_time_mask] for trial in decoded]
    )
    state = np.concatenate(
        [trial.predicted_state[trial.trial.valid_time_mask] for trial in decoded]
    )
    if embedding.shape[0] > 20000:
        index = np.linspace(0, embedding.shape[0] - 1, 20000).astype(int)
        embedding = embedding[index]
        state = state[index]
    projected = PCA(n_components=2, random_state=0).fit_transform(embedding)
    figure, axis = plt.subplots(figsize=(7, 6))
    points = axis.scatter(projected[:, 0], projected[:, 1], c=state, s=2, cmap="tab20", alpha=0.5)
    axis.set(xlabel="Visualization PCA 1", ylabel="Visualization PCA 2", title="Frozen embeddings colored by provisional state")
    figure.colorbar(points, ax=axis, label="State")
    figure.tight_layout()
    path = output / "embedding_pca_states.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path.resolve()))
    return paths
