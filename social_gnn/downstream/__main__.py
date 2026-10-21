"""Command dispatcher for the downstream temporal pipeline."""

from __future__ import annotations

import sys


HELP = """usage: python -m social_gnn.downstream COMMAND [options]

Commands:
  extract    Extract frozen patch-level graph embeddings.
  search     Fit/search Gaussian HMM or explicit-duration Gaussian HSMM models.
  benchmark  Compare KMeans, diagonal GMM, Gaussian HMM, and Gaussian HSMM.
  aggregate  Aggregate multiseed benchmark tables, agreement, and plots.
  ari-stability  Decompose saved confirmation ARIs by encoder/downstream seeds.
  consistency-audit  Audit video timing and HMM state correspondence.
  montage    Export longest-bout source-video montages for one saved HMM run.
  export     Decode trials and export states, bouts, summaries, and figures.
  synthetic  Run synthetic recovery and numerical checks.

Run `python -m social_gnn.downstream COMMAND --help` for command options.
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(HELP)
        return 2
    command = sys.argv.pop(1)
    if command in {"-h", "--help"}:
        print(HELP)
        return 0
    if command == "extract":
        from .embeddings import main as run
    elif command == "search":
        from .search import main as run
    elif command == "benchmark":
        from .benchmark import main as run
    elif command == "aggregate":
        from .aggregate import main as run
    elif command == "ari-stability":
        from .ari_stability import main as run
    elif command == "consistency-audit":
        from .temporal_state_consistency import main as run
    elif command == "montage":
        from .video_montage import main as run
    elif command == "export":
        from .state_export import main as run
    elif command == "synthetic":
        from .synthetic import main as run
    else:
        raise SystemExit(f"unknown downstream command {command!r}")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
