# Sticky HMM transition-prior implementation audit

The implementation uses a row-wise Dirichlet pseudocount matrix inside every
EM transition update. For timestep `dt` and target mean geometric dwell `tau`,
the prior target is `p_stay = 1 - dt / tau`. At `dt=0.04` s and
`tau=2.0` s this is `0.98`.

For K=6, a strength S contributes total mass S per source-state row:
`S * 0.98` on the diagonal and `S * (0.02 / 5)` to each off-diagonal cell.
The ordinary isotropic transition pseudocount `0.01` is then added to every
cell, so the exact total prior row mass is `S + 0.06`. Strength is therefore
not diagonal-only mass.

At every EM iteration, the implementation initializes transition counts with
this prior matrix, adds posterior expected soft transition counts `xi` from
forward-backward separately for every valid sequence, and row-normalizes the
sum. It never inserts a transition at a trial boundary or invalid gap. The
same target matrix also initializes the transition probabilities before the
first E-step. No transition clipping or probability floor is applied to the
M-step result; a numerical floor of `1e-300` is used only before logarithms.

The exported `transition_prior_vs_counts.csv` contains exact post-fit soft
expected counts recomputed with each fitted model, not Viterbi hard counts or
patch occupancy. `transition_diagnostics.npz` preserves the full learned
matrix, prior matrix, and soft expected-count matrix for every newly fitted
run. Existing formal 50000/2 s checkpoints remain unchanged and are referenced
by hash.
