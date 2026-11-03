# K=8 rSLDS runtime failure audit

Audit date: 2026-11-03.

## Run identity

- Cell: `K=8, M=4, seed=0`.
- Start: `2026-11-02T18:09:12+08:00`.
- Iterations requested: 50.
- Workers: 1.
- Runtime: isolated WSL/Linux `social-rslds` environment.
- Source manifest: `evaluation_runs/rslds_k8_retry/active_k8_m4_seed0.json`.
- Source stderr: `evaluation_runs/rslds_k8_retry/k8_m4_seed0_20261102_180912.stderr.log`.

## Directly observed

1. AR-HMM initialization completed all 25 iterations.
2. Formal rSLDS EM reached 32/50 iterations.
3. The last recorded ELBO was approximately `-3,800,927.6`.
4. The log ends without Python traceback, `MemoryError`, or an `ssm` exception.
5. No final K=8 model/checkpoint/metric artifact was written.
6. WSL recorded an external clean shutdown around the end of the log rather than an in-process model failure.

## Interpretation boundary

The evidence supports the statement “the process was interrupted outside the Python fitting loop.” It does
not identify the unique external cause. Concurrent Inscopix memory pressure is a plausible workstation-level
risk because the application has shown very large memory consumption, but exact causation at this timestamp
has not been established.

## Required controls before the next long retry

- run only one rSLDS cell and one worker;
- keep all model inputs and outputs on local storage;
- record host RAM/commit, WSL RAM, process RSS, and system events on a fixed interval;
- retain separate stdout/stderr plus a heartbeat file;
- add iteration-level checkpoint/resume before another multi-hour K=8 run;
- do not use a partial ELBO trace as a completed-model score.
