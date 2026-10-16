# SBI Implementation Strategy for joshpy

Oct 16, 2026 · @nick · checked against joshpy v0.0.9.29 and sbi 0.27.0

This is the engineering plan for running the filter-and-refine search in [SBI.md](SBI.md) through joshpy. It is first proved on the published ForeverTree demo (`examples/demo/`), then scaled to the grass–shrub and Joshua tree experiments. Line references are to v0.0.9.29 and will drift; the function names are the stable pointers.

## Decision

**We use [sbi](https://github.com/sbi-dev/sbi) (Mackelab) as the inference engine. joshpy implements no statistics.** joshpy's job is to turn a batch of parameter sets into Josh runs, record every run in the registry, and return summary statistics. The algorithm (proposals, weights, tolerance schedule) comes from sbi.

- **Default method: `sbi.inference.SMCABC`.** This is ABC with sequential Monte Carlo (Toni et al. 2009; Beaumont et al. 2009), the method SBI.md specifies. It's called as `SMCABC(...)(x_o, num_particles, num_initial_pop, num_simulations, epsilon_decay, return_summary=True)`; the summary holds per-generation particles, weights, distances and tolerances. `MCABC` (plain rejection) is available for the coarse first sweep.
- **Room to grow:** the same package provides neural estimators (`NPE_*`, `NLE_A`, `NRE_*`, `NPE_PFN`, flow-matching and score-based methods) and calibration diagnostics (SBC, coverage). They run on the same joshpy plumbing, so switching method is a configuration change. SBI.md Appendix C parks neural methods for now; this plan keeps that option open without committing to it.

### Alternatives considered

| Option | Verdict | Reason |
|---|---|---|
| Optuna (existing `OptunaStrategy`) | Keep for point calibration only | An optimizer, not ABC. It can do rejection ABC with a random sampler, but anything more means writing the method ourselves. TPE concentrates trials near the best fit and does not give a posterior. |
| pyABC | Fallback if we need true resume or adaptive distances | Runs a full SMC-ABC, but proposes one particle at a time and requires its own SQL database for algorithm state. Integration needs a thread pool plus a lock around the registry, and a second database to manage. |
| Stan / NIMBLE | Rejected | Stan needs a differentiable likelihood written in Stan. NIMBLE has no ABC, is R-only, and runs a serial chain. |
| Hand-rolled ABC in joshpy | Rejected | Puts scientific code in joshpy, which we want to avoid. |

### Why sbi fits our infrastructure

Checked against the sbi source; not yet run end to end.

- **No second database.** sbi keeps its state in memory and returns tensors. The joshpy registry is the only persistent store.
- **Batch-shaped simulator.** sbi calls `simulator(theta)` with `theta` of shape `[B, d]` and expects `x` of shape `[B, k]`. Within a generation, `SMCABC` proposes `min(open slots, remaining budget)` candidates per call and repeats until the population is full. **A generation is therefore one full joshpy sweep followed by smaller top-up sweeps** (see Risks). Nothing is simulated beyond the remaining budget, and no thread pool is needed.
- **Single-threaded caller.** Parallelism lives in Josh and Kubernetes, as for any sweep, so the registry needs no locking. Keep sbi's `num_workers=1`: higher values fork worker processes (`simulate_in_batches`), which can't share the registry's DuckDB connection.
- **Dependencies:** torch, nflows, zuko and skorch. A pip dry-run in the project environment (`pixi run`, Python 3.14.3) resolved sbi 0.27.0 with CPU-only torch 2.14.1. A pixi feature solve (conda-forge vs PyPI torch) hasn't been tried yet.

## Architecture

```
SweepManager.run()
  └─ strategy.is_adaptive ?
       ├─ no  → run_sweep(job_set)                      (unchanged)
       └─ yes → run_adaptive_sweep(...)                 (module-level wrapper, kept)
                  └─ builds AdaptiveContext → strategy.run_adaptive(ctx)
                       ├─ OptunaStrategy: ask → ctx.evaluate([params]) → tell
                       └─ SbiStrategy:    sbi calls simulator(theta[B,d])
                                            └─ ctx.evaluate(list_of_param_dicts)
                                                 ├─ register each job (+ attributes)
                                                 ├─ build one zipped CompoundSweepParameter (B rows)
                                                 ├─ run_sweep(...)   local | remote | batch_remote
                                                 ├─ load results into registry (explicitly)
                                                 └─ summaries(registry, run_hash, job) per row → x[B,k]
```

Each library owns its own loop and its own in-memory state. joshpy supplies one function: run this batch of parameter sets, record them, and return their summaries. Future strategies (batched Optuna, reinforcement learning, other SBI libraries) plug in the same way.

`SweepManager.run` must keep calling the module-level `joshpy.strategies.run_adaptive_sweep`, which builds the context and delegates. `tests/test_sweep.py` patches that exact name, so inlining the context construction into `SweepManager.run` would break the "Optuna tests pass unchanged" acceptance.

## Components

### 1. `evaluate_batch`: the shared primitive

```python
def evaluate_batch(
    params: list[dict[str, Any]],   # B parameter sets
    *,
    summaries: SummaryFn,           # (registry, run_hash, job) -> dict[str, float]
    label: str | None = None,       # e.g. "gen-3"; stored as a run attribute
) -> BatchResult:                   # run_hashes[B], run_ids, summaries array[B, k], failures
```

`AdaptiveContext` (cli, config, registry, session_id, execution-mode kwargs) binds this function once as `ctx.evaluate`. Optuna and sbi call the same object; there are no adaptive-specific copies of the register/execute/load logic.

What it does, reusing existing code:

- **Registration.** `run_sweep` does not register job configs. On the batch path, registration happens when `SweepManager` is built; adaptive strategies skip that step and start with an empty job set. So `evaluate_batch` registers each job itself (`register_run`, then `set_attributes(run_hash, **job.attributes)`), as `run_adaptive_sweep` does per trial today. The registry's foreign key rejects runs whose config wasn't registered, so a missed step fails loudly. A test should still pin the registration step directly.
- **Expansion.** A single `CompoundSweepParameter` whose inner `ConfigSweepParameter`s zip the B rows. `CartesianStrategy` expands it into exactly B jobs in label order (`"0"`, `"1"`, ...), so row `i` maps to job `i`. `job.parameters` also gains the compound's name→label entry; drop it when mapping results back to `theta` rows.
- **Execution.** `run_sweep(...)` in whatever mode the caller chose (`local`, `remote`, or `batch_remote` with `batch_no_wait=True`). Routing through `run_sweep` also gives adaptive runs everything the batch path already has (listed under Component 2).
- **Loading.** `run_sweep` ingests automatically only on the `batch_remote` path with `auto_ingest=True`. `evaluate_batch` loads explicitly in every mode (`load_job_results`) and sets `auto_ingest` deliberately rather than inheriting a mode-dependent default. For bucket-resident outputs at scale, summaries can instead be computed with `registry.query_remote(...)` (#81), which reads the CSVs on MinIO/GCS without ingesting. Start with ingest; switch if ingest time dominates.
- **Summaries.** The caller's `summaries` function, with the same signature as today's `ObjectiveFn` but returning a dict. Key order is fixed from the first call.
- **Failures.** A failed run gets NaN summaries and its `run_hash` goes in `failures`; the strategy decides what to do. For SBI.md, a run that dies out of memory is logged as missing, not as "growing".
- **Run IDs.** `BatchResult.run_ids` is threaded back to `SweepManager._last_run_ids` so `manager.load_results()` works after an adaptive run (today it raises). `manager.ingest()` needs its own fix: it takes `label or self.job_set.jobs[0].run_hash` and raises `IndexError` on the empty adaptive job set. Give it a fallback, and add a `run_ids` field to `AdaptiveSweepResult`.

`run_single()` (#85) isn't the right primitive here: it's registry-free and single-job by design, and `evaluate_batch` needs the registry for memoization and provenance.

Every simulation is registered whether or not the inference method later accepts it. That gives SBI.md's "store every run" with no extra work.

### 2. Generic adaptive dispatch

Today `SweepManager.run` sends every adaptive strategy to `run_adaptive_sweep`, which raises unless given an `OptunaStrategy`.

- Add `run_adaptive(ctx: AdaptiveContext) -> Any` to `SweepStrategy`, required for adaptive strategies.
- Move the body of `run_adaptive_sweep` into `OptunaStrategy.run_adaptive`. Optuna calls `ctx.evaluate([params])` with a batch of one, so its behaviour is unchanged. A batched-Optuna mode (`ask` k trials, evaluate them as one batch, then `tell`) is a small later extension.
- Keep `run_adaptive_sweep()` as a thin wrapper, and keep `SweepManager.run` dispatching through it.
- **Delete `_create_single_job`** once Optuna routes through `ctx.evaluate`; don't fix it. Its hand-rolled `run_hash` omits `seed` and `import_files`. Once results are looked up by `run_hash` (Resume, Layer 1), that would cause false cache hits: editing an imported `.josh` file wouldn't change the hash, so old results would be served, and runs with different seeds would collide.

Routing Optuna through `run_sweep` also closes these gaps in the current adaptive path:

- collision policies (`skip` / `pool`)
- `dry_run`, bottling and `on_complete`
- complete run hashes (`seed`, `import_files`)
- `template_vars` / `source_template_path` merging when rendering
- per-trial temp directories that are never cleaned up

**Acceptance:** the existing Optuna tests and the adaptive-sweep tutorial pass unchanged, including the test that patches `run_adaptive_sweep`. New tests: B rows give B registered `job_configs` with attributes, summaries come back in row order, and failures give NaN; `manager.load_results()` and `manager.ingest()` both work after an adaptive run.

### 3. `SbiStrategy`

```python
@register_strategy("sbi")
@dataclass
class SbiStrategy(SweepStrategy):
    method: str = "smcabc"                 # "mcabc" | "smcabc"; later "npe", "nle", ...
    prior: dict[str, dict] = ...           # {"maxNewGrowth": {"uniform": [0.2, 2.0]}, ...}
    summaries: SummaryFn | str = ...       # callable or "module:function"
    observed: dict[str, float] | str = ... # x_o, or a path / "module:function"
    method_kwargs: dict = ...              # passed to sbi: num_particles, num_initial_pop,
                                           # num_simulations, epsilon_decay, ...
    seed: int = 0                          # seeds torch; required for replay-based resume
```

- `is_adaptive = True`. `run_adaptive` builds the sbi prior and a simulator closure (`theta[B, d]` → list of dicts → `ctx.evaluate` → `x[B, k]`). It then calls the sbi method with `return_summary=True`, `simulation_batch_size=None` (the whole proposal batch in one call) and `num_workers=1`.
- **The prior is a plain dict**, so it round-trips through YAML like the other strategies' `to_dict`/`from_dict`. If every parameter is uniform it becomes `sbi.utils.BoxUniform`; otherwise `MultipleIndependent`. Parameter order follows the dict and is recorded in session metadata.
- **Template variables come from the prior's keys.** `SweepConfig.config_parameters` can be empty. Note that `SweepConfig.__len__` raises for adaptive strategies, so nothing on this path may call `len(sweep)`; use truthiness (`__bool__`) instead.
- **Distance:** sbi's `"l2"` on summaries the user has standardized. Scaling is the user's scientific choice and lives in the summaries function or the example, not in joshpy. sbi has no adaptive distance weighting; if that becomes necessary, it's the trigger to revisit pyABC.
- **Result:** an `SbiResult` with the final weighted particles and the per-generation summary. For neural methods it would also hold the trained posterior.

### 4. Recording results in the registry

joshpy records only what the analysis needs, through existing APIs. It doesn't build a second store.

- **Per run, at creation:** `sbi_generation` and `sbi_label` as `JobConfig.attributes` (#84), attached when `evaluate_batch` registers each job.
- **Per run, after each generation:** `registry.set_attributes(run_hash, ...)` with `sbi_weight`, `sbi_distance` and `sbi_accepted=True` for runs in the population. joshpy built each job, so the row-to-`run_hash` mapping is known without passing IDs through sbi. `set_attributes` raises `KeyError` for unknown hashes.
- **Per session:** method, prior spec, parameter order, observed summaries, tolerance per generation, simulation counts, the sbi and torch versions, and CPU/GPU, plus the generation checkpoints (Resume, Layer 2). Add an `update_session_metadata` method to `RunRegistry` and use it here. Today the only writer, Optuna's `_store_study_outcomes`, updates `sweep_sessions.metadata` with raw SQL; `tag_by_session_id` writes to `run_tags` instead. Move `_store_study_outcomes` onto the new method too.
- The outcome groups in SBI.md then become ordinary DuckDB queries joining `config_parameters`, run attributes (`run_tags`) and `cell_data`.

### 5. Packaging

- Add `sbi = ["joshpy[registry]", "sbi>=0.27"]` to `pyproject.toml`, following the existing `adaptive = ["joshpy[registry]", ...]` extra.
- Add a pixi feature and environment that pins CPU-only torch, so CI and the devcontainer don't pull CUDA wheels. pixi has only a `dev` feature today, so this is new.
- `_check_sbi()` raises a clear `ImportError`, mirroring `_check_optuna()`.

## Resume and crash safety

The days-long cost is the Josh simulations, and those are joshpy's responsibility, not sbi's. Resume is a property of the simulation layer, so it works the same way for every strategy (SMC-ABC, neural rounds, Optuna). The recovery point is each individual run, finer than pyABC's built-in resume, which writes results once per generation.

### Layer 1: the registry is the run store

We don't build a separate store. Most of what a run store needs already exists:

| Need | Status |
|---|---|
| Content-addressed run ID | **Exists.** `run_hash` = model + rendered config + input data + seed + imports (`_compute_run_hash`) |
| Per-run status and outcome | **Exists.** `job_runs` (started, completed, exit code, error); `job_configs.status` (`active` / `superseded` / `bad`; a `bad` run is reset and redispatched, #79/#80) |
| Skip completed work on rerun | **Exists.** `collision_policy="skip"` / `"pool"` check existing outputs (local or MinIO) on every dispatch path; `pool` gap-fills exactly the missing replicate indices via `--replicate-indices` (#58, #59) |
| Re-ingest never duplicates data | **Exists.** Ingest identity is `(run_hash, replicate)`; replicates already loaded are skipped (#58) |
| Results written as they land | **Exists for blocking runs. Missing for `batch_no_wait`** (see below) |
| Reattach to Kubernetes jobs still running | **Missing.** `batch_no_wait` keeps job IDs only in an in-memory dict (`_async_dispatched` in `run_sweep`) |

**`batch_no_wait` records dispatch as success.** When a job is dispatched, `run_sweep` calls `RegistryCallback.record`, which runs `start_run` and then `complete_run` with the dispatch command's exit code (0), and skips output registration and ingest. The later poll loop updates only in-memory counters: it never registers outputs, never ingests, and never records the poll outcome. As a result:

- A job that fails on the cluster is recorded as a successful run.
- The registry can't tell "running on the cluster" from "finished but never loaded".

Three changes, all general fixes to batch sweeps rather than SBI-specific:

1. **Split dispatch from completion.** At dispatch, `start_run` only, with the `jobId` and target in `job_runs.metadata` and no exit code. When the poll exits 0, `complete_run`. When it fails, `complete_run` with the failure. Blocking paths keep their current single `start_run` / `complete_run` pair.
2. **Load as jobs complete.** In the poll loop, when a job finishes successfully, run `_register_job_outputs` and ingest for that job, then `on_complete`.
3. **Reconcile on entry.** For each requested `run_hash`, `evaluate_batch` checks the registry before dispatching:
   - **done** (completed and loaded): compute summaries from `cell_data`; no dispatch.
   - **submitted** (has a `jobId`, not completed): poll that job and load it when it finishes.
   - **failed or `bad`:** follow the existing policy (reset and redispatch).
   - **otherwise:** dispatch.

A stored result comes back unchanged, so a run that completed before a crash returns identical summaries after it.

### Layer 2: resuming the algorithm

**SMC-ABC: resume by replaying with a fixed seed.** `SbiStrategy` seeds torch before calling sbi. After a crash, rerun the same strategy:

1. sbi proposes exactly the same parameter sets.
2. They render to the same `.jshc` and the same `run_hash`.
3. Layer 1 returns stored summaries without simulating.
4. The run fast-forwards through completed work in seconds and continues from the crash point.

This depends on:

- the same seed, sbi version, torch version and CPU execution. `SbiStrategy` records all of these in session metadata and refuses to replay on a mismatch.
- deterministic sbi proposals on CPU (verified by the Phase 2 replay test).
- complete run hashes, which is why `_create_single_job` must be deleted, not kept.

In-flight runs are reattached rather than rerun. If a run must be re-simulated (it failed, or its job was lost), the new result is an independent draw: [REPLICATE_BACKFILL.md](REPLICATE_BACKFILL.md) settled that the replicate index is a label, with no per-`(seed, index)` reproducibility. The trajectory then diverges from that point. It is still a valid ABC run, just not identical to an uninterrupted one. **Exact replay holds only when every run that had completed before the crash is served from the registry or reattached.**

**Fallback: warm start from a checkpointed generation.** After each generation, `SbiStrategy` writes the population (particles, weights, tolerance, simulation count) to session metadata. If replay isn't possible, a new `SMCABC` run can start from that population:

- **How:** wrap the prior so that `sample()` returns the checkpointed particles once, while `log_prob()` stays the real prior. sbi draws its initial population from `self.prior.sample(...)`, and set `num_initial_pop = num_particles` (sbi requires `num_particles <= num_initial_pop`). Those parameter sets are already in the registry, so Layer 1 returns their results without simulating.
- **What it gives:** a warm start, not a true resume. sbi resets the weights to uniform and re-derives the tolerance and kernel variance from this population. Later generations' weights still use the real prior, so the result is valid ABC, but its trajectory differs from an uninterrupted run.
- **Caveat:** this relies on private sbi behaviour (`_set_xo_and_sample_initial_population` calling `prior.sample`), which can change between versions. Pin it with a test against the pinned sbi version.

**Neural methods (rounds of simulate → train; Phase 5):**

- Each round draws its parameter sets from a seeded proposal and records them before dispatch, so replay reproduces the round's design and Layer 1 fills in completed runs.
- Training reads parameters (`config_parameters`) and summaries (from `cell_data`) straight from the registry, passed to `append_simulations(theta, x)`. The registry's raw arrays are the canonical record.
- Pickled posteriors are a convenience checkpoint only, because sbi pickles break across versions. Long training uses sbi's `train(resume_training=True)`.

**Optuna gets the same benefit.** With a fixed sampler seed and stored results, replaying a study reaches the same state.

## Execution on Kubernetes

**Current mapping:** one parameter set → one `batchRemote` → one Kubernetes indexed Job with completions equal to `replicates` (one pod per replicate). Each parameter set is staged to its own MinIO prefix (`sweeps/{session}/jobs/{run_hash}/`, assembled by `assemble_batch_workdir`), including its own copy of the `.jshd` inputs. With `batch_no_wait=True`, a batch of B parameter sets becomes B Jobs dispatched together and then polled.

**Replicates are a scientific setting, not the scaling setting.** The inference wants many distinct parameter sets.

- **ABC and the neural methods:** `replicates=1` per parameter set by default.
- **Several simulations per set:** use joshpy's `replicates` (one Job, R pods) and have the summaries function combine the R replicate rows into one `x` row.
  - Don't use sbi's `num_iid_samples` for this. It's meant for distances that compare sets of simulations (MMD, Wasserstein), changes `x` to shape `[B, R, k]`, and is untested with `l2`.
  - Identical parameter rows render to the same `.jshc` and share a `run_hash`, so dispatched as separate jobs they collide under `skip`/`pool`. The row→job mapping must fold duplicate rows into one job with `replicates=R`. Decide and test this in Phase 2.
- **SBI.md's stability check** ("most replicates") needs R > 1 per set for Joshua tree.
- **ForeverTree demo:** `replicates=1`; measured replicate noise is negligible (see below).

**Scaling path:**

1. **Now:** one Job per parameter set. Works today and is adequate for hundreds of parameter sets per batch.
2. **When staging cost matters** (large climate `.jshd`): stage shared data once per session and only the per-set `.jshc` per job. This is a joshpy-side change to `assemble_batch_workdir` and staging. Check first that Josh resolves data from a shared prefix.
3. **When Job overhead matters** (thousands of parameter sets per wave, as in SBI.md's 1,000–2,000-set coarse sweep): a Josh-side multi-config indexed Job, where each pod index maps to a (parameter set, replicate) pair. This is a Josh feature request, not joshpy work.

Measure per-Job overhead (staging, Job creation, scheduling) on the demo before choosing between steps 2 and 3.

## Demo experiment: ForeverTree

The known-truth check that the machinery works.

- **Model:** `examples/demo/forevertree.josh` + `.jshc` + `temperature.jshd` / `precipitation.jshd`. 63 cells, 11 steps, 10 trees per cell.
- **Parameters and prior:** `maxNewGrowth ~ U(0.2, 2.0) m`, `minPrecipImpactPct ~ U(0, 100) %`.
- **Truth:** one run at a chosen point, e.g. (1.0 m, 40%), gives `x_o`.
- **Summaries (2):** at the final step, the landscape mean of `meanHeight`, and the between-cell coefficient of variation of `meanHeight`.
  - The mean alone is confounded: (1.0, 50%) → 7.35 m vs (1.5, 0%) → 7.21 m.
  - The between-cell spread depends almost only on `minPrecipImpactPct`, which breaks the tie. Running the mean-only version as well is a deliberate demonstration of SBI.md's equifinality point.
- **Settings:** `SMCABC`, about 50–100 particles, 4–5 generations, `replicates=1`, 500–1,000 simulations.
- **Pass criteria:**
  - The truth lies inside the final population's central 90% for both parameters.
  - The mean-only run shows the expected ridge.
  - Every simulation is in the registry with its generation label.

**Measured so far (local, DEV jar, 3×3 grid sweep, 3 replicates):**

- About 5–6 s per job, dominated by JVM startup.
- Local jobs run one after another.
- Replicate sd of the landscape mean is about 0.005 m on values of 2–15 m.

Expect about 1 hour locally for 600 simulations, and minutes on Kubernetes.

## Phases

| Phase | Work | Done when | Size |
|---|---|---|---|
| 0 | Fix the two problems the probe found (below). Don't fix `_create_single_job`; it's deleted in Phase 1 | Batch and adaptive paths name configs the same way; the docs note `load_results()` | ½ day |
| 1 | `evaluate_batch` + `AdaptiveContext` (with per-job registration); move the Optuna loop into `OptunaStrategy.run_adaptive`; keep module-level `run_adaptive_sweep` dispatch; thread `run_ids` back and fix `ingest()`; delete `_create_single_job` | Optuna tests and tutorial unchanged; `evaluate_batch` tests pass; `load_results()` and `ingest()` work after an adaptive run | 1–1½ days |
| 1b | Resume, Layer 1: split dispatch from completion in `batch_no_wait`; load on poll success; registry lookup and reconcile-on-entry in `evaluate_batch` | A rerun of a completed batch dispatches nothing; killing the process mid-batch and rerunning reattaches to running jobs and dispatches only the missing ones; a `batch_no_wait` job that fails on the cluster is recorded as failed | 1–1½ days |
| 2 | `SbiStrategy` (`mcabc`, `smcabc`) with `seed` and version guard; registry recording; `update_session_metadata`; generation checkpoints; `joshpy[sbi]` extra and pixi feature | Unit tests with a stub CLI; the demo recovers the truth locally; replay test (kill mid-generation, rerun, final particles identical to an uninterrupted run); warm-start test pinned to the sbi version | 1½–2 days |
| 3 | Demo notebook/tutorial (`docs/tutorials/sbi-abc.qmd`) | Renders in CI, or as a frozen render because of runtime | ½–1 day |
| 4 | Kubernetes run of the demo with `batch_remote` | Same result as local; per-Job overhead measured | ½ day |
| 5 (later) | Neural methods through the same strategy (`method="npe"`, `"nle"`, ...), plus sbi SBC/coverage diagnostics | Recovery test passes and agrees with the ABC result | when needed |

Phases 0–4 total about 5–7 days.

### Problems found while probing

- **Config naming differs between paths.** The batch path names the rendered config from `JobConfig.config_name`, defaulting to `sweep_config.jshc`. The adaptive path (`_create_single_job`) names it from the template stem and ignores `config_name`. The ForeverTree model reads `config forevertree.*`, so batch sweeps need `config_name="forevertree"` passed explicitly. `evaluate_batch` goes through the batch path, so the two must agree.
- **Results aren't loaded on local and remote runs.** `SweepManager.run()` doesn't load results into `cell_data` there; the caller must call `load_results()`. `batch_remote` with the default `auto_ingest=True` does ingest. `evaluate_batch` loads explicitly in every mode.

## Risks and open questions

- **Replay depends on determinism.** If the Phase 2 replay test fails, use the warm start from the last checkpointed generation (valid ABC, different trajectory). If that proves insufficient too, pyABC becomes the better choice.
- **Shrinking batches.** Each generation is one full batch followed by smaller top-up batches until the population is full. On Kubernetes, each top-up pays queue latency again. Watch this in Phase 4.
- **Reusing ABC simulations for neural training.** Simulations from later ABC generations aren't drawn from the prior. Likelihood and ratio estimators (`NLE`, `NRE`) are valid with any proposal; posterior estimators need their sequential, proposal-corrected variants. Every run records its generation so this stays possible.
- **Float rendering and run hashes.** Continuous draws render into `.jshc` at full precision, so every draw gets a unique `run_hash`, and `skip`/`pool` will never reuse a run across draws. Those full-precision strings also land in each job's `custom_tags`; worth watching at 1,000+ runs per generation.
- **Summary scaling.** `l2` on unscaled summaries lets the largest summary dominate. Standardizing is part of each experiment's summaries function and goes in its rules sheet (SBI.md workplan step 5) before the first wave.
- **Fire scenarios are per-set data files, but the prior only feeds template variables.** SBI.md drives uncertain fire parameters through externally generated scenario files, and `evaluate_batch` takes `params: list[dict]` with no per-set data file. `FileSweepParameter` entries inside the zipped compound can carry fixed files, and generated scenarios could be staged per `run_hash`, but the prior→job mapping needs a file-parameter design before the Joshua tree experiment. Decide it in Phase 2.
- **Outcome-group labels need fixed attribute names.** SBI.md's four groups (stability × data check) and its declining/replacing/growing labels come from the user's scoring function and should land as run attributes (e.g. `check_stability`, `check_data`, `group`), so the four-group queries stay ordinary `run_tags` joins. Fix the names in the rules sheet alongside summary standardization.
- **sbi API stability.** sbi changes quickly between minor versions. Pin a compatible range and cover the adapter, including the warm-start wrapper, with tests.
- **torch packaging.** The pip dry-run resolved, but the pixi feature solve hasn't been tried. If it fails, install sbi and torch from PyPI inside the pixi feature rather than from conda-forge.

## Source references (v0.0.9.29)

- `batch_no_wait` dispatch recorded as complete: `joshpy/jobs.py:2336-2350` → `RegistryCallback.record` (`joshpy/registry.py:4623`); poll loop: `joshpy/jobs.py:2464-2531`.
- Build-time registration (batch path): `joshpy/sweep.py:2286`; per-trial registration (adaptive): `joshpy/strategies.py:1044`.
- `ingest()` identifier: `joshpy/sweep.py:1786`; `_last_run_ids` set only on the batch branch: `joshpy/sweep.py:1705`.
- Adaptive run hash missing `seed`/`import_files`: `joshpy/strategies.py:1297-1301` vs `joshpy/jobs.py:1451-1457`.
- Raw-SQL session metadata: `_store_study_outcomes`, `joshpy/strategies.py:1378`.
- Dispatch test patch target: `tests/test_sweep.py:853`.
- sbi 0.27.0: `SMCABC._set_xo_and_sample_initial_population` (assert `num_particles <= num_initial_pop`, `prior.sample`), `SMCABC._sample_next_population` (proposal rounds), `simulate_in_batches` (`None` batch size, `num_workers` multiprocessing).
