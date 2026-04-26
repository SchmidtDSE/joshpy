# PR7+ Plan: joshpy-side follow-ups for batch-run hardening

This doc tracks joshpy-side changes that build on (or extend) [JOSH_FIXES.md](JOSH_FIXES.md). Items here are NOT pre-merge blockers for `feat/batch-run` → `main` — except item 6 (the static collision check), which closes a silent-overwrite footgun and should land before merging.

The numbering continues from `JOSH_FIXES.md` (1–3 = josh-side merged, 4–6 = josh-side TODO). Items 4–7 below are joshpy-side.

| # | Item | Depends on | Pre-merge? |
|---|------|------------|------------|
| 4 | Auto-inject `run_hash` (and other tags) into `BatchRemoteConfig` as custom-tags | josh#5 (Fix 5 in JOSH_FIXES.md) | No — post-merge |
| 5 | Pre-dispatch MinIO list + replicate-offset computation + collision policy on builder | josh#6 (Fix 6) | No — post-merge |
| 6 | Static collision check in `SweepManager.build()` / `.run()` | nothing | **Yes** |
| 7 | (optional) Hash `seed` in `_compute_run_hash` if explicitly set | nothing | No — separate small PR |

---

## Item 4: Auto-inject `run_hash` (and user tags) at batch-remote dispatch

### Why

Pre-merge state: `BatchRemoteConfig` has no `custom_tags` field (was removed in PR5 because batchRemote didn't accept them). Once josh adds `--custom-tag` passthrough on `batchRemote` ([JOSH_FIXES.md Fix 5](JOSH_FIXES.md#fix-5-batchremote-accepts---custom-tag-and-propagates-to-pods)), joshpy can put it back and use it to inject canonical tags every batch dispatch should include — most importantly `run_hash`, which lets users put `{run_hash}` in their export paths.

This is the principled alternative to pre-interpolating `{run_hash}` in `sim.josh` at stage time (which would break the "same .josh runs everywhere" invariant).

### Files & changes

| File | Change |
|------|--------|
| [`joshpy/cli.py`](joshpy/cli.py) | Re-add `custom_tags: dict[str, str] = field(default_factory=dict)` to `BatchRemoteConfig`. Wire `--custom-tag` flag emission in `JoshCLI.batch_remote()`, mirroring the pattern in `RunConfig` / `RunRemoteConfig`. |
| [`joshpy/jobs.py`](joshpy/jobs.py) | In `to_batch_remote_config()`, auto-populate `custom_tags = {"run_hash": job.run_hash, **job.custom_tags}`. User tags can override the auto-injected ones if a user explicitly sets `run_hash` themselves (unlikely but defensive). |
| [`joshpy/jobs.py`](joshpy/jobs.py) | Same auto-injection in `to_run_config()` and `to_run_remote_config()` — currently `run_hash` is in `job.custom_tags` from job expansion, but verify and unify. |
| [`tests/test_cli.py`](tests/test_cli.py) | Re-introduce the `test_custom_tags` case for batch (was dropped in PR5) — assert `--custom-tag run_hash=...` is in the args. |
| [`tests/test_jobs.py`](tests/test_jobs.py) | Assert `to_batch_remote_config(job, target, prefix).custom_tags["run_hash"] == job.run_hash`. |

### Verification

E2E: a fixture with `exportFiles.patch = "minio://bucket/exp/{run_hash}/output_{replicate}.csv"` should write CSVs into a folder named with the actual run hash. Test 4's existing fixture is a good base — modify the export path to use `{run_hash}` and re-run.

### Notes

- Keep `BatchRemoteConfig.custom_tags` empty by default in the dataclass. The auto-injection is in `to_batch_remote_config()` (the joshpy convention), not in the JAR-wrapper layer (the josh CLI mirror). That preserves the principle that `JoshCLI.batch_remote(BatchRemoteConfig(...))` is a faithful 1:1 mirror of the JAR CLI.
- Document the auto-injection convention in the `to_batch_remote_config` docstring.

---

## Item 5: Pre-dispatch MinIO listing + replicate-offset dispatch + collision policy

### Why

Once josh adds `--replicate-start=K` ([JOSH_FIXES.md Fix 6](JOSH_FIXES.md#fix-6-run-accepts---replicate-startk-entrypoint-adds-josh_replicate_offset)), joshpy can support real pool/resume semantics for sweeps with stable run hashes. This is the `pool` collision policy the user requested:

> If there's a change, new run_hash, no collision. If no change, see run csvs in output, start with new idx, no collision.

### Behavior

```python
manager = (
    SweepManager.builder(config)
    .with_registry("exp.duckdb")
    .with_cli(cli)
    .with_batch_remote("gke-test")
    .with_collision_policy("pool")  # default; or "replace" / "skip" / "fail"
    .build()
)
manager.run()
```

For each `ExpandedJob` in the sweep, `manager.run()` does:

1. Compute the export path template via `cli.inspect_exports(...)` (cache once per sweep).
2. Resolve the per-job MinIO prefix that would contain its outputs.
3. List existing objects under that prefix; extract replicate indices from filenames matching the template.
4. Apply the policy:
   - **`pool`**: K = max(existing) + 1; if K ≥ job.replicates skip; else dispatch with `replicate_start=K, replicates=(job.replicates - K)`.
   - **`replace`**: delete existing objects; dispatch from K=0 with full replicate count.
   - **`skip`**: if K == job.replicates, no-op (idempotent re-runs in CI).
   - **`fail`**: raise `SweepCollisionError` listing the conflicting paths. Default for safety.

### Files & changes

| File | Change |
|------|--------|
| [`joshpy/cli.py`](joshpy/cli.py) | `BatchRemoteConfig` adds `replicate_start: int = 0`. `JoshCLI.batch_remote()` emits `--replicate-start=K` when nonzero. |
| [`joshpy/jobs.py`](joshpy/jobs.py) | New helper `_list_existing_replicates(cli, registry, run_hash, export_paths) -> set[int]`. Uses MinIO `ls` (single subprocess via `cli.stage_from_minio` is overkill — direct `boto3` or `gcloud storage ls` parsing). |
| [`joshpy/jobs.py`](joshpy/jobs.py) | `to_batch_remote_config()` accepts `replicate_start` kwarg; defaults to 0. |
| [`joshpy/jobs.py`](joshpy/jobs.py) | `run_sweep()` batch-remote branch consults the policy: lists existing → computes start/count → dispatches. |
| [`joshpy/sweep.py`](joshpy/sweep.py) | New `SweepCollisionError` exception class. `SweepManagerBuilder.with_collision_policy(policy: Literal["pool","replace","skip","fail"])`. Threaded through to `run_sweep` via `SweepManager`. |
| [`joshpy/registry.py`](joshpy/registry.py) | Add `min_replicate_id` and `max_replicate_id` columns to `job_runs` (or a new `run_replicate_ranges` table). Schema migration for existing registries. |
| [`tests/`](tests/) | Unit tests for each policy branch; E2E: dispatch 0–4, dispatch 5–9, verify final state. |

### Open questions

- **MinIO listing mechanism**: `cli.stage_from_minio` to a temp dir + listdir works but is wasteful. Direct S3 listing via `httpfs`/DuckDB or boto3 is faster. We already have `configure_s3()` for DuckDB; `SELECT name FROM glob('s3://bucket/prefix/*')` might work. Verify.
- **Path-template parsing**: extracting replicate indices from existing filenames requires parsing the template (find the position of `{replicate}`, regex-match against filenames). Edge case: template uses `{step}` or `{variable}` which produces multiple files per replicate. The first MVP can be `{replicate}`-only and bail out on more complex templates.
- **Registry schema migration**: how to add columns without breaking existing DBs. We have a precedent in earlier registry changes; check the migration utilities.

### Verification

- E2E: dispatch sweep with N=5, succeed. Re-run same sweep with N=10 + policy="pool". Result: 10 CSVs total in MinIO; registry shows 10 replicates for the run_hash.
- E2E: as above with policy="replace". Result: 10 fresh CSVs; old ones gone.
- E2E: as above with policy="fail". Raises SweepCollisionError with helpful message.

---

## Item 6: Static collision check in `SweepManager` (pre-merge candidate)

### Why pre-merge

The current `feat/batch-run` lets users silently overwrite their own MinIO outputs by running the same sweep twice with an export path template like `output_{replicate}.csv`. Registry says "10 replicates total" but MinIO only has 5 files (the latest run's). Silent data corruption — the kind of failure mode that erodes trust in batch results.

The static check requires no josh changes and catches this at `SweepManager.build()` time. Worth landing before main merge.

### Behavior

At `SweepManager.build()` (or first `.run()` call — design choice; build-time is earlier-feedback):

1. `cli.inspect_exports(...)` → patch path template.
2. If template contains `{timestamp}` OR `{run_hash}` → safe; skip check.
3. Else, for each job in the sweep, query `registry.get_runs_for_hash(job.run_hash)`.
4. If any prior runs exist, **raise `SweepCollisionError`** with the conflicting hashes and remediation guidance.

When item 5's `with_collision_policy()` lands, the static check becomes a special case: `policy="fail"` (the default) calls this check; other policies bypass it (they have their own runtime handling).

For pre-merge, we ship just the check + a `force=True` kwarg as the escape hatch. After item 5 lands, `force=True` becomes equivalent to `with_collision_policy("replace")`.

### Files & changes

| File | Change |
|------|--------|
| [`joshpy/sweep.py`](joshpy/sweep.py) | Add `SweepCollisionError(Exception)` class. New helper `_check_export_path_safety(cli, job_set, registry, export_paths)`. |
| [`joshpy/sweep.py`](joshpy/sweep.py) | `SweepManager.run(force: bool = False, ...)`. When `force=False` and `batch_remote=True`, run the check. Raise on conflict; print actionable error including the prior runs' hashes and three remediation options. |
| [`joshpy/sweep.py`](joshpy/sweep.py) | Caches the inspect_exports call (one subprocess per sweep, not per job). |
| [`tests/test_sweep.py`](tests/test_sweep.py) | Unit: template with `{timestamp}` → check passes. Template with only `{replicate}` + prior runs → raises. Template with only `{replicate}` + no prior runs → passes. `force=True` → passes regardless. |

### Recommended error message

```
SweepCollisionError: 1 job in this sweep would silently overwrite prior MinIO outputs.

Run hash a347d7d05d74 has 1 prior run(s) in the registry, and your export path
template ('minio://bucket/output_{replicate}.csv') doesn't include {timestamp}
or {run_hash} — re-dispatching will overwrite the existing CSVs while the
registry still references them. The registry would report more replicates than
MinIO actually contains.

Fix one of:
  1. Add {timestamp} to your export path (recommended — every dispatch
     gets a fresh folder):
         exportFiles.patch = "minio://bucket/{timestamp}/output_{replicate}.csv"

  2. Once josh#5 ships and joshpy auto-injects run_hash as a custom tag,
     use {run_hash} for deterministic per-simulation paths:
         exportFiles.patch = "minio://bucket/{run_hash}/output_{replicate}.csv"

  3. Drop the prior run(s) from the registry if you intend a fresh re-run:
         registry.delete_run("a347d7d05d74")

  4. Pass force=True to SweepManager.run() to proceed anyway (you accept
     that subsequent ingest() calls will count duplicate replicates).
```

### Verification

- Unit tests as above.
- E2E: re-run the existing test6 fixture twice without changing the path; second run should raise `SweepCollisionError`. Add `force=True`; second run should proceed.

### Risk

**LOW.** Pure check; new exception class. Existing tests that re-use registries across runs will need the `force=True` kwarg or registry-cleanup setUp; ~5 test files affected. Update them as part of the PR.

---

## Item 7: Hash `seed` in `_compute_run_hash` if explicitly set

### Why

Today's [`_compute_run_hash`](joshpy/jobs.py#L124-L183) hashes `.josh` content, rendered `.jshc` content, and `.jshd` file contents — but NOT `seed`. Two runs with otherwise-identical inputs but `seed=42` vs `seed=99` produce the same `run_hash` and write to the same MinIO path. Aggregate statistics across the pool are still valid (same simulation specification, same expected distribution), but per-replicate trajectories diverge — bad for reproducibility-as-trajectory use cases.

### Proposal

In `_compute_run_hash`, if `seed is not None`, include it:

```python
# 4. Random seed (if explicitly set; preserves hash backward-compat for unseeded runs)
if seed is not None:
    hasher.update(b"seed=")
    hasher.update(str(seed).encode("utf-8"))
```

Hashes for runs that don't set `seed` are unchanged (preserves existing registry data). Runs that do set `seed` get a hash that disambiguates seed values, so `seed=42` and `seed=99` end up in different MinIO folders even with `{run_hash}` paths.

### Files & changes

| File | Change |
|------|--------|
| [`joshpy/jobs.py`](joshpy/jobs.py) | `_compute_run_hash` accepts `seed: int | None = None` kwarg; hashes when not-None. |
| [`joshpy/jobs.py`](joshpy/jobs.py) | All `_compute_run_hash` call sites pass `job.seed`. |
| [`tests/test_jobs.py`](tests/test_jobs.py) | Test: same inputs without seed → same hash. Same inputs with same seed → same hash. Same inputs with different seeds → different hashes. Same inputs with seed=None vs seed=42 → different hashes (intentional — the user explicitly asked for reproducibility). |

### Risk

**LOW** for unseeded runs (no behavior change). **MEDIUM-LOW** for seeded runs in active registries: existing registered hashes for seeded runs would no longer match the new computation. If anyone has a registry with seeded runs, they'd see those runs as new on re-registration. Worth a release note, but not a data-loss issue.

### Open question

Whether to make this opt-in via a `JobConfig.hash_seed: bool = False` flag for the first release, then flip the default later. Probably not necessary — the change is principled and the migration cost is low.

---

## Sequencing

```
Pre-merge (feat/batch-run → main):
  ├─ josh: Fix 4 (DNS retry)        ─┐
  └─ joshpy: Item 6 (static check)  ─┘  (independent; can land in parallel)

Post-merge:
  ├─ josh: Fix 5 (--custom-tag)
  │   └─ joshpy: Item 4 (auto-inject run_hash)
  ├─ josh: Fix 6 (--replicate-start)
  │   └─ joshpy: Item 5 (pool/resume + collision policy)
  └─ joshpy: Item 7 (hash seed)  (orthogonal, anytime)
```

Item 6 alone closes the silent-overwrite footgun for the immediate merge. Items 4 & 5 are the longer-term polish that turns batch-run from "works but tread carefully" into "intuitive and idempotent."
