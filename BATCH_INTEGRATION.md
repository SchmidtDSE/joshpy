# Plan: Batch Remote Execution for joshpy

Tracking issue: [joshpy#31](https://github.com/SchmidtDSE/joshpy/issues/31)
Companion Java plan: [josh#374](https://github.com/SchmidtDSE/josh/issues/374)
Dependency: [josh#406](https://github.com/SchmidtDSE/josh/issues/406) — `pollBatch` CLI command for async job status polling

## Context

joshsim (Java) has added `batchRemote` — a parallel execution path using MinIO staging and target profiles instead of HTTP streaming. PRs 1-7 are merged on the Java side ([josh#374](https://github.com/SchmidtDSE/josh/issues/374)). joshpy needs to wrap these new capabilities and provide efficient Python-level orchestration for parameter sweeps.

**Immediate motivation:** A production run has 5 of 6 replicate CSVs sitting in MinIO (the 6th OOM'd). The run is registered in the local RunRegistry with a label. We need a way to recover those results NOW — look up the run by label, discover the `minio://` export paths, read the CSVs directly into DuckDB via S3, and load them into the registry. This drives the PR ordering: result ingestion first, then the rest of the batch infrastructure.

**Access model (Model A):** MinIO/S3 CSVs are the source of truth. The local `.duckdb` is a materialized cache that any machine can rebuild by re-ingesting from S3. DuckDB reads CSVs directly from S3 via `httpfs` — no download, no local disk needed for the CSV data. This supports future access patterns: browser WASM reading S3, serverless aggregators attaching `.duckdb`, multi-machine access.

**State ownership:** josh is stateless/ephemeral — it dispatches jobs and can check their status, but holds no long-running state. joshpy owns all state via RunRegistry (what was run, parameters, label, job ID). When joshpy dispatches a `--no-wait` batch job, it stores the `batch_job_id` in `job_runs.metadata`. To poll, joshpy calls josh's `pollBatch` CLI command ([josh#406](https://github.com/SchmidtDSE/josh/issues/406)) which knows HOW to check status for each target type (MinIO status file, K8s Job API, etc.). joshpy doesn't know or care about the polling mechanism internals — it just gets back "running" / "complete" / "error".

**Key design decisions:**
- `batchRemote` has no `--data` flags — files stage to/from MinIO. First positional arg can be a `.josh` file OR a directory. The caller stages data; the worker pulls via `stageFromMinio`.
- Auto-pull results from MinIO after jobs complete, with opt-out for fire-and-forget. Plus a generic "ingest CSVs after the fact" code path that works for both batch remote AND local OOM recovery (DRY).
- Target config system is SHARED between josh and joshpy — joshpy reads AND creates `~/.josh/targets/<name>.json`.
- MinIO cred resolution hierarchy (mirrors joshsim's `HierarchyConfig`): CLI flags > profile JSON > env vars (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`). Secrets don't need to live in profile JSON.
- K8s targets have a separate `pod_minio_endpoint` — the in-cluster MinIO endpoint pods use, which may differ from the outer `minio_endpoint` used for host-side staging.
- For sweeps: stage shared data (.josh, .jshd) to MinIO ONCE, then per-job stage only the unique .jshc config. joshpy orchestrates staging directly (not via `batchRemote`).

---

## Validated Access Pattern (2026-04-16)

Proved out end-to-end against GKE Autopilot + GCS (S3-compatible). The client (joshpy / devcontainer) never runs simulations — it orchestrates:

```
Client (joshsim CLI)                    GCS (S3-compatible)                 GKE Autopilot
─────────────────────                   ────────────────────                ──────────────
1. Stage local files ──────────────────→ batch-jobs/<jobId>/inputs/
2. Create K8s Secret (MinIO creds) ────────────────────────────────────→ josh-creds-<jobId>
3. Create K8s Job ─────────────────────────────────────────────────────→ josh-<jobId>
4. Poll Job API ───────────────────────────────────────────────────────→ status?
                                                                         │
                                        Pod starts:                      │
                                        ← stageFromMinio (inputs)        │
                                        → run simulation                 │
                                        → write results to GCS ─────────→ gke-test-results/smoke_3.csv
                                                                         │
5. Poll returns COMPLETE ←─────────────────────────────────────────────←─┘
6. (preprocessBatch only) Download result ←── batch-jobs/<jobId>/outputs/output.jshd
```

### What the client needs

- **The fat JAR** — `java -jar joshsim-fat.jar batchRemote ...`
- **kubectl context** — `gcloud container clusters get-credentials` (Fabric8 reads `~/.kube/config`)
- **MinIO/GCS credentials** — `MINIO_ACCESS_KEY` + `MINIO_SECRET_KEY` as env vars
- **A target profile** — `~/.josh/targets/<name>.json` with cluster context, namespace, image, resource requests, GCS bucket
- **A .josh simulation file** — with `minio://` export paths pointing at the GCS bucket

### Commands validated

```bash
# batchRemote — run simulation on K8s, results land in GCS
java -jar joshsim-fat.jar batchRemote sim.josh SimName \
  --target=gke-test --replicates=5

# preprocessBatch — preprocess data on K8s, download result .jshd
java -jar joshsim-fat.jar preprocessBatch sim.josh SimName \
  data.nc variable units output.jshd --target=gke-test

# No-wait mode — dispatch and exit, check status later
java -jar joshsim-fat.jar batchRemote sim.josh SimName \
  --target=gke-test --replicates=10 --no-wait
```

### Implications for joshpy

- **K8s targets require the Java CLI** — it uses the Fabric8 K8s client to create Jobs and Secrets directly. There is no HTTP intermediary. joshpy must shell out to `java -jar joshsim-fat.jar batchRemote ...`.
- **HTTP targets can use direct POST** — `POST /runBatch` is the HTTP equivalent, which joshpy could call directly (PR 5 optimization).
- **Env vars are the credential transport** — set `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` before invoking the JAR. The JAR resolves them via HierarchyConfig.
- **GKE cluster is already running** — server-side infrastructure is deployed. joshpy just needs the client-side plumbing.

---

## PR Plan

```
PR1 (S3-native ingest) ✅ DONE → PR2 (target profiles) → PR3 (CLI wrappers) → PR4 (sweep integration) → PR5 (shared staging optimization) → PR6 (polish)
```

### Regression gates (every PR)
- `pixi run test` passes (867 unit tests, integration tests excluded)
- `pixi run test-integration` passes (17 MinIO integration tests, CI only)
- Existing `runRemote` path completely untouched

---

### PR 1: Result Recovery — S3-native `ingest_results()` ✅ DONE

**Status:** Merged via [joshpy#32](https://github.com/SchmidtDSE/joshpy/pull/32). All code shipped, unit tests passing, MinIO integration tests added.

#### What shipped

| Component | File | Description |
|-----------|------|-------------|
| `configure_s3()` | `joshpy/registry.py` | DuckDB httpfs + S3 credential setup (parameterized `use_ssl`) |
| `CellDataLoader.load_csv()` | `joshpy/cell_data.py` | Accepts `s3://` URL strings alongside local `Path` |
| `ingest_results()` | `joshpy/sweep.py` | Full recovery by label: metadata → exports → S3 read → load |
| `_resolve_ingest_metadata()` | `joshpy/sweep.py` | Helper: label/hash → run metadata |
| `_get_josh_source()` | `joshpy/sweep.py` | Helper: josh file from disk or stored content |
| `_configure_minio_access()` | `joshpy/sweep.py` | Helper: S3 direct read or stageFromMinio download |
| `_load_ingest_replicates()` | `joshpy/sweep.py` | Helper: per-replicate CSV loading with graceful skip |
| `StageFromMinioConfig` | `joshpy/cli.py` | Config for `stageFromMinio` CLI command |
| `JoshCLI.stage_from_minio()` | `joshpy/cli.py` | `download=True` fallback path |
| `SweepManager.ingest()` | `joshpy/sweep.py` | Convenience wrapper for `ingest_results()` |

#### CI infrastructure shipped alongside PR 1

| Component | File | Description |
|-----------|------|-------------|
| Unit test workflow | `.github/workflows/test.yml` | `unit-tests` job: 867 tests via pixi |
| Integration test workflow | `.github/workflows/test.yml` | `integration-tests` job: MinIO service container + JAR |
| MinIO test simulation | `tests/fixtures/minio_export.josh` | Minimal .josh with `minio://` exports |
| Shared fixtures | `tests/conftest.py` | `minio_conn`, `minio_registry`, `seed_csv`, `josh_cli`, etc. |
| Integration tests | `tests/test_minio_integration.py` | 17 tests across 6 escalating levels |
| Pytest marker | `pyproject.toml` | `integration` marker registered |
| Pixi tasks | `pixi.toml` | `test` (unit only), `test-integration` (MinIO only) |

#### Integration test levels

| Level | Class | Tests | What it proves |
|-------|-------|-------|---------------|
| 1 | `TestMinioWrite` | 2 | DuckDB httpfs writes/reads CSVs to MinIO |
| 2 | `TestMinioJarWrite` | 3 | Real Josh JAR exports to MinIO, Python reads back |
| 3 | `TestMinioCellDataLoader` | 4 | `CellDataLoader.load_csv("s3://...")` ingests into registry |
| 4 | `TestMinioIngestResults` | 2 | Full `ingest_results()` by label, data queryable |
| 5 | `TestMinioPartialRecovery` | 3 | Missing replicates skipped gracefully (2/3, 0/3, 1/10) |
| Edge | `TestMinioEdgeCases` | 3 | Bad creds, missing bucket, run_hash namespace isolation |

---

### PR 2: Target Profile System

New file `joshpy/targets.py`. joshpy reads AND writes `~/.josh/targets/<name>.json` — shared config between josh and joshpy.

**Dataclasses:** `TargetProfile`, `HttpTargetConfig`, `KubernetesTargetConfig` (mirrors joshsim JSON structure).

**JSON serialization:** Python snake_case <-> JSON camelCase where joshsim expects it (`api_key` -> `apiKey`, `timeout_seconds` -> `timeoutSeconds`).

**Functions:**
- `load_target(name)` / `save_target(name, profile)` — read/write `~/.josh/targets/<name>.json`
- `list_targets()` / `delete_target(name)` — manage profiles
- `resolve_minio_creds(target=None)` — hierarchy: profile JSON -> env vars

**K8s-specific fields** (from validated access pattern):
- `pod_minio_endpoint` — in-cluster MinIO endpoint pods use, distinct from outer `minio_endpoint`
- `cluster_context` — kubectl context name (Fabric8 reads `~/.kube/config`)
- `namespace` — K8s namespace for jobs
- `image` — container image for simulation pods
- `resource_requests` — CPU/memory for job pods
- `gcs_bucket` — GCS bucket for results

**Tests:** `tests/test_targets.py` — round-trip, hierarchy, validation, auto-create dirs.

**Risk: LOW — all new files, no modifications to existing code.**

---

### PR 3: CLI Wrappers — `batch_remote()`, `preprocess_batch()`, `stage_to_minio()`

`stage_from_minio()` already shipped in PR 1. This adds the remaining commands validated against GKE.

**`BatchRemoteConfig`:**
```python
@dataclass(frozen=True)
class BatchRemoteConfig:
    script_or_dir: Path    # .josh file or directory
    simulation: str
    target: str            # required — profile name
    replicates: int = 1
    no_wait: bool = False
    poll_interval: int | None = None
    timeout: int | None = None
    custom_tags: dict[str, str] = field(default_factory=dict)
```

**`PreprocessBatchConfig`:**
```python
@dataclass(frozen=True)
class PreprocessBatchConfig:
    script: Path           # .josh file
    simulation: str
    data_file: Path        # input .nc file
    variable: str
    units: str
    output: Path           # output .jshd file
    target: str            # required — profile name
```

**`StageToMinioConfig`:** `input_dir`, `prefix`, optional `minio_*` creds.

**Methods:** `JoshCLI.batch_remote()`, `JoshCLI.preprocess_batch()`, `JoshCLI.stage_to_minio()`.

**Note:** For K8s targets, the Java CLI is required — it uses Fabric8 to create K8s Jobs and Secrets directly. There is no HTTP intermediary. joshpy shells out to the JAR.

**Tests:** `tests/test_cli.py` — mock subprocess, verify arg building for all three commands.

**Risk: LOW — additive, follows existing `run_remote()` / `stage_from_minio()` patterns exactly.**

---

### PR 4: Sweep Integration — `run_sweep()` + `SweepManager` + adaptive

Wires batch remote into the sweep loop. Two modes:

**Blocking mode (default, `batch_no_wait=False`):** Each job calls `batchRemote` without `--no-wait`. The subprocess blocks until josh finishes polling internally. joshpy gets exit code, records in registry, then calls `ingest_results()` to read CSVs from S3. Sequential but simple — same pattern as existing `run_remote()`.

**Async mode (`batch_no_wait=True`):** Each job calls `batchRemote --no-wait`, gets back a job ID. joshpy stores `batch_job_id` in `job_runs.metadata`, then dispatches the next job. After all jobs are dispatched, joshpy polls via `cli.poll_batch(job_id, target)` ([josh#406](https://github.com/SchmidtDSE/josh/issues/406)) until all complete. Then ingests results. This is the path to parallel runs on big-memory machines.

**New CLI wrapper (depends on [josh#406](https://github.com/SchmidtDSE/josh/issues/406)):**
```python
@dataclass(frozen=True)
class PollBatchConfig:
    job_id: str
    target: str

def poll_batch(self, config: PollBatchConfig, timeout: float | None = None) -> CLIResult:
    # calls: java -jar joshsim.jar pollBatch <jobId> --target=<name>
    # exit code: 0=complete, 1=error, 2=running
```

**New parameters on `run_sweep()`:** `batch_remote`, `target`, `poll_interval`, `batch_timeout`, `batch_no_wait`, `auto_pull`.

**New functions in `joshpy/jobs.py`:**
- `assemble_batch_workdir(job, workdir)` — creates per-job dir with symlinked shared files + written .jshc
- `to_batch_remote_config(job, target, workdir)` — converts `ExpandedJob` to `BatchRemoteConfig`

**Validation:** `batch_remote` and `remote` mutually exclusive; `target` required when `batch_remote=True`.

**Extends:** `SweepManager.run()`, `run_adaptive_sweep()` with same parameters.

**Tests:** `tests/test_jobs.py`, `tests/test_sweep.py`, `tests/test_strategies.py`.

**Risk: LOW — mostly new code. Small modifications to existing function signatures (additive parameters). Async mode depends on josh#406.**

---

### PR 5: Shared Staging Optimization for Sweeps

Stage shared data (.josh, .jshd) to MinIO ONCE, per-job stage only the unique .jshc config. Avoids re-uploading GBs of .jshd per job.

New file `joshpy/batch_orchestrator.py` with `BatchOrchestrator`:
- `stage_shared(jobs)` — stage shared files once
- `dispatch_job(job, shared_prefix)` — stage per-job config + dispatch via HTTP POST to `/runBatch`
- `poll(job_id)` / `pull_results(job_id, output_dir)`

**Dispatch approach:** HTTP POST to `/runBatch` directly from Python (~20 lines with `urllib.request`). Self-contained, no joshsim changes needed, `/runBatch` endpoint already exists. This path is for HTTP targets only — K8s targets still require the JAR.

**Risk: MEDIUM — introduces direct HTTP dispatch from joshpy. Well-isolated in new file.**

---

### PR 6: Polish — Builder, Docs, Bottle Metadata

- `SweepManagerBuilder.with_batch_remote(target, ...)` convenience method
- MinIO metadata in bottle manifest
- Update `llms-full.txt` with all new APIs

**Risk: LOW**

---

## Files Modified (all PRs)

| File | PRs | Changes |
|------|-----|---------|
| `joshpy/cli.py` | ✅1, 3, 4 | `StageFromMinioConfig` + `stage_from_minio()` (✅PR 1); `BatchRemoteConfig` + `batch_remote()`, `PreprocessBatchConfig` + `preprocess_batch()`, `StageToMinioConfig` + `stage_to_minio()` (PR 3); `PollBatchConfig` + `poll_batch()` (PR 4) |
| `joshpy/cell_data.py` | ✅1 | `load_csv()` accepts `str` (S3 URL) in addition to `Path` |
| `joshpy/registry.py` | ✅1 | `configure_s3()` utility for DuckDB httpfs + S3 credential setup |
| `joshpy/sweep.py` | ✅1, 4, 6 | `ingest_results()` + helpers + `SweepManager.ingest()` (✅PR 1); extend `.run()` (PR 4); builder (PR 6) |
| **NEW** `joshpy/targets.py` | 2 | Target profile system (read/write/list/creds hierarchy) |
| `joshpy/jobs.py` | 4 | `assemble_batch_workdir`, `to_batch_remote_config`, extend `run_sweep()` |
| `joshpy/strategies.py` | 4 | Extend `run_adaptive_sweep()` |
| **NEW** `joshpy/batch_orchestrator.py` | 5 | Shared staging orchestration |
| `joshpy/bottle.py` | 6 | MinIO metadata in manifest |
| `joshpy/__init__.py` | 1-3 | Export new symbols |
| `tests/test_cli.py` | ✅1, 3 | `StageFromMinio` tests (✅PR 1); remaining CLI tests (PR 3) |
| `tests/test_sweep.py` | ✅1, 4 | `ingest_results` tests (✅PR 1); SweepManager batch_remote tests (PR 4) |
| `tests/conftest.py` | ✅1 | Shared fixtures, marker registration |
| `tests/test_minio_integration.py` | ✅1 | 17 MinIO integration tests (5 levels + edge cases) |
| **NEW** `tests/test_targets.py` | 2 | Target profile tests |
| `tests/test_jobs.py` | 4 | Workdir, converter, sweep tests |
| `tests/test_strategies.py` | 4 | Adaptive batch remote tests |

---

## Verification

PR 1 end-to-end (immediate need — ✅ validated):
```bash
# In josh-models repo:
pixi run recover my-label
# -> Looks up "my-label" in registry
# -> Discovers minio:// export paths via inspect-exports
# -> Configures DuckDB httpfs with S3 creds from env vars
# -> Reads CSVs directly from S3 into DuckDB (no download)
# -> Loads into registry, skipping missing replicate from OOM
# -> Prints: "Done: 1234567 rows loaded for 'my-label'"
```

CI verification (✅ in place):
```bash
# Unit tests (no MinIO needed):
pixi run test              # 867 passed, 17 deselected

# Integration tests (MinIO service container + JAR):
pixi run test-integration  # 17 tests across 5 levels + edge cases

# Local integration test:
docker run -d --name minio-test -p 9000:9000 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  -e MINIO_DEFAULT_BUCKETS=josh-test-bucket:public \
  bitnamilegacy/minio:latest
pixi run get-jars
pixi run -e dev test-integration
docker rm -f minio-test
```

Full batch remote integration (target: PR 4):
```python
# Sweep with batch remote on GKE
manager = SweepManager.from_config(config, registry="exp.duckdb")
results = manager.run(batch_remote=True, target="gke-test")
manager.load_results()

# Fire-and-forget -> recover later
results = manager.run(batch_remote=True, target="gke-test", batch_no_wait=True)
# ... later ...
manager.ingest()

# Or download locally
manager.ingest(download=True, output_dir=Path("./local_results"))
```
