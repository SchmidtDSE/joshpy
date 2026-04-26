# Plan: Batch Remote Execution for joshpy

Tracking issue: [joshpy#31](https://github.com/SchmidtDSE/joshpy/issues/31)
Umbrella PR on joshpy: [joshpy#34](https://github.com/SchmidtDSE/joshpy/pull/34) — `feat/batch-run` accumulates PRs 1–4 shipped; PR5 will land on the same branch.
Companion Java umbrella: [josh#374](https://github.com/SchmidtDSE/josh/issues/374)

**Upstream josh dependencies (all merged):**
- [josh#409](https://github.com/SchmidtDSE/josh/pull/409) — `pollBatch` CLI (closed [josh#406](https://github.com/SchmidtDSE/josh/issues/406))
- [josh#414](https://github.com/SchmidtDSE/josh/pull/414) — K8s batch execution system (PRs 5–9 + GKE integration)
- [josh#423](https://github.com/SchmidtDSE/josh/pull/423) — `batchRemote` flag-based UX with `.josh-staged.json` sentinel (**breaking CLI change — joshpy PR5 must adapt**)

**Filed joshpy-facing issues (tracking):**
- [josh#416](https://github.com/SchmidtDSE/josh/issues/416) — target profile JSON uses mixed snake_case/camelCase; joshpy works around it with an explicit coercion layer
- [josh#418](https://github.com/SchmidtDSE/josh/issues/418) — Cloud Run dev deployment stuck at `running` (container scaled down before sim completes); blocks HTTP-target e2e
- [josh#425](https://github.com/SchmidtDSE/josh/issues/425) — `.jshd` → `.jsdz` XZ/LZMA2 compression (pressure-release for per-job upload/download duplication; 5–20× expected)
- [josh#426](https://github.com/SchmidtDSE/josh/issues/426) — streaming `.jshd` from S3 + inline `.josh`/`.jshc` dispatch (deferred pending #425 evaluation)

**Related joshpy PR (separate from feat/batch-run):**
- [joshpy#35](https://github.com/SchmidtDSE/joshpy/pull/35) — `configure_s3` strips scheme from full URL endpoints (found during GKE e2e)

## Context

joshsim (Java) has added `batchRemote` — a parallel execution path using MinIO staging and target profiles instead of HTTP streaming. The full K8s batch execution system has merged to josh ([josh#414](https://github.com/SchmidtDSE/josh/pull/414)) and the CLI UX was reshaped in [josh#423](https://github.com/SchmidtDSE/josh/pull/423) to separate staging from dispatch via the `.josh-staged.json` sentinel. joshpy needs to wrap these capabilities and provide Python-level orchestration for parameter sweeps.

**Immediate motivation:** A production run has 5 of 6 replicate CSVs sitting in MinIO (the 6th OOM'd). The run is registered in the local RunRegistry with a label. We need a way to recover those results NOW — look up the run by label, discover the `minio://` export paths, read the CSVs directly into DuckDB via S3, and load them into the registry. This drives the PR ordering: result ingestion first, then the rest of the batch infrastructure.

**Access model (Model A):** MinIO/S3 CSVs are the source of truth. The local `.duckdb` is a materialized cache that any machine can rebuild by re-ingesting from S3. DuckDB reads CSVs directly from S3 via `httpfs` — no download, no local disk needed for the CSV data. This supports future access patterns: browser WASM reading S3, serverless aggregators attaching `.duckdb`, multi-machine access.

**State ownership:** josh is stateless/ephemeral — it dispatches jobs and can check their status, but holds no long-running state. joshpy owns all state via RunRegistry (what was run, parameters, label, job ID). When joshpy dispatches a `--no-wait` batch job, it stores the `batch_job_id` in `job_runs.metadata`. To poll, joshpy calls `cli.poll_batch(job_id, target)` (josh#409, shipped) which knows HOW to check status for each target type (MinIO status file for HTTP, K8s Job API for K8s). joshpy doesn't know or care about the polling mechanism internals — it just gets back exit codes (0 complete / 1 error / 2 running / 100 transient) + a JSON status line.

**Key design decisions:**
- **Staging is a separate concern from dispatch (post-#423).** `batchRemote` no longer auto-stages. It takes `--minio-prefix` pointing at an already-populated MinIO location, guarded by a `.josh-staged.json` sentinel. Callers populate the prefix via `stageToMinio` (explicit) or `--stage-from-local-dir` (convenience wrapper that calls `stageToMinio` before dispatching). For sweeps, joshpy uses `cli.stage_to_minio()` then `cli.batch_remote(require_prestaged=True)`.
- **Target config is SHARED between josh and joshpy.** joshpy reads AND writes `~/.josh/targets/<name>.json`.
- **MinIO cred resolution hierarchy** (mirrors joshsim's `HierarchyConfig`): CLI flags > profile JSON > env vars (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`). Secrets don't need to live in profile JSON.
- **K8s targets have a separate `pod_minio_endpoint`** — the in-cluster MinIO endpoint pods use, which may differ from the outer `minio_endpoint` used for host-side staging.
- **Auto-pull results from MinIO after completion**, with opt-out for fire-and-forget. The same `ingest_results()` code path serves batch remote AND local OOM recovery (DRY).
- **Per-job duplication is accepted, not optimized** (decided via [josh#425](https://github.com/SchmidtDSE/josh/issues/425) / [josh#426](https://github.com/SchmidtDSE/josh/issues/426)). Each `ExpandedJob` stages its own copy of `.josh` + `.jshd` + `.jshc` to its own MinIO prefix. Sharing shared files across sweep jobs would require a PVC or a josh-side multi-prefix-merge feature; neither is available today. Compression (josh#425, expected 5–20× on geospatial `.jshd`) is the accepted pressure-release.
- **Pods do NOT share disk.** K8s indexed Jobs fan out N pods per dispatch; each pod has its own container FS and runs `stageFromMinio` independently. The "shared" in "20 replicates share external data" means shared MinIO prefix (one upload, N parallel downloads), not shared disk.

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

### Commands (pre-#423, historical)

```bash
# Validated against pre-#423 JAR with positional <input> arg.  Retained
# here only to explain what was tested end-to-end; the CLI surface has
# since changed.  See the "Authoritative CLI Surface" section below for
# the current syntax joshpy must target.
java -jar joshsim-fat.jar batchRemote sim.josh SimName --target=gke-test --replicates=5
```

### Implications for joshpy

- **K8s targets require the Java CLI** — it uses the Fabric8 K8s client to create Jobs and Secrets directly. There is no HTTP intermediary. joshpy shells out to `java -jar joshsim-fat.jar batchRemote ...`.
- **HTTP targets also go through the JAR.** The JAR is the single source of truth for dispatch semantics across both target types. joshpy does not POST to `/runBatch` directly — the ~3s JVM startup cost per job is negligible vs job runtimes and is easily amortized in sweeps. (Rejected: the "Python-side direct HTTP dispatch" option from earlier drafts.)
- **Env vars are the credential transport** — set `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` before invoking the JAR. The JAR resolves them via HierarchyConfig.
- **GKE cluster is already running** — server-side infrastructure is deployed. joshpy just needs the client-side plumbing.

---

## Authoritative CLI Surface (post-#423)

The following is verbatim from `joshsim-fat-dev.jar` (latest `dev` build, 2026-04-24) and is the source of truth joshpy code must match.  The josh-side `llms-full.txt` documents this at a higher level; the JAR help is canonical.

### `batchRemote`

```
joshsim batchRemote [--no-wait] [--require-prestaged] [--suppress-errors] [--suppress-info]
                    --minio-prefix=<minioPrefix>
                    [--poll-interval=<pollIntervalSeconds>]
                    [--replicates=<replicates>]
                    [--stage-from-local-dir=<stageFromLocalDir>]
                    --target=<targetName>
                    [--timeout=<timeoutSeconds>]
                    <simulation>
```

Required:
- Positional `<simulation>` — simulation name (e.g. `Main`)
- `--minio-prefix=<prefix>` — where inputs live (e.g. `batch-jobs/my-run/inputs/`)
- `--target=<name>` — profile from `~/.josh/targets/<name>.json`

Three modes (mutually exclusive):
- `--stage-from-local-dir=<dir>` — upload local dir to `--minio-prefix`, write `.josh-staged.json` sentinel, then dispatch
- default (neither flag) — read sentinel at `--minio-prefix`: warn if absent, fail if `staging`/`error`, proceed if `complete`
- `--require-prestaged` — fail fast unless sentinel reports `complete` (recommended for the sweep use case)

Removed vs pre-#423: positional `<input>` (was the local dir or `.josh` file), `--custom-tag` flag.

### `pollBatch`

```
joshsim pollBatch [--suppress-errors] [--suppress-info] --target=<targetName> <jobId>
```

Exit codes:
- `0` — complete (success)
- `1` — error (simulation failed or dispatcher reported terminal failure)
- `2` — running / pending (still in progress)
- `100` — poll failure (transient; caller should retry)

Stdout JSON (one line):
```json
{"status": "running",  "jobId": "...", "startedAt":   "<iso8601>"}
{"status": "complete", "jobId": "...", "completedAt": "<iso8601>"}
{"status": "error",    "jobId": "...", "failedAt":    "<iso8601>", "message": "<reason>"}
```

### `stageToMinio`

```
joshsim stageToMinio [--ensure-bucket-exists] [--suppress-errors] [--suppress-info]
                     [--config-file=<configFile>]
                     --input-dir=<inputDir> --prefix=<prefix>
                     [--minio-endpoint=...] [--minio-access-key=...]
                     [--minio-secret-key=...] [--minio-bucket=...]
                     [--minio-path=...]
```

Behavior: walks `<input-dir>` recursively, uploads every regular file to `<prefix>` + relative path.  Does **not** delete existing keys at the prefix (overlay/additive).  Always writes `.josh-staged.json` at `<prefix>`: `status=staging` → `status=complete` on success, or `status=error` with `message=<exception>` on failure.

### `stageFromMinio`

```
joshsim stageFromMinio [--ensure-bucket-exists] [--suppress-errors] [--suppress-info]
                       [--config-file=<configFile>]
                       --output-dir=<outputDir> --prefix=<prefix>
                       [--minio-endpoint=...] [--minio-access-key=...]
                       [--minio-secret-key=...] [--minio-bucket=...]
                       [--minio-path=...]
```

Used both pod-side (in the batch worker entrypoint) and client-side (fallback path for `ingest_results(download=True)`).  Filters out `.josh-staged.json` at any depth so pods never see the sentinel; throws if every key was filtered out (prevents silently empty workdirs).

### `preprocessBatch` (unchanged schema for now)

```
joshsim preprocessBatch <input> <simulation> <dataFile> <variable> <units> <outputFile>
                        --target=<targetName>
                        [--no-wait] [--poll-interval=S] [--timeout=S]
                        [--crs=<crs>] [--default-value=<v>]
                        [--x-coord=<name>] [--y-coord=<name>]
                        [--time-dim=<name>] [--timestep=<int>]
                        [--parallel] [--amend]
```

Note: `preprocessBatch` has **not** been refactored to the flag-based `--minio-prefix` UX yet — it still takes a positional `<input>` that gets auto-staged by the JAR.  If josh reshapes this in a follow-up, joshpy's `PreprocessBatchConfig` will need the same refactor as `BatchRemoteConfig`.

### `.josh-staged.json` sentinel

Written by `stageToMinio` at the root of its `--prefix`.  JSON shape (from josh#423):

```json
{"status": "staging",  "startedAt":   "<iso8601>"}
{"status": "complete", "completedAt": "<iso8601>"}
{"status": "error",    "failedAt":    "<iso8601>", "message": "<exception>"}
```

Readers:
- `batchRemote` (default mode) — proceeds if `complete`, warns if absent, fails on `staging`/`error`
- `batchRemote --require-prestaged` — fails hard unless `complete`
- pod entrypoint — filtered out of `stageFromMinio` so pods never see it

---

## PR Plan

```
PR1 ✅ → PR2 ✅ → PR3 ✅ → PR4 ✅ → PR5 (CLI refactor for #423 + per-job workdir) → PR6 (polish)
```

PRs 1–4 are merged to `feat/batch-run` (joshpy#34).  PR3 shipped `BatchRemoteConfig` / `cli.batch_remote()` against the pre-#423 CLI; PR5 rewrites both against the new surface.  PR4 shipped `to_batch_remote_config()` which needs the same refactor.  No downstream consumers, so the break is contained.

### Regression gates (every PR)
- `pixi run test` passes (current: 935 unit tests; integration tests excluded)
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

### PR 2: Target Profile System ✅ DONE

**Status:** Merged on `feat/batch-run` (commit `a6f82d7`). No changes needed for #423 — the target profile JSON schema is unchanged.

#### What shipped
- `joshpy/targets.py` — `TargetProfile`, `HttpTargetConfig`, `KubernetesTargetConfig`, `ResolvedMinioCreds` dataclasses; `save_target`/`load_target`/`list_targets`/`delete_target`; `resolve_minio_creds()` (profile > env vars hierarchy)
- `joshpy/__init__.py` — all target symbols exported
- `tests/test_targets.py` — 32 tests (dataclass construction, serialization round-trip, filesystem CRUD, credential resolution)
- `.devcontainer/Dockerfile` + `install_gcloud.sh`/`install_kubectl.sh` — SHA256-pinned gcloud SDK 526.0.0 + kubectl v1.31.4 baked into the image
- Explicit Python↔JSON key mapping for 4 fields where conventions differ: `target_type` ↔ `type`, `api_key` ↔ `apiKey`, `timeout_seconds` ↔ `timeoutSeconds`, `ttl_seconds_after_finished` ↔ `ttlSecondsAfterFinished`

#### Known gap
- [josh#416](https://github.com/SchmidtDSE/josh/issues/416) — target profile JSON mixes snake_case and camelCase in the same file (`minio_endpoint` vs `apiKey`). joshpy handles it with the explicit coercion layer above. Filed for future consistency.

---

### PR 3: CLI Wrappers — `batch_remote()`, `preprocess_batch()`, `stage_to_minio()` ✅ DONE (needs refactor in PR5)

**Status:** Merged on `feat/batch-run` (commit `346b9b9`). Built against the **pre-#423** CLI.  **Must be refactored in PR5** to match the new flag-based `batchRemote` surface.

#### What shipped (pre-#423 shape — partially obsolete)

| Dataclass | Status | Notes |
|-----------|--------|-------|
| `StageToMinioConfig` | ✅ additive fields TBD in PR5 | Missing `ensure_bucket_exists`, `config_file`, `minio_path` (new in josh dev) |
| `StageFromMinioConfig` | ✅ (shipped in PR1) additive fields TBD in PR5 | Same new flags available |
| `BatchRemoteConfig` | ⚠️ **breaking refactor in PR5** | Currently has positional `script_or_dir` + `custom_tags`, both removed by #423 |
| `PreprocessBatchConfig` | ✅ barebones; expandable | Still uses positional args (josh hasn't refactored `preprocessBatch` yet) |

#### What PR5 must do to this module
- Rewrite `BatchRemoteConfig` to require `minio_prefix`, add `stage_from_local_dir` and `require_prestaged` (mutex), drop `script_or_dir` and `custom_tags`
- Rewrite `cli.batch_remote()` command building against the new flag set
- Add optional `ensure_bucket_exists`, `config_file`, `minio_path` to `StageToMinioConfig` / `StageFromMinioConfig`
- Expand `PreprocessBatchConfig` with newly-exposed flags (`--crs`, `--x-coord`, `--y-coord`, `--time-dim`, `--timestep`, `--default-value`, `--parallel`, `--amend`)

#### Tests
- `tests/test_cli.py` — `TestStageToMinio`, `TestStageToMinioConfig`, `TestBatchRemote`, `TestBatchRemoteConfig`, `TestPreprocessBatch`, `TestPreprocessBatchConfig`, `TestPollBatch`, `TestPollBatchConfig` (17 tests; will need updates in PR5 to match new CLI shape)

---

### PR 4: Sweep Integration — `run_sweep()` + `SweepManager` + adaptive ✅ DONE (needs refactor in PR5)

**Status:** Merged on `feat/batch-run` (commit `ff661a3`). Built against the **pre-#423** CLI. The dispatch wiring in `run_sweep()` and `run_adaptive_sweep()` is sound, but the `to_batch_remote_config()` helper and the exact `cli.batch_remote()` call sites must change in PR5.

#### What shipped
- `PollBatchConfig` + `JoshCLI.poll_batch()` — wraps `pollBatch <jobId> --target=<name>`. **Still correct** post-#423 (pollBatch CLI didn't change).
- `to_batch_remote_config(job, target, *, no_wait, poll_interval, timeout)` — converts `ExpandedJob` → `BatchRemoteConfig`. **Will be rewritten in PR5** against the new config shape.
- `run_sweep()` new params: `batch_remote`, `target`, `batch_no_wait`, `poll_interval`, `batch_timeout`, `auto_ingest`. Mutually-exclusive with `remote`.
- Two dispatch modes wired:
  - **Blocking** (default): `batch_remote(no_wait=False)`, JAR polls internally, `ingest_results()` loads CSVs from S3
  - **Async** (`batch_no_wait=True`): `batch_remote(no_wait=True)` per job, parse `jobId` from stdout JSON, store in registry metadata, then poll loop using `cli.poll_batch()` until all complete
- `SweepManager.run()` / `run_adaptive_sweep()` thread the new params through

#### What PR5 must do to this module
- Update the `to_batch_remote_config()` call sites in `jobs.py` and `strategies.py` to:
  1. Assemble per-job workdir (new helper)
  2. Call `cli.stage_to_minio(input_dir=workdir, prefix=per_job_prefix)`
  3. Build `BatchRemoteConfig(minio_prefix=per_job_prefix, require_prestaged=True, ...)` (new schema)
  4. Call `cli.batch_remote(br_config)`
- The async polling loop (`_async_dispatched` / `cli.poll_batch`) stays as-is.

#### Tests shipped
- `tests/test_jobs.py::TestToBatchRemoteConfig` (3 tests) — will be updated for new shape
- `tests/test_jobs.py::TestRunSweepBatchRemote` (4 tests) — covers validation, blocking dispatch, async JSON parsing; the dispatch-path tests will need reworking against `stage_to_minio` + `batch_remote(--require-prestaged)` flow

---

### PR 5: Refactor to post-#423 CLI + per-job workdir assembly

This is the consolidated "catch up to josh#423 and fix the e2e-surfaced staging gaps" PR.  Two real problems it solves:

1. **Adapt to josh#423's breaking CLI change.** `batchRemote` no longer auto-stages; it expects `--minio-prefix` pointing at a sentinel-protected MinIO location. joshpy's `BatchRemoteConfig` / `cli.batch_remote()` / `to_batch_remote_config()` must be rewritten.
2. **Directory contamination fix (found during GKE e2e).** When the pre-#423 JAR was given a `.josh` file, it staged the entire containing directory. Sibling `.josh` files (e.g. test fixtures) got swept in. Per-job workdir assembly solves this cleanly.

**Explicitly NOT in scope** (discussed and deferred per [josh#425](https://github.com/SchmidtDSE/josh/issues/425) / [josh#426](https://github.com/SchmidtDSE/josh/issues/426)):
- Cross-job sharing of `.josh`/`.jshd` via a shared MinIO prefix. josh does not support multi-prefix merge today; joshpy-side server-side copy would re-couple infra logic. Accepted duplication is the design; compression is the pressure-release.
- Python-side HTTP POST to `/runBatch`. JAR is the single dispatcher.

#### New file: `joshpy/batch_orchestrator.py`

```python
def assemble_batch_workdir(job: ExpandedJob, workdir: Path) -> Path:
    """Create a per-ExpandedJob staging directory.

    Layout::

        workdir/<run_hash>/
          sim.josh            # symlink to job.source_path
          config.jshc         # unique rendered config for this job
          <file_mapping>.jshd # symlinks for each entry in job.file_mappings

    Returns the path that should be passed to ``cli.stage_to_minio(input_dir=...)``.
    Uses symlinks (not copies) to avoid disk duplication for large .jshd files.
    """
```

(No `BatchOrchestrator` class. Pure function. joshpy doesn't own staging state — josh does, via the sentinel.)

#### CLI-layer refactor (`joshpy/cli.py`)

```python
@dataclass(frozen=True)
class BatchRemoteConfig:
    simulation: str
    target: str
    minio_prefix: str                       # REQUIRED (new)
    replicates: int = 1
    no_wait: bool = False
    poll_interval: int | None = None
    timeout: int | None = None
    stage_from_local_dir: Path | None = None  # mutex with require_prestaged
    require_prestaged: bool = False           # recommended for sweeps
    # removed: script_or_dir, custom_tags

    def __post_init__(self) -> None:
        if self.stage_from_local_dir and self.require_prestaged:
            raise ValueError(
                "stage_from_local_dir and require_prestaged are mutually exclusive"
            )
```

`StageToMinioConfig` / `StageFromMinioConfig`: add optional `ensure_bucket_exists: bool = False`, `config_file: Path | None = None`, `minio_path: str | None = None`.

`PreprocessBatchConfig`: expand with `crs`, `x_coord`, `y_coord`, `time_dim`, `timestep`, `default_value`, `parallel`, `amend`.

#### Sweep-loop rewire (`joshpy/jobs.py` + `joshpy/strategies.py`)

```python
# New shape of to_batch_remote_config: takes a pre-staged prefix.
def to_batch_remote_config(
    job: ExpandedJob,
    target: str,
    minio_prefix: str,
    *,
    no_wait: bool = False,
    poll_interval: int | None = None,
    timeout: int | None = None,
    require_prestaged: bool = True,
) -> BatchRemoteConfig: ...

# run_sweep batch-remote path becomes:
workdir = tempfile.mkdtemp(prefix=f"joshpy-sweep-{session_id}-")
for job in job_set:
    job_dir = assemble_batch_workdir(job, Path(workdir))
    per_job_prefix = f"sweeps/{session_id}/jobs/{job.run_hash}/"
    cli.stage_to_minio(
        StageToMinioConfig(input_dir=job_dir, prefix=per_job_prefix)
    )
    cli.batch_remote(
        to_batch_remote_config(job, target, per_job_prefix,
                               require_prestaged=True,
                               no_wait=batch_no_wait,
                               poll_interval=poll_interval,
                               timeout=batch_timeout)
    )
```

The async `_async_dispatched` / `cli.poll_batch` loop from PR4 stays exactly as-is.

#### Tests
- `tests/test_cli.py` — update `TestBatchRemote*` for the new flag set; add `TestBatchRemoteConfig.test_mutex` for the mutex validation; add tests for new optional flags on stage configs
- `tests/test_batch_orchestrator.py` (NEW) — `assemble_batch_workdir` covers symlinks vs copies, run_hash subdir naming, file_mappings fan-out, `.jshc` content write
- `tests/test_jobs.py::TestToBatchRemoteConfig` — rewrite for new signature
- `tests/test_jobs.py::TestRunSweepBatchRemote` — mock both `stage_to_minio` and `batch_remote`; assert ordering (stage first, dispatch second); assert `require_prestaged=True` on the batch call

#### E2E against GKE (2026-04-23+ JAR required)
- Dispatch a single ExpandedJob with 5 replicates through the full joshpy sweep loop
- Verify GCS prefix contains `.josh-staged.json` with `status=complete` after `stage_to_minio`
- Verify `batch_remote(--require-prestaged)` proceeds and K8s indexed Job fans out 5 pods
- Verify `ingest_results()` pulls all 5 CSVs from the per-job prefix's output location

**Risk: LOW.** CLI refactor is mechanical. Workdir assembly is pure filesystem. No infra/dispatch logic moves into joshpy.  HTTP-target e2e is blocked on [josh#418](https://github.com/SchmidtDSE/josh/issues/418) but K8s path is unaffected.

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
| `joshpy/cli.py` | ✅1, ✅3, ✅4, **5** | `StageFromMinioConfig` + `stage_from_minio()` (✅PR 1); `BatchRemoteConfig` + `batch_remote()`, `PreprocessBatchConfig` + `preprocess_batch()`, `StageToMinioConfig` + `stage_to_minio()` (✅PR 3); `PollBatchConfig` + `poll_batch()` (✅PR 4); **refactor `BatchRemoteConfig`/`cli.batch_remote()` for josh#423 new flag surface; add `ensure_bucket_exists`/`config_file`/`minio_path` to stage configs; expand `PreprocessBatchConfig` (PR 5)** |
| `joshpy/cell_data.py` | ✅1 | `load_csv()` accepts `str` (S3 URL) in addition to `Path` |
| `joshpy/registry.py` | ✅1 | `configure_s3()` utility for DuckDB httpfs + S3 credential setup (separate PR [joshpy#35](https://github.com/SchmidtDSE/joshpy/pull/35) adds scheme-stripping on top) |
| `joshpy/sweep.py` | ✅1, ✅4, 6 | `ingest_results()` + helpers + `SweepManager.ingest()` (✅PR 1); `batch_remote`/`target`/`batch_no_wait`/`poll_interval`/`batch_timeout`/`auto_ingest` on `.run()` (✅PR 4); builder (PR 6) |
| **NEW** `joshpy/targets.py` | ✅2 | Target profile system (read/write/list/creds hierarchy) |
| `joshpy/jobs.py` | ✅4, **5** | `to_batch_remote_config()` + extend `run_sweep()` (✅PR 4); **rewrite `to_batch_remote_config()` for new CLI; wire `assemble_batch_workdir` + `stage_to_minio` before dispatch (PR 5)** |
| `joshpy/strategies.py` | ✅4, **5** | Extend `run_adaptive_sweep()` (✅PR 4); same dispatch-path rewrite (PR 5) |
| **NEW** `joshpy/batch_orchestrator.py` | **5** | `assemble_batch_workdir()` pure-function helper |
| `joshpy/bottle.py` | 6 | MinIO metadata in manifest |
| `joshpy/__init__.py` | ✅1-4, **5** | Export new symbols; remove/add as CLI shape changes (PR 5) |
| **NEW** `.devcontainer/scripts/on_build/install_gcloud.sh` + `install_kubectl.sh` | ✅2 | SHA256-pinned system tool installs |
| `.devcontainer/Dockerfile` | ✅2 | Install curl + run both gcloud/kubectl scripts |
| `tests/test_cli.py` | ✅1, ✅3, **5** | `StageFromMinio` tests (✅PR 1); remaining CLI tests (✅PR 3); **update `TestBatchRemote*` for new flags, add mutex test (PR 5)** |
| `tests/test_sweep.py` | ✅1, ✅4 | `ingest_results` tests (✅PR 1); SweepManager batch_remote tests (✅PR 4); `TestConfigureS3` scheme-handling tests (joshpy#35) |
| `tests/conftest.py` | ✅1 | Shared fixtures, marker registration |
| `tests/test_minio_integration.py` | ✅1 | 17 MinIO integration tests (5 levels + edge cases) |
| **NEW** `tests/test_targets.py` | ✅2 | Target profile tests (32 cases) |
| `tests/test_jobs.py` | ✅4, **5** | Workdir + converter + sweep tests (✅PR 4); rewrite for new `to_batch_remote_config` shape and `stage_to_minio → batch_remote` ordering (PR 5) |
| `tests/test_strategies.py` | 4, **5** | Adaptive batch remote tests; adapt for new dispatch path (PR 5) |
| **NEW** `tests/test_batch_orchestrator.py` | **5** | `assemble_batch_workdir` tests (symlinks, run_hash dirs, .jshc content) |

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
pixi run test              # 935 passed, 17 deselected (as of 2026-04-23)

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

Full batch remote integration (target: PR 5 once refactor lands):
```python
# Sweep with batch remote on GKE.  joshpy stages each ExpandedJob's
# workdir to its own MinIO prefix, then dispatches with
# require_prestaged=True.
manager = SweepManager.from_config(config, registry="exp.duckdb")
results = manager.run(batch_remote=True, target="gke-test")
manager.load_results()

# Fire-and-forget -> recover later (stores batch_job_id in registry metadata).
results = manager.run(batch_remote=True, target="gke-test", batch_no_wait=True)
# ... later, same machine or another:
manager.ingest()

# Or download locally:
manager.ingest(download=True, output_dir=Path("./local_results"))
```

E2E status (2026-04-24):
- **K8s target (gke-test):** ✅ fully working end-to-end against `dse-nps` GKE Autopilot with the pre-#423 JAR. Dispatch → poll → ingest all validated. Re-validation needed once PR5 lands and we switch to the post-#423 JAR (`jar/joshsim-fat-dev.jar`).
- **HTTP target (cloudrun-dev):** ❌ blocked on [josh#418](https://github.com/SchmidtDSE/josh/issues/418). Dispatch succeeds, `status=running` is written to GCS, then Cloud Run scales the container down before the simulation completes. No output CSV produced. Not a joshpy issue.

---

## Summary: why PR5 is the main remaining work

josh#423 arrived after PRs 3–4 were designed. The new flag-based `batchRemote` is a strictly better architecture — staging and dispatch are cleanly separated, the `.josh-staged.json` sentinel gives us an explicit readiness contract, and the multi-dispatch workflow (one stage, many dispatches) is now first-class. But it's a breaking CLI change. joshpy PR5 is the "catch up + fix the workdir-contamination bug + wire `stage_to_minio` + `batch_remote(--require-prestaged)` through the sweep loop" consolidation. After PR5, the plan completes with PR6 (polish).
