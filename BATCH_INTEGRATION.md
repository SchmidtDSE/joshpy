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
- Dev JARs are outdated — implement against spec, test when updated.

---

## PR Plan

```
PR1 (S3-native ingest) → PR2 (target profiles) → PR3 (CLI wrappers) → PR4 (sweep integration) → PR5 (shared staging optimization) → PR6 (polish)
```

### Regression gates (every PR)
- `pixi run pytest` passes
- Existing `runRemote` path completely untouched

---

### PR 1: Result Recovery — S3-native `ingest_results()`

**Solves the immediate need.** Enables recovering results from MinIO into the registry by label. DuckDB reads CSVs directly from S3 via httpfs — no download, no local disk needed for the CSV data. Also provides `download=True` fallback via `stageFromMinio` for users who want local copies.

#### New utility: `configure_s3()` in `joshpy/registry.py` (or `joshpy/s3.py`)

Reusable DuckDB S3/MinIO connection setup — the foundation for all future S3 access (serverless aggregators, WASM, multi-machine):

```python
def configure_s3(conn, endpoint: str, access_key: str, secret_key: str, url_style: str = "path") -> None:
    """Configure DuckDB connection for S3/MinIO access via httpfs."""
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"""
        CREATE OR REPLACE SECRET (
            TYPE s3,
            KEY_ID '{access_key}',
            SECRET '{secret_key}',
            ENDPOINT '{endpoint}',
            URL_STYLE '{url_style}',
            USE_SSL true
        )
    """)
```

S3 credentials resolve via hierarchy: explicit args > env vars (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`). The function takes explicit args; the caller (`ingest_results`) handles the env var fallback.

#### Modify `CellDataLoader.load_csv()` in `joshpy/cell_data.py`

Accept `str` (S3 URL) in addition to `Path`:
```python
def load_csv(self, csv_path: Path | str, run_id: str, run_hash: str, ...) -> int:
    if isinstance(csv_path, str) and csv_path.startswith("s3://"):
        csv_path_str = csv_path  # S3 URL — pass directly to read_csv_auto
    else:
        csv_path_str = str(Path(csv_path).resolve())  # local path (existing behavior)
    # ... rest unchanged — read_csv_auto handles both
```

The existing `read_csv_auto()` call works with S3 URLs natively once httpfs is loaded.

#### New function: `ingest_results()` in `joshpy/sweep.py`

The core recovery function. Works by label (not by `JobSet`):

```python
def ingest_results(
    cli: JoshCLI,
    registry: RunRegistry,
    label_or_hash: str,
    *,
    export_type: str = "patch",
    download: bool = False,           # if True, download via stageFromMinio instead of S3 read
    output_dir: Path | None = None,   # download destination (only used when download=True)
    minio_bucket: str | None = None,  # override bucket (else from ExportFileInfo.host)
    quiet: bool = False,
) -> int:
```

**Flow:**
1. `registry._resolve_label_or_hash(label_or_hash)` -> `run_hash`
2. `registry.get_config_by_hash(run_hash)` -> `ConfigInfo` (josh_path, josh_content, parameters, label)
3. `registry.get_session(config.session_id)` -> `SessionInfo` (simulation, total_replicates)
4. Get josh source on disk: if `config.josh_path` exists use it; otherwise write `config.josh_content` to a temp file
5. `cli.inspect_exports(script, simulation)` -> `ExportPaths`
6. Get `ExportFileInfo` for `export_type` -> check `info.protocol`
7. If `protocol == "minio"` and NOT `download`:
   - Configure S3 on registry connection: `configure_s3(registry.conn, endpoint, access_key, secret_key)`
   - Translate `minio://bucket/path` to `s3://bucket/path` for DuckDB
8. If `protocol == "minio"` and `download`:
   - Call `cli.stage_from_minio(...)` to download locally (fallback path)
9. `registry._resolve_run_id_for_hash(run_hash)` -> `run_id` for the latest execution
10. For each replicate 0..`total_replicates-1`:
    - Build template vars: `{simulation, replicate, **config.parameters, label: config.label}`
    - Resolve path template -> concrete path
    - If minio (no download): remap to `s3://bucket/resolved_path`
    - If minio (download): remap to `output_dir / filename`
    - Load via `CellDataLoader.load_csv(csv_path_or_url, run_id, run_hash)`
    - If file/object doesn't exist -> skip gracefully (the OOM'd replicate), print which one
11. Return total rows loaded

#### Also in this PR: `StageFromMinioConfig` + `stage_from_minio()` for `download=True` path

```python
@dataclass(frozen=True)
class StageFromMinioConfig:
    output_dir: Path
    prefix: str
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None
```

Plus `JoshCLI.stage_from_minio()` method wrapping `stageFromMinio --output-dir=... --prefix=... [--minio-* options]`.

#### New method: `SweepManager.ingest()` in `joshpy/sweep.py`

```python
def ingest(self, export_type="patch", download=False, output_dir=None, quiet=False) -> int:
    label = getattr(self, '_label', None) or self.job_set.jobs[0].run_hash
    return ingest_results(self.cli, self.registry, label, export_type=export_type,
                          download=download, output_dir=output_dir, quiet=quiet)
```

#### Exports: `joshpy/__init__.py`
- Add `StageFromMinioConfig`, `configure_s3` to CLI exports
- Add `ingest_results` to sweep exports

#### Tests
- `tests/test_cli.py`: `StageFromMinioConfig` defaults, `stage_from_minio()` arg building
- `tests/test_sweep.py`: `ingest_results` with mocked registry + mocked DuckDB (S3 URL construction, download fallback, missing replicate skip, josh_content temp file fallback)

#### User-facing example (pixi task in josh-models)

```toml
recover = { cmd = "python scripts/recover.py", env = { JOSH_LABEL = "{{ LABEL }}" }, args = [{ arg = "LABEL" }], description = "Recover results from MinIO: pixi run recover <LABEL>." }
```

```python
# scripts/recover.py
import os
from dotenv import load_dotenv
load_dotenv()

from joshpy.cli import JoshCLI
from joshpy.jar import JarMode
from joshpy.registry import RunRegistry
from joshpy.sweep import ingest_results

registry = RunRegistry(os.environ["JOSH_REGISTRY"])
cli = JoshCLI(josh_jar=JarMode[os.environ.get("JOSH_JAR_MODE", "DEV")])

rows = ingest_results(cli, registry, os.environ["JOSH_LABEL"])
print(f"Done: {rows} rows loaded for '{os.environ['JOSH_LABEL']}'")
registry.close()
```

**Polling:** PR 1 does NOT poll — `ingest_results()` assumes the job is already done (called after blocking `batchRemote`, or manually by user for recovery). It reads whatever CSVs exist in S3 and skips missing ones. Async polling comes in PR 4/5 via josh's `pollBatch` CLI ([josh#406](https://github.com/SchmidtDSE/josh/issues/406)).

**`batch_job_id` in registry:** When batch remote jobs are dispatched, the job ID is stored in `job_runs.metadata` as `{"batch_job_id": "...", "target": "..."}`. This field is optional — absent for local runs and blocking batch runs where the ID isn't needed. `ingest_results()` does not require it; it works by label/run_hash alone.

**Risk: LOW — additive. Existing load_csv() local path behavior unchanged. httpfs is opt-in (only configured when minio:// detected).**

---

### PR 2: Target Profile System

New file `joshpy/targets.py`. joshpy reads AND writes `~/.josh/targets/<name>.json` — shared config between josh and joshpy.

**Dataclasses:** `TargetProfile`, `HttpTargetConfig`, `KubernetesTargetConfig` (mirrors joshsim JSON structure).

**JSON serialization:** Python snake_case <-> JSON camelCase where joshsim expects it (`api_key` -> `apiKey`, `timeout_seconds` -> `timeoutSeconds`).

**Functions:**
- `load_target(name)` / `save_target(name, profile)` — read/write `~/.josh/targets/<name>.json`
- `list_targets()` / `delete_target(name)` — manage profiles
- `resolve_minio_creds(target=None)` — hierarchy: profile JSON -> env vars

**K8s note:** `pod_minio_endpoint` in `KubernetesTargetConfig` — the in-cluster MinIO endpoint pods use, distinct from outer `minio_endpoint`.

**Tests:** `tests/test_targets.py` — round-trip, hierarchy, validation, auto-create dirs.

**Risk: LOW — all new files, no modifications to existing code.**

---

### PR 3: Remaining CLI Wrappers — `batch_remote()`, `stage_to_minio()`

`stage_from_minio()` already shipped in PR 1. This adds the remaining two.

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
```

**`StageToMinioConfig`:** `input_dir`, `prefix`, optional `minio_*` creds.

**Methods:** `JoshCLI.batch_remote()`, `JoshCLI.stage_to_minio()`.

**Tests:** `tests/test_cli.py` — mock subprocess, verify arg building.

**Risk: LOW — additive, follows existing `run_remote()` pattern exactly.**

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

**Dispatch approach:** HTTP POST to `/runBatch` directly from Python (~20 lines with `urllib.request`). Self-contained, no joshsim changes needed, `/runBatch` endpoint already exists.

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
| `joshpy/cli.py` | 1, 3, 4 | `StageFromMinioConfig` + `stage_from_minio()` (PR 1); `BatchRemoteConfig` + `batch_remote()`, `StageToMinioConfig` + `stage_to_minio()` (PR 3); `PollBatchConfig` + `poll_batch()` (PR 4) |
| `joshpy/cell_data.py` | 1 | `load_csv()` accepts `str` (S3 URL) in addition to `Path` |
| `joshpy/registry.py` | 1 | `configure_s3()` utility for DuckDB httpfs + S3 credential setup |
| `joshpy/sweep.py` | 1, 4, 6 | `ingest_results()` + `SweepManager.ingest()` (PR 1); extend `.run()` (PR 4); builder (PR 6) |
| **NEW** `joshpy/targets.py` | 2 | Target profile system (read/write/list/creds hierarchy) |
| `joshpy/jobs.py` | 4 | `assemble_batch_workdir`, `to_batch_remote_config`, extend `run_sweep()` |
| `joshpy/strategies.py` | 4 | Extend `run_adaptive_sweep()` |
| **NEW** `joshpy/batch_orchestrator.py` | 5 | Shared staging orchestration |
| `joshpy/bottle.py` | 6 | MinIO metadata in manifest |
| `joshpy/__init__.py` | 1-3 | Export new symbols |
| `tests/test_cli.py` | 1, 3 | `StageFromMinio` tests (PR 1); remaining CLI tests (PR 3) |
| `tests/test_sweep.py` | 1, 4 | `ingest_results` tests (PR 1); SweepManager batch_remote tests (PR 4) |
| **NEW** `tests/test_targets.py` | 2 | Target profile tests |
| `tests/test_jobs.py` | 4 | Workdir, converter, sweep tests |
| `tests/test_strategies.py` | 4 | Adaptive batch remote tests |

---

## Verification

PR 1 end-to-end (immediate need):
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

Full integration (when dev JARs update):
```python
# Sweep with batch remote
manager = SweepManager.from_config(config, registry="exp.duckdb")
results = manager.run(batch_remote=True, target="my-server")
manager.load_results()

# Fire-and-forget -> recover later
results = manager.run(batch_remote=True, target="my-server", batch_no_wait=True)
# ... later ...
manager.ingest()

# Or download locally
manager.ingest(download=True, output_dir=Path("./local_results"))
```
