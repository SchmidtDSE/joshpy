# Registry provenance model: label, tags, status — and bucket-resident analysis

> **Status: PROPOSED (design record).** Consolidates three downstream asks from
> the I&M validation matrix (bucket aggregation without local ingest; tag
> currency; run supersession/annotation) into one coherent registry model,
> rather than three bolt-on features. No real users yet — this is the moment to
> get the data model right. Nothing here is implemented; this is the spec to
> build from.

## TL;DR

Three downstream pains all trace to **one missing concept**: the registry has no
first-class notion of *which run_hash is canonical*. Today "current" is an
emergent side-effect of a label-renaming hack (`on_collision="timestamp"`), so:

- tags can't distinguish current from archived,
- supersession keeps no record of *why* or *what replaced what*, and
- aggregating "the current results" from the bucket means re-deriving currency
  and export paths per project.

The fix is to stop overloading `label`, and give each concern exactly one home:

> **`run_hash`** is *what it is* · **`label`** is *what I call it* ·
> **tags** are *what it's about* · **`status`** is *whether I should use it*.

"Current" stops being stored or inferred from label strings — it **collapses
entirely into `status = 'active'`**. Supersession replaces the timestamp-mangling
hack. Bucket aggregation then composes on top of data the registry *already
captures* (`run_outputs`), with no ingest step.

---

## 1. The real duplication: `label` is doing three jobs

A label like `JOTR001_spinup` (bare) vs `JOTR001_spinup_20260729_183643`
(archived) silently encodes three unrelated things:

1. **a name** — the handle you `resolve_label()` to get the run (its only
   legitimate job);
2. **currency** — bare = current, timestamp-suffixed = superseded (the
   collision-rename at `registry.py:label_run`);
3. **attributes** — `site=JOTR001`, `treatment=spinup`, denormalized into the
   string, which analysis then *re-parses back out*.

Job (2) is why tags can't tell current from archived. Job (3) is why analysis
regexes labels/filenames. So the fix is not "merge label/tag/status" — it is to
**strip jobs (2) and (3) off `label`** and let it be only a name.

## 2. The model — four concepts, one home each

| Question | Home | Vocabulary | Cardinality | Drives behavior? |
|---|---|---|---|---|
| What **is** it? | `run_hash` | content hash | immutable PK | — |
| What do I **call** it? | `label` (`job_configs.label`) | free string | 1:1, unique, resolvable | no |
| What is it **about**? | tags (`run_tags`) | open (site, treatment, biome, join keys) | many:many | no |
| Should I **use** it? | `status` (`job_configs.status`) | closed enum | 1:1 | yes |

- **`label` stays, demoted to a pure alias.** You still need `resolve(name) → the
  one run`; tags can't give that (they're deliberately many:many). Nothing else.
- **tags become the sanctioned home for run attributes** — anything carried into
  analysis (site, treatment, scenario) or used as a **join key** linking registry
  runs to Josh's CSV output. This is the fix for label-string-parsing: stop
  encoding `site`/`treatment` in the label; tag them.
- **`status` is the lifecycle axis** — closed enum `{active, superseded, bad}`,
  and it *drives behavior* (reruns, pooling, default filtering). **"Current" is
  not a fourth concept — it is `status = 'active'`.**

### Why `status` earns a typed column and is *not* just a reserved tag

`status` looks like a tag (a value keyed by run_hash), so "just make it a tag"
feels DRY. It is **false DRY** — coincidental shape, different semantics:

| | tags | status |
|---|---|---|
| vocabulary | open | closed enum |
| cardinality | many:many | 1:1 |
| effect | inert (analysis reads them) | drives reruns / pooling / default filter |
| validation | none | enum-checked |

Collapsing them would put `status` next to `site` in `list_tag_keys()`, lose enum
validation, and make read-path special-casing illegible. Reuse a storage
substrate when the *semantics* match, not when the shapes rhyme. `status` lives
as a typed column on `job_configs`, right next to `label` — the row already keyed
1:1 by `run_hash`.

## 3. Decisions locked (this turn)

1. **Retire `on_collision="timestamp"` and `resolve_latest()`** in favor of
   explicit supersession semantics (§5).
2. **Attributes are the sanctioned place** for analysis factors and join keys;
   `label` becomes a pure alias. The vocabulary is **renamed `tags` →
   `attributes`** (§11.3) — done now while there are no real users. Throughout
   this memo, "tags" refers to the renamed `attributes` API.
3. **Read "current" by default**, via a non-materialized `cell_data_current`
   **view** (zero storage/refresh cost) for local reads and a
   `current_only=True` default on structured/remote read methods (§6).
4. **Completeness is a read-side concept keyed on attributes, not run_hashes**,
   asserted by a persisted external `TargetDesign` (§12). It never dispatches.
5. **The only dispatch-side change is one collision rule** — sweep dedup treats
   a cell whose only occupant is `bad` (or missing) as unsatisfied (§11.1).
6. **Attribute keys stay open** — documented conventions only (§11.2).

## 4. Schema changes (additive, backward-compatible)

Two columns on `job_configs`, plus a view. No new table.

```sql
ALTER TABLE job_configs ADD COLUMN status        VARCHAR;   -- NULL == 'active'
ALTER TABLE job_configs ADD COLUMN superseded_by VARCHAR;   -- run_hash, nullable
ALTER TABLE job_configs ADD COLUMN status_reason TEXT;      -- free-text 'why'
ALTER TABLE job_configs ADD COLUMN status_updated_at TIMESTAMP;

CREATE VIEW cell_data_current AS
  SELECT c.* FROM cell_data c
  JOIN job_configs j USING (run_hash)
  WHERE coalesce(j.status, 'active') = 'active';
```

- Applied through the existing probe-and-`ALTER` path (`registry._migrate_schema`,
  `registry.py:842`). Old databases get `status = NULL`, treated as `'active'`
  everywhere via `coalesce` — **zero behavior change for existing data.**
- `status ∈ {active, superseded, bad}`; `NULL` is read as `active`.

## 5. Supersession replaces label-mangling

### API

```python
registry.mark_run(run_hash, status="bad",
                  reason="spinup used scenario=ssp245, not historical")
registry.mark_run(old_hash, status="superseded", superseded_by=new_hash,
                  reason="rerun with scenario=historical")
registry.get_run_status(run_hash) -> RunStatus   # {status, superseded_by, reason, updated_at}
```

`mark_run` validates the enum and the `superseded_by` link (target must exist).

### The collision path (drop-in replacement for `"timestamp"`)

`label_run(run_hash, label)` keeps raising on collision by default. The archival
behavior changes from *rename the old label* to *supersede the old run*:

```python
label_run(new_hash, "JOTR001_spinup", on_collision="supersede", reason=...)
# old run holding "JOTR001_spinup":  label = NULL, status = 'superseded',
#                                    superseded_by = new_hash, reason = ...
# new run:                           label = "JOTR001_spinup", status = 'active'
```

- The value `"timestamp"` is **removed**; `"supersede"` replaces it. `force=True`
  (silent reassign, no provenance) is unchanged — it stays for the "I mislabeled,
  just fix it" case.
- Because the archived run **releases its label** (`label = NULL`), it no longer
  needs a timestamp suffix, and it *automatically disappears* from `list_labels()`
  and (with §6) from `find_tagged(current_only=True)` — the pollution problem
  dissolves.
- Convenience sugar: `registry.supersede(new_hash, replaces="JOTR001_spinup",
  reason=...)` wrapping the above, for the partial-rerun story.

### Provenance, replacing `resolve_latest()`

Archived runs are reachable by walking the `superseded_by` chain, not by
prefix-matching mangled label strings:

```python
registry.run_history("JOTR001_spinup") -> [RunDetail, ...]   # current → oldest
```

### Retirement checklist

- `registry.label_run`: drop `"timestamp"`; add `"supersede"` + `reason`.
- Remove `registry.resolve_latest()` (its only use case — "latest among
  `baseline*`" — is gone once labels aren't suffixed). Add `run_history()`.
- `SweepManagerBuilder.with_label(..., on_collision=...)` (`sweep.py:2062`) and the
  build wiring (`sweep.py:2264`): `"timestamp"` → `"supersede"`.
- Update tutorials `docs/tutorials/complete-example.qmd`,
  `docs/tutorials/manual-workflow.qmd` and `tests/test_registry.py`,
  `tests/test_sweep.py`.

## 6. "Current" by default — the honoring rule

"Current" is `status = 'active'` (via `coalesce(status,'active')`). Honoring it by
default has one hard boundary: **`registry.query(sql)` is a raw DuckDB passthrough
(`registry.py:883`) and cannot be transparently filtered.** So honoring is applied
at the *structured* surfaces plus a view:

| Read path | How "current" is honored |
|---|---|
| ad-hoc local SQL | query `cell_data_current` view instead of `cell_data` (opt-in) |
| `find_tagged` | new `current_only=True` **default** (§7) |
| `query_remote` (§8) | `current_only=True` **default**; applied in the join |
| cell_data convenience queries (`CellDataLoader`/`DiagnosticQueries`) | back onto `cell_data_current` |
| raw `query(sql)` | **not** auto-filtered — documented; use the view |

Every filter is `coalesce(status,'active') = 'active'`, always with an explicit
`current_only=False` / `include_superseded=True` escape hatch. The view is a plain
(non-materialized) view: computed at query time, no refresh burden.

## 7. Tag currency (downstream ask 2) — falls out of `status`

Once currency is `status = 'active'`, tag currency is a join, not label-string
archaeology:

```python
registry.find_tagged("site", "JOTR001", current_only=True)   # NEW default True
registry.find_current_tagged("site", "JOTR001")
#   -> [(label, run_hash, tags), ...]   already resolved, active only
```

`find_current_tagged` joins `run_tags` ↔ `job_configs` (for label + status),
returning resolved tuples so no consumer re-implements the label↔tag
cross-reference. Note: building `current_only` *without* §5 would force it to
reconstruct "current" from the bare-label heuristic — the exact fragile thing we
are deleting. Ask 2 sits **on** §5, not beside it.

## 8. Bucket-resident aggregation without ingest (downstream ask 1)

This is the ask most worth getting right, and the new model plus *existing*
machinery makes it far cleaner than the current workaround (scan `s3://.../*/*.csv`
with `filename=true`, `regexp_extract` run_hash from the filename, join a
hand-maintained `label → current_run_hash` dict, `GROUP BY`, cache to CSV).

### What the registry already has (three quarters of it)

1. **`configure_s3()`** (`registry.py:110`) — S3/MinIO secret setup. Exists.
2. **`query()`** already runs S3 SQL (`read_csv_auto('s3://...')`) — the passthrough
   works today.
3. **`run_outputs.file_path` already stores the resolved export URI per run
   execution**, tagged `export.patch` / `debug.organism` / …, written by
   `_register_job_outputs` on **any registry-attached run, regardless of
   `load_results`/`auto_ingest`** (`jobs.py:2305`). *The registry already knows the
   exact bucket URI for every `(run_hash, replicate, export_type)`.*

So we do **not** parse run_hash out of filenames, and we do **not** hand-maintain
a current-dict — the registry *is* the current-dict.

### 8a. Export-URI accessor (the "export path convention as a public method")

```python
registry.get_output_uris(label_or_hash, output_type="patch", current_only=True)
#   -> [(run_hash, replicate, uri), ...]
```

Primary source is `run_outputs ⋈ job_runs ⋈ job_configs` (jar-free):

```sql
SELECT jr.run_hash, jr.replicate, ro.file_path
FROM run_outputs ro
JOIN job_runs   jr USING (run_id)
JOIN job_configs jc USING (run_hash)
WHERE ro.output_type = 'export.' || ?          -- 'patch'
  AND coalesce(jc.status,'active') = 'active';  -- current_only
```

**Fallback** when outputs were never registered (older/no-registry runs): compute
from `cli.inspect_exports()` + `ExportPaths.resolve_path(run_hash=…, replicate=…)`
over the expected replicate range. That path needs a `cli`, so it lives on
`SweepManager.remote_export_uri(...)`; `registry.get_output_uris` stays jar-free.
Since `run_outputs` is populated on every registry-attached run, jar-free is the
common path.

### 8b. `query_remote` — aggregate directly against the bucket

```python
registry.query_remote(
    "treeCount", agg="mean",
    group_by=["label", "step", "replicate"],
    output_type="patch",
    current_only=True,       # default
    where=None,              # optional row filter on CSV columns
    cache=None,              # optional local parquet/csv path
) -> DataFrame
```

Mechanics (one scan, provenance from the registry, not the path):

1. Build a **URI manifest** of current runs via §8a → rows of
   `(uri, run_hash, replicate, label)`.
2. `read_csv_auto([uris], filename=true)` — one scan over the known list.
3. **Join `filename == manifest.uri`** to attach `run_hash / replicate / label`
   from the registry (never regex the path), which also enforces `current_only`
   (archived URIs simply aren't in the manifest — stale files in the same folder
   are excluded for free).
4. `SELECT <group_by>, <agg>(<variable>) ... GROUP BY <group_by>`.
5. Return a DataFrame; optionally cache to `parquet`/`csv`.

Properties:

- **No ingest.** Nothing touches `cell_data`; this is the whole point of the
  `load_results:false` batch.
- **Jar-free** when `run_outputs` is populated (the norm).
- **S3 creds:** ensure `configure_s3()` has run, or resolve from env
  (`MINIO_ENDPOINT`/`_ACCESS_KEY`/`_SECRET_KEY`) exactly as `ingest_results`'
  `_resolve_s3_credentials` already does — shared, not reinvented.
- **Grain:** `group_by` accepts any CSV column (`step`, position) plus the injected
  registry columns (`label`, `run_hash`, `replicate`). `group_by=[label, step,
  replicate]` → per-patch mean, which is the reported grain.

### 8c. `check_remote_consistency()` — the bucket-aware sibling

Mirrors `check_consistency()` (`registry.py:2729`) but against the bucket, so
"did the corrected rerun land everywhere it should have" stops being answered by
eyeballing row counts:

```python
registry.check_remote_consistency(current_only=True) -> list[ConsistencyIssue]
```

For each active run, expected `(run_hash, replicate)` set (from recorded
`job_runs` / the sweep's configured count) is checked against CSV presence at the
expected URIs. New issue kinds:

- `missing_remote_output` — expected URI absent (**error**): a replicate that
  never landed.
- `remote_count_mismatch` — fewer/more CSVs than the target replicate count
  (**error**).
- `orphan_remote_output` — a CSV present in the folder that belongs to a
  **non-active** run_hash (**warning**): the stale/superseded files the workaround
  filters by hand.

## 9. How the three asks collapse onto one model

```
             status (active / superseded / bad)   ← §5, the keystone
            /            |                    \
  find_tagged      cell_data_current       query_remote / check_remote_consistency
  current_only        view                  (current_only via the same status join)
   (ask 2)          (ask 3 local)                    (ask 1)
```

`status` is the single source of "current." Ask 2 is a join on it. Ask 1 is a join
on it plus the URIs `run_outputs` already stores. No concept is defined twice.

## 10. Phasing

1. **Phase 1 — `status` + supersession + view. ✅ IMPLEMENTED (v0.0.9.24).**
   Columns + migration; `mark_run`/`get_run_status`/`RunStatus`; `label_run`
   `"supersede"` (removed `"timestamp"`/`resolve_latest`, added `run_history`
   and `supersede` sugar); `cell_data_current` view; `status` surfaced in
   `ConfigInfo`/`describe_run`; `SweepManager.with_label(on_collision=,
   reason=)` wired. *Useful standalone.*
2. **Phase 2 — attribute currency + target coverage. ✅ IMPLEMENTED
   (v0.0.9.25).** Run-level `attributes` slice (`set_attributes`/
   `get_attributes`/`list_attribute_keys`, currency-aware `find_by_attribute`/
   `find_current_by_attribute` + `AttributeMatch`); the general `run_tags`
   facility (session/run_id/custom scopes) kept as `tags` (§11.3).
   `DiagnosticQueries` reads through the `cell_data_current` view (with a
   `_refresh_current_view` hook so dynamically-added variable columns stay
   visible). `TargetDesign`/`Requirement`/`check_design`/`TargetCoverageReport`
   coverage layer (§12) with `target_designs`/`target_requirements` tables. The
   one dispatch-side rule (§11.1): a `bad` run is `reset_run`'d (results cleared,
   status reactivated, config kept) and re-dispatched in full.
3. **Phase 3 — remote aggregation.** `get_output_uris` →
   `SweepManager.remote_export_uri` (fallback) → `query_remote` →
   `check_remote_consistency`.

## 11. Resolved decisions (superseding the open questions)

The three open questions are now decided.

### 11.1 `bad` vs `superseded`, and what actually triggers a rerun

Currency (`status`) and **completeness** are different questions. `status`
answers "is *this* occupant usable"; completeness answers "does every required
factor combination have enough usable occupants" — and it is keyed on
**attributes, not run_hashes** (§12).

Content hashing already handles the most common badness cause for free. If a run
was `bad` because it used the wrong external data, correcting that data changes
its inputs → **new run_hash**, so a re-dispatch runs it as a fresh run with no
collision at all; the old run stays `bad` and drops out of every read. So "bad
triggers a rerun" splits into two cases, and only one needs new machinery:

| Why it's bad | Re-dispatch behavior | New mechanism? |
|---|---|---|
| **Wrong inputs** (e.g. corrected external data) | fixed inputs → new hash → dispatched as a fresh run; old `bad` run excluded | **None** — content addressing does it |
| **Same inputs** (transient failure, partial output) | identical hash → dedup currently *skips* it | **Yes** — dedup must treat an existing-`bad` occupant as unsatisfied → redo |

So the only dispatch-side change is one precise rule in the SweepManager
collision wiring: **a grid cell whose only occupant is `bad` (or missing) is
unsatisfied and gets dispatched.** Everything else falls out of hashing +
`status`. Completeness itself is a **read-side** concern (§12), never triggers a
run, and makes no assumption that a sweep exists.

### 11.2 Attribute key reservations

**Stay open.** Documented conventions only; no reserved key set beyond the
auto-injected `run_hash` custom tag.

### 11.3 Rename `tags` → `attributes`

**Done as a run-level slice** (v0.0.9.25), not a blanket rename. On building it,
"tags" turned out to be a whole *scoped* metadata subsystem (`run_hash` +
`session_id` + `run_id` + custom synthetic scopes, ~12 methods, a JSON
`run_tags` table) — genuinely metadata, and "attribute_by_run_id" reads wrong.
So `attributes` is the sanctioned name for the **`run_hash` slice only** — what
the memo's model actually means by attributes (run factors + join keys):
`set_attributes` / `get_attributes` / `list_attribute_keys` /
`find_by_attribute` / `find_current_by_attribute`. The general
`tag_by_session_id` / `tag_by_run_id` / `tag_custom` / `find_tagged` facility
keeps the `tags` name and the underlying `run_tags` store is unchanged (no
migration). Throughout this memo, "attributes" = that run-level slice; "tags" =
the general facility.

## 12. Target designs — coverage over attributes (Phase 2 rider)

Completeness is asserted by an **external, persisted `TargetDesign`** that makes
no claim on how runs were produced. It relies on the user tagging runs
diligently, and in exchange assumes nothing about JobConfig/sweep construction —
so yaml-dispatched jobs, interactive runs, and sweeps all count toward the same
requirement identically. This is the deliberate trade: fewer assumptions, at the
cost of resting on the user's tagging discipline.

### Model

A `TargetDesign` is a named set of `Requirement`s. Each requirement is a
conjunction of required `attributes` plus a `min_active` count:

```python
design = TargetDesign("jotr_fire_2026", requirements=[
    Requirement(attributes={"scenario": "historical", "treatment": "spinup"},    min_active=1),
    Requirement(attributes={"scenario": "historical", "treatment": "no_spinup"}, min_active=1),
])
registry.register_design(design)                      # persisted (see schema below)
registry.check_design("jotr_fire_2026") -> TargetCoverageReport
```

### Satisfaction

- **Superset match** — a run tagged `{scenario:historical, treatment:spinup,
  site:JOTR001}` satisfies `{scenario:historical, treatment:spinup}`. A
  requirement is as specific as it is written.
- **Count = distinct `active` run_hashes** matching; `min_active` default 1.
  `bad`/`superseded` runs never count — regardless of their hashes.
- **Replicate adequacy is out of scope here** — that is `check_consistency`'s job
  (replicates per hash). Coverage answers "is this cell occupied by enough
  *current* runs"; consistency answers "did each run land all its replicates."
  Orthogonal axes.

### `TargetCoverageReport`

Per requirement: `satisfied: bool`, `found` vs `required`, the matching active
run_hashes, and — crucially — the **non-active matches with their status**, so an
empty cell that *does* have runs reads as "2 runs present but both `bad`," not
"nothing ran." Top level: `complete: bool` plus the list of unmet requirements.
The unmet list is plain data; a caller (or SweepManager) may feed it into a
dispatch, but coverage never triggers one.

### Persistence

Two additive tables:

```sql
CREATE TABLE target_designs (
  name        VARCHAR PRIMARY KEY,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE target_requirements (
  design_name VARCHAR REFERENCES target_designs(name),
  attributes  JSON,          -- {key: value, ...}, the required conjunction
  min_active  INTEGER DEFAULT 1
);
```

`check_design` joins `target_requirements` against `run_tags ⋈ job_configs`
(active only) by attribute-superset, counts distinct current hashes per
requirement, and assembles the report. Pure read path; jar-free.
