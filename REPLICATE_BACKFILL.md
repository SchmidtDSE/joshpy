# Replicate backfill: what joshpy needs from the Josh JAR

## TL;DR

joshpy can now reconcile a sweep to a target replicate count idempotently, and
applies the `fail`/`pool`/`skip` collision policies **uniformly across local,
Josh Cloud, and batch-remote** dispatch — *except* for one case it cannot do
without the JAR's help: **dispatching a specific subset of replicate indices on
`run` / `runRemote`.**

We ask Josh to let `run` and `runRemote` (ideally unified with `batchRemote`)
accept an **explicit array of replicate indices** to compute, e.g.

```
josh run sim.josh Main --replicate-indices=3,7,8
```

writing one output per index (`output_3.csv`, `output_7.csv`, `output_8.csv`).
This is strictly more general than `batchRemote`'s existing `--replicate-start`
offset, and it lets joshpy **backfill exactly the missing replicates** so a
"20-replicate" run again means indices **0–19**, not a sparse set.

---

## Background: where joshpy is today

The replicate **index is treated as a collision-avoidance tag, not a
coordinate** — what matters is the *count* of replicates. On that basis joshpy
now:

- **Ingests idempotently.** Identity is `(run_hash, replicate)`; re-ingesting an
  already-loaded index is a no-op (`RunRegistry.loaded_replicates`,
  `get_replicate_count` = `COUNT(DISTINCT replicate)`).
- **Pools by count.** `collision_policy="pool"` dispatches `target − have`
  replicates at a fresh, non-colliding offset (`max(existing)+1`).
- **Applies policy on every path.** `fail`/`pool`/`skip` are evaluated for local,
  cloud, and batch via `resolve_collision_action` + `list_output_replicates`
  (MinIO glob *and* local-filesystem glob).

What works **with no JAR change** (Layer 1, shipped):

| Policy | local / cloud | batch-remote |
|---|---|---|
| `fail` (default) | not enforced¹ | raises `SweepCollisionError` |
| `skip` (any existing → no-op) | ✅ | ✅ |
| `pool`, already complete (`have ≥ target`) → no-op | ✅ | ✅ |
| `pool`, partial top-up (`have < target`) | ⚠️ re-runs full target² | ✅ dispatches only the missing count |

¹ The silent-overwrite guard is a MinIO concern; local overwrites are visible, so
the default isn't newly blocked on local.
² Correct (ingest dedups so no duplicate rows), but the **compute is wasted** —
the whole target is recomputed and all-but-the-missing results are discarded at
ingest. This is the gap this document is about.

The reason is purely the CLI surface:

- `batchRemote` has `--replicate-start=K` → run replicates `K..K+N-1`
  (`BatchRemoteConfig.replicate_start`).
- `run` and `runRemote` accept only `--replicates=N` → always `0..N-1`, no offset
  and no index selection (`RunConfig` / `RunRemoteConfig` have no offset field).

So joshpy cannot tell a local/cloud run "compute only replicates 12–19" (let
alone "compute only 3, 7, 8").

---

## The ask: dispatch an explicit set of replicate indices

### Capability

Add to `run` and `runRemote` (and, ideally, fold `batchRemote`'s offset into the
same flag) the ability to run an **explicit list of replicate indices**:

```
--replicate-indices=<comma-separated ints>     # e.g. 3,7,8  or  12,13,14,15
```

Semantics:

- Run exactly those replicate indices (not a count starting at 0).
- Each index `i` produces the output whose `{replicate}` template slot is `i`
  (e.g. `output_3.csv`), identical to how `{replicate}` already resolves today.
- `--replicate-indices` and `--replicates`/`--replicate-start` are mutually
  exclusive; when indices are given, the count is `len(indices)`.

### Why an index array, not just an offset

We considered just adding `--replicate-start` to `run`/`runRemote` (mirroring
`batchRemote`). An explicit **index array** is better and subsumes it:

- **It restores dense, canonical numbering.** joshpy can dispatch
  `missing = set(range(target)) − existing`, so a 20-replicate run ends up with
  indices **0–19** instead of an ever-sparser `{1,4,5,6,7,…}` produced by
  append-at-`max+1`. The index becomes meaningful again.
- **It is true gap-fill.** If replicate 3 failed/was lost, joshpy backfills *3*
  specifically, not "one more at the end".
- **It subsumes offset.** `--replicate-start=K` for `N` is just
  `--replicate-indices=K,K+1,…,K+N-1`. One flag covers batch's current need and
  the new backfill need.

---

## Open question for Josh: per-index reproducibility (seeding)

For backfill to be *scientifically* meaningful (re-running "replicate 3" should
mean the same thing whenever it runs), the JAR should ideally derive each
replicate's RNG seed **deterministically from `(run_hash or base seed, replicate
index)`**, so:

- index → seed is stable across invocations, and
- backfilling index 3 reproduces the replicate that index 3 *would* have been in
  the original run.

If instead replicates draw fresh entropy per process, the index is purely a label
and backfill still works (any new draw is a valid replicate) — but then "dense
0–19" is cosmetic, not reproducible. Please confirm which model Josh uses; it
determines whether dense numbering carries reproducibility guarantees or is just
bookkeeping. (Today joshpy passes `--seed` only when explicitly set.)

---

## joshpy-side follow-up (once the JAR supports it)

Small and contained — mostly mirroring what `batchRemote` already has:

1. Add `replicate_indices: list[int] | None` to `RunConfig`, `RunRemoteConfig`,
   and `BatchRemoteConfig` (`joshpy/cli.py`); emit `--replicate-indices` in the
   `run` / `runRemote` / `batchRemote` arg builders.
2. Thread it through `to_run_config` / `to_run_remote_config` /
   `to_batch_remote_config` (`joshpy/jobs.py`).
3. Add a **gap-fill mode** to `_apply_collision_policy` (`joshpy/sweep.py`): when
   index dispatch is available, return the explicit
   `missing = sorted(set(range(target)) − existing)` instead of count + `max+1`
   offset. The `_CollisionAction` grows a `replicate_indices` field; the dispatch
   sites pass it on every path.
4. Drop the `warn_partial_backfill_unsupported` fallback for local/cloud once
   those paths can dispatch the missing indices.

No change needed to ingest or counting: `list_output_replicates`,
`loaded_replicates`, and `(run_hash, replicate)` identity are already index-aware,
so dense backfilled sets and existing sparse sets both load correctly.

### Behavior after the JAR change

| Scenario | Before (today) | After (index dispatch) |
|---|---|---|
| local, have `{0,1,2}`, target 10 | re-run all 10, ingest keeps 3–9 (7 wasted recomputes) | dispatch indices 3–9 only; result is dense `0–9` |
| any path, replicate 3 lost | append one at `max+1` (sparse) | backfill exactly index 3 (stays dense) |
| batch `--replicate-start=K` | offset only | expressed as an index range; same result |

---

## Summary of the request to Josh

1. **`--replicate-indices=<ints>` on `run` and `runRemote`** (and unify
   `batchRemote`'s offset into it). Run exactly those indices; output filenames
   use the index in the `{replicate}` slot.
2. **Confirm seeding semantics** — deterministic per `(seed/run_hash, index)` is
   preferred so backfilled replicates are reproducible.

With (1), joshpy makes `pool`/backfill dispatch only the missing replicates on
every execution path (no wasted recompute) and restores canonical `0..N-1`
replicate numbering.
