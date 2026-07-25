# Josh Fixes for joshpy `feat/explicit-time-axis` Integration

This doc tracks josh-side issues found while integrating joshpy against a Josh JAR build. joshpy-side context lives in [tests/test_jshdz_integration.py](tests/test_jshdz_integration.py).

## Status

| # | Issue | Status |
|---|-------|--------|
| 1 | `preprocess` rejects `--time-*` flags | ✅ Fixed (confirmed 2026-07-25, JAR sha256 `2d93e898...`) |
| 2 | NetCDF (raster) preprocessing fails with "Error interpolating value for patch" | ✅ Fixed (confirmed 2026-07-25, JAR sha256 `29c89386...`) |
| 3 | `preprocess_netcdf()` sizes its stub script from the grid's `steps`, not the per-call `time_count` ([josh#494](https://github.com/SchmidtDSE/josh/pull/494) invariant) | ✅ Fixed joshpy-side, 2026-07-25 |

All five real-JAR integration tests in `tests/test_jshdz_integration.py` now pass:
`test_csv_compress_roundtrip`, `test_netcdf_temporal_compress_roundtrip`,
`test_netcdf_temporal_default_uncompressed_e2e`, `test_compress_default_false_unchanged_e2e`,
`test_shorter_and_longer_sources_both_preprocess_against_shared_grid`.

The original temporal-read fix was only ever verified against `.jshdz` (compressed) output.
`test_netcdf_temporal_default_uncompressed_e2e` closes that gap: same declared count/year axis,
same `external ... at year ...` read, against plain `.jshd` (`compress=False`, the default) —
confirms the fix isn't compression-path-specific.

---

## Issue 1: `preprocess` rejected `--time-*` flags ✅ FIXED

Confirmed by running `test_netcdf_temporal_compress_roundtrip` — the CLI invocation now includes
`--time-dim`, `--time-type`, `--time-start`, `--time-unit`, `--time-count`, `--time-increment`
without a "Unknown option" rejection.

## Issue 2: NetCDF preprocessing failed with "Error interpolating value for patch" ✅ FIXED

Root cause per the josh team's handoff doc (`joshpy Handoff: preprocess Contract`, shared
2026-07-25): a source variable with no `units` attribute previously failed with
`Error interpolating value for patch: <Patch@hash>`. This is now accepted — joshpy does not need
to synthesize a `units` attribute on generated NetCDF fixtures.

Confirmed independently of joshpy/pytest with the same minimal repro used to originally isolate
the bug (hand-written `.josh` + direct `java -jar ... preprocess ...`), against
`jar/joshsim-fat-dev.jar` (sha256 `29c8938697c36ee6834fcb7c8f3f7d1b7d383b31d53daa40682cb8727f2ac4f2`).
Exit code `0`, `.jshdz` produced successfully.

---

## joshpy changes made in response to the new JAR contract

The josh team's handoff doc clarified two things that required joshpy-side changes, neither of
which was a JAR bug — both were gaps/incorrect assumptions on the joshpy side:

### 1. `--no-time-dim` support added

The handoff doc documents a new flag for flat, timeless NetCDF sources (2D rasters): `--no-time-dim`,
which takes precedence over `--time-dim` and reads the single available slice for every grid
timestep. Previously joshpy had no way to emit this — `NetcdfPreprocessConfig.time_coord` defaulted
to `"time"` and there was no falsy path other than silently omitting `--time-dim` (which lets the
JAR fall back to its `calendar_year` default — almost never correct for an external source).

Changed `time_coord: str = "time"` to `time_coord: str | None = "time"` on
`NetcdfPreprocessConfig` (`joshpy/cli.py`) and `GridSpec.preprocess_netcdf()`
(`joshpy/grid/_core.py`). Passing `time_coord=None` now emits `--no-time-dim` instead of omitting
the flag. Default behavior (`"time"`) is unchanged. Covered by
`tests/test_cli.py::TestJoshCLI::test_preprocess_netcdf_no_time_dim`.

### 2. `test_netcdf_temporal_compress_roundtrip`'s consuming script was wrong

Once Issues 1 and 2 above were fixed, this test's *run* step (not preprocessing) newly failed with:

```
java.lang.IllegalStateException: Unable to resolve 'meta.time' as a value: no attribute,
variable, or built-in by that name is in scope here
```

This was never caught before because preprocessing itself was blocking the test from ever reaching
the run step. Root cause: the test's `.josh` script used `external temperature at time meta.time`,
but the declared axis is a **count** axis (`--time-type count --time-unit year`), and the handoff
doc's engine-semantics section is explicit that the read-clause keyword must match the axis's own
unit (`at year ...`) — `at time`/`meta.time` is only for **ISO** axes.

Two further wrinkles surfaced empirically (confirmed by direct `java -jar ... run ...` against the
same JAR, script by script) and are now baked into the test's generated `.josh`:

- `meta.year` returns the **raw 0-based simulation step** (a warning is printed:
  `meta.year is using raw simulation timestep 0, not a declared calendar`) unless the simulation
  itself declares a calendar. Since the test's grid steps are 0/1 but the declared axis starts at
  2015, the coordinate has to be computed explicitly: `year.step = meta.year + 2015 year`.
- The axis's declared unit (`year`) needs a matching `start unit year / alias years / end unit`
  block in the script, or the read fails with `No conversion exists between "years" and "year"`
  (this matches the handoff doc's "Unit conversion caveat for count axes" section).

Fixed in `tests/test_jshdz_integration.py`. Also corrected the test's final assertion: it expected
exactly 3 output rows (header + 2 steps), assuming `exportFiles.patch` aggregates to one row per
step. It doesn't — it writes one row per patch per step, so a 4×4-patch grid over 2 steps writes 33
rows. The assertion now checks per-step temperature value sets (ignoring uncovered cells, which
default to `0`) instead of a fixed row count.

### Repro environment

- JAR: `jar/joshsim-fat-dev.jar`, sha256 `29c8938697c36ee6834fcb7c8f3f7d1b7d383b31d53daa40682cb8727f2ac4f2`.
  `jar/joshsim-fat-prod.jar` (sha256 `8ca776f2b53aa4d1f15cf03f42af7130ef6f62e10a0d3217df92130af717f4c1`)
  was updated the same day but was not independently tested against this repro.
- Java: bundled in `.pixi/envs/dev`.
- joshpy tests: `pixi run -e dev test-integration` (5 passed, 17 skipped — MinIO tests require a
  running MinIO and are unaffected by this work) and `pytest tests/ -m "not integration"`
  (1131 passed).

---

## Issue 3: `preprocess_netcdf()`'s stub script ignored per-call `time_count` ✅ FIXED (joshpy-side)

Reported externally (SchmidtDSE/josh-models, `dev_fine` grid): [josh#494](https://github.com/SchmidtDSE/josh/pull/494)
added a strict check that a declared `--time-count` must equal the output-slice count derived
from `preprocess`'s throwaway stub simulation (`steps.high - steps.low + 1`). joshpy's
`GridSpec._render_preprocess_script()` always sized that stub from the grid's own
`timestep_count` (`grid.yaml`'s `steps`, or a grid-level `TimeAxis.count`), regardless of the
`time_count` actually declared for a given `preprocess_netcdf()` call. Any grid reused to
preprocess two sources of different native lengths (e.g. a 65-year historical series and an
86-year SSP scenario against the same `dev_fine` grid) failed preprocessing for whichever source
didn't match the grid's declared `steps` — not a caller misconfiguration; there was no way to
tell `preprocess_netcdf()` "this file is N timesteps long, independent of the grid's nominal
count."

This was not a JAR bug — [josh#494](https://github.com/SchmidtDSE/josh/pull/494)'s invariant is
correct; joshpy's stub sizing just never accounted for it.

### Fix

`_render_preprocess_script()` now accepts an optional `step_count`, and `preprocess_netcdf()`
resolves it per call: the axis's `time_count` when a temporal axis is declared (from `time=`, a
per-call `time_count=`, or the grid's own `TimeAxis`), `1` for a bare `time_instant` (single-slice
form), or `None` (falling back to the grid's `timestep_count`, preserving prior behavior) when no
temporal options are given at all. This mirrors how the real Josh model decouples spinup length
from scenario length rather than relying on one project-wide constant.

Confirmed by reverting the fix locally and re-running
`TestPreprocessMismatchedNativeLengths::test_shorter_and_longer_sources_both_preprocess_against_shared_grid`:
it reproduces the reported `--time-count must equal the number of output slices` error exactly,
and passes once the fix is restored. Also covered by five `tests/test_grid.py` unit tests
asserting the rendered stub's `steps.high` in each resolution path (per-call `time_count`,
grid-level `TimeAxis`, per-resource `time=` override, bare `time_instant`, and the no-axis
fallback).
