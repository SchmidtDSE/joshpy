# Josh Fixes Required Before joshpy `feat/batch-run` Merge

Two separate bugs surfaced during pre-merge integration testing of joshpy's `feat/batch-run` branch against the current `joshsim-fat-dev.jar`. Both are small, localized changes in `josh` itself. Neither is a joshpy-side regression — joshpy is doing the right thing in both cases.

This doc captures the investigation trail and proposed fix so the josh team can land them quickly. joshpy's side of the pre-merge work is tracked separately (Track A).

---

## Fix 1: `preprocessBatch` mis-propagates the `<dataFile>` positional argument to the K8s pod

### Symptom

When joshpy invokes `preprocessBatch` via the K8s target (joshpy passes all paths as absolute, per its `.resolve()` convention), the pod fails with:

```
Preprocessing failed: Failed to stream on patches:
java.io.IOException: Failed to open NetCDF file:
/tmp/work/tmp/test5_workdir/data.nc (No such file or directory)
```

The actual file is at `/tmp/work/data.nc` (confirmed in pod download logs). The pod is concatenating `$WORK_DIR` with the full client-local absolute path.

Reproducible with blocking mode OR `--no-wait` mode. Not async-specific.

### Root cause

Three artifacts in the flow:

1. **Client CLI** ([`src/main/java/org/joshsim/command/PreprocessBatchCommand.java:55-71`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/PreprocessBatchCommand.java#L55-L71))
    - `<input>` positional is typed `File` (line 56)
    - `<dataFile>` positional is typed `String` (line 62) with description **"Data file name within the input directory"**
    - `<outputFile>` positional is typed `File` (line 71)

2. **PreprocessParams construction** ([`PreprocessBatchCommand.java:203-208`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/PreprocessBatchCommand.java#L203-L208)):
   ```java
   final PreprocessParams params = new PreprocessParams(
       dataFile, variable, units, outputFile.getName(),
       //  ^^^^^^^^^                       ^^^^^^^^^^^^^^^^^^^
       //  raw String, not basename'd      .getName() => basename
       crs, horizCoordName, vertCoordName, timeDim,
       ...
   );
   ```

   **The inconsistency is right here**: `outputFile.getName()` gives the basename (e.g. `result.jshd`), but `dataFile` is passed through verbatim. If the user passes `/abs/path/data.nc`, that's exactly what reaches `PreprocessParams.dataFile`.

3. **K8s target env var** ([`src/main/java/org/joshsim/pipeline/target/KubernetesPreprocessTarget.java:219`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/pipeline/target/KubernetesPreprocessTarget.java#L219)):
   ```java
   envVars.add(plainEnvVar("JOSH_DATA_FILE", params.getDataFile()));
   ```
   Propagates the raw (possibly absolute) string to the K8s Job spec.

4. **Pod entrypoint** ([`cloud-img/preprocess-entrypoint.sh:51-54`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/preprocess-entrypoint.sh#L51-L54)):
   ```sh
   java -jar "$JAR" preprocess "$SCRIPT" "$JOSH_SIMULATION" \
     "$WORK_DIR/$JOSH_DATA_FILE" "$JOSH_VARIABLE" "$JOSH_UNITS" \
     "$WORK_DIR/$JOSH_OUTPUT_FILE" \
     $OPTS
   ```
   Concatenates `/tmp/work/` + `/abs/path/data.nc` = `/tmp/work//abs/path/data.nc`. Broken.

### Why `PreprocessParams`'s javadoc already says the right thing

[`PreprocessParams.java:40`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/pipeline/target/PreprocessParams.java#L40) already documents the intended contract:

```java
/**
 * @param dataFile Filename of the data file within workDir.
 */
```

So the fix is to enforce the contract at construction time, in `PreprocessBatchCommand.java:203-208`.

### Why `stageDirectory` works fine (doesn't exhibit the bug)

`stageDirectory` at [`PreprocessBatchCommand.java:241-254`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/PreprocessBatchCommand.java#L241-L254) uses `basePath.relativize(file)` to compute each object's path in MinIO. So uploaded file object names are already basenames (or relative paths within `inputDir`). The upload object name and the value needed for `JOSH_DATA_FILE` should be the SAME string — they both refer to "the file's path relative to workDir".

The upload path does this correctly. The env var doesn't. That's the bug.

### Proposed fix

In [`PreprocessBatchCommand.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/PreprocessBatchCommand.java), normalize `dataFile` to its path relative to `inputDir` before constructing `PreprocessParams`. ~5 LoC, one-to-one with the `outputFile.getName()` already-existing normalization.

```java
// Replace current line 203-208:
final Path inputDirPath = inputDir.toPath().toAbsolutePath();
final Path dataFilePath = Paths.get(dataFile).toAbsolutePath();
final String relativeDataFile;
if (dataFilePath.startsWith(inputDirPath)) {
    // Client passed an absolute path under inputDir — relativize it.
    relativeDataFile = inputDirPath.relativize(dataFilePath).toString();
} else if (!Paths.get(dataFile).isAbsolute()) {
    // Client passed a relative path — already matches contract.
    relativeDataFile = dataFile;
} else {
    throw new IllegalArgumentException(
        "dataFile '" + dataFile + "' is outside input directory '" + inputDir + "'. "
        + "Data files must live under the input directory for batch dispatch."
    );
}

final PreprocessParams params = new PreprocessParams(
    relativeDataFile, variable, units, outputFile.getName(),
    crs, horizCoordName, vertCoordName, timeDim,
    timestep.isBlank() ? null : timestep,
    defaultValue, parallel, amend
);
```

### Why fix in `PreprocessBatchCommand` and not in `KubernetesPreprocessTarget`

Three reasons:

1. **Single source of truth.** `PreprocessParams`'s javadoc already documents the contract ("Filename of the data file within workDir"). Fixing it at param construction enforces that contract for every target (K8s, HTTP, future targets). Fixing it only in `KubernetesPreprocessTarget` leaves the same bug waiting in `HttpPreprocessTarget` and any future target.

2. **Symmetry with `outputFile.getName()`.** The fix matches the existing idiom right next to it on the same line.

3. **Better error message.** If the user gives an absolute path outside `inputDir`, we can raise a clear error instead of silently constructing a path the pod can't resolve.

### Test plan (josh side)

- Unit test: pass `dataFile="/abs/path/to/inputdir/data.nc"` with `input="/abs/path/to/inputdir/sim.josh"`, assert `PreprocessParams.getDataFile() == "data.nc"`.
- Unit test: pass `dataFile="data.nc"` with `input="sim.josh"` (relative), assert `PreprocessParams.getDataFile() == "data.nc"`.
- Unit test: pass `dataFile="/unrelated/path/data.nc"`, assert `IllegalArgumentException`.
- E2E: `cd /tmp/inputdir && java -jar joshsim.jar preprocessBatch /tmp/inputdir/sim.josh Main /tmp/inputdir/data.nc var units result.jshd --target=gke-test ...` — should now succeed (currently fails).

### Test plan (joshpy side after the josh fix ships)

Re-run `tests/test5_preprocess_async.py` and `tests/test5b_blocking.py` from [joshpy e2e reports](/tmp/e2e_reports/05_preprocess_async.md). Both should go from red to green without any joshpy-side changes.

---

## Fix 2: `MinimalEngineBridge.getExternal` hardcodes `.jshd` extension — can't reach `.jshdz` files

### Symptom

When a simulation runs against a workdir containing `soil_quality.jshdz` (but not `soil_quality.jshd`), the pod fails with:

```
java.lang.RuntimeException: Failed to open stream from working directory: soil_quality.jshd
    at org.joshsim.lang.io.JvmWorkingDirInputGetter.loadFromWorkingDir(JvmWorkingDirInputGetter.java:46)
    at org.joshsim.lang.io.JvmWorkingDirInputGetter.readNamePath(JvmWorkingDirInputGetter.java:26)
    at org.joshsim.lang.io.JvmInputGetter.open(JvmInputGetter.java:29)
    at org.joshsim.precompute.JshdExternalGetter.getResource(JshdExternalGetter.java:43)
    at org.joshsim.precompute.MultiFormatExternalGetter.getResource(MultiFormatExternalGetter.java:40)
    at org.joshsim.lang.bridge.MinimalEngineBridge.lambda$getExternal$4(MinimalEngineBridge.java:197)
```

Trace reads: the engine asked for `soil_quality.jshd` (the uncompressed form), even though only `soil_quality.jshdz` exists on disk.

### Root cause

josh#427 introduced `MultiFormatExternalGetter` so the JVM runtime could transparently read both `.jshd` and `.jshdz`. The dispatcher itself is correct — it routes by the extension of the filename it receives:

**[`src/main/java/org/joshsim/precompute/MultiFormatExternalGetter.java:38-49`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/precompute/MultiFormatExternalGetter.java)**:

```java
@Override
public DataGridLayer getResource(String name) {
    if (name.endsWith(".jshdz")) {
        return jshdzGetter.getResource(name);
    } else if (name.endsWith(".jshd")) {
        return jshdGetter.getResource(name);
    } else {
        throw new IllegalArgumentException(
            "Expected a .jshd or .jshdz file name. Got: " + name
        );
    }
}
```

**The bug is upstream of this dispatcher**, in the caller that synthesizes the filename from the `external "<name>"` reference:

**[`src/main/java/org/joshsim/lang/bridge/MinimalEngineBridge.java:193-198`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/lang/bridge/MinimalEngineBridge.java#L193-L198)**:

```java
@Override
public EngineValue getExternal(GeoKey key, String name, long step) {
    String fileName = name.endsWith(".jshd") ? name : name + ".jshd";
    //                                                        ^^^^^^^
    //       UNCONDITIONALLY appends ".jshd", never considers ".jshdz"
    DataGridLayer layer = externalData.computeIfAbsent(name,
        k -> externalResourceGetter.getResource(fileName));
    return layer.getAt(key, step);
}
```

So `getExternal("soil_quality")` unconditionally asks `MultiFormatExternalGetter` for `soil_quality.jshd`. The dispatcher routes it to `JshdExternalGetter` (correctly, per its extension), which then fails to find the `.jshd` file on disk (since only `.jshdz` is there).

The `MultiFormatExternalGetter` can't rescue this — it only ever sees the `.jshd`-suffixed name.

### Evidence this is a recent introduction

Same block handles `.jshc` config files just a few lines below (line ~208):
```java
String configFileName = configName.endsWith(".jshc") ? configName : configName + ".jshc";
```
`.jshc` doesn't have a compressed equivalent, so that line is fine. But the `.jshd` line assumed the same single-extension world and got left behind when `.jshdz` was added.

### Proposed fix (Option A — cleaner, one place)

Move the extension-probing into `MultiFormatExternalGetter` so the dispatcher owns format resolution end-to-end. `MinimalEngineBridge` stops synthesizing extensions.

**[`MultiFormatExternalGetter.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/precompute/MultiFormatExternalGetter.java)**:

```java
@Override
public DataGridLayer getResource(String name) {
    if (name.endsWith(".jshdz")) {
        return jshdzGetter.getResource(name);
    } else if (name.endsWith(".jshd")) {
        return jshdGetter.getResource(name);
    }
    // Name without recognized extension — probe both.
    // Prefer .jshdz (assume authors compress when they can); fall back to .jshd.
    try {
        return jshdzGetter.getResource(name + ".jshdz");
    } catch (RuntimeException compressedMiss) {
        try {
            return jshdGetter.getResource(name + ".jshd");
        } catch (RuntimeException uncompressedMiss) {
            throw new RuntimeException(
                "External resource not found as " + name + ".jshdz or "
                + name + ".jshd", compressedMiss
            );
        }
    }
}
```

**[`MinimalEngineBridge.java:193-198`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/lang/bridge/MinimalEngineBridge.java#L193-L198)** — drop the extension synthesis:

```java
@Override
public EngineValue getExternal(GeoKey key, String name, long step) {
    DataGridLayer layer = externalData.computeIfAbsent(name,
        k -> externalResourceGetter.getResource(name));
    return layer.getAt(key, step);
}
```

### Proposed fix (Option B — smallest diff, keep existing responsibility split)

Leave the dispatcher alone; fix the synthesizer. `MinimalEngineBridge.getExternal` probes `.jshdz` first and falls back to `.jshd`:

```java
@Override
public EngineValue getExternal(GeoKey key, String name, long step) {
    DataGridLayer layer = externalData.computeIfAbsent(name, k -> {
        // Try compressed first, then uncompressed. Bare names only —
        // if caller passed a fully-suffixed name, pass through.
        if (name.endsWith(".jshd") || name.endsWith(".jshdz")) {
            return externalResourceGetter.getResource(name);
        }
        try {
            return externalResourceGetter.getResource(name + ".jshdz");
        } catch (RuntimeException e) {
            return externalResourceGetter.getResource(name + ".jshd");
        }
    });
    return layer.getAt(key, step);
}
```

### Recommendation

**Option A** — moves the knowledge of "which extensions exist for external data" into the dispatcher class whose explicit purpose is format routing. Future format additions (e.g., an `.jshd_stream` variant) only need to be taught to `MultiFormatExternalGetter`. And `MinimalEngineBridge` stops having to know the extension list at all.

Option B is fine if the team prefers a minimal diff, but it encodes a single static fallback order at the bridge layer.

### Test plan (josh side)

- Unit test: `MultiFormatExternalGetter.getResource("foo")` where only `foo.jshdz` is present — returns valid layer.
- Unit test: `MultiFormatExternalGetter.getResource("foo")` where only `foo.jshd` is present — returns valid layer.
- Unit test: neither present — raises clear error mentioning both extensions.
- Unit test: both present — prefers `.jshdz` (per recommendation) or `.jshd` (per team call; document the chosen precedence).
- E2E: run a simulation against a workdir with only `.jshdz` external data. Currently fails; should succeed.

### Test plan (joshpy side after the josh fix ships)

Re-run [`tests/test4_jshdz.py`](/tmp/test4_jshdz.py) from [joshpy e2e reports](/tmp/e2e_reports/04_jshdz_pipeline.md). The assemble-workdir + stage + dispatch portion is already green; only the pod-side read currently fails. After this fix, the full E2E should go green end-to-end.

---

## Summary for josh triage

| Fix | File | Lines | Size | Risk |
|-----|------|-------|------|------|
| 1. preprocess absolute-path propagation | `src/main/java/org/joshsim/command/PreprocessBatchCommand.java` | 203-208 | ~15 LoC | LOW — localized; unit-testable; explicit error on out-of-dir inputs |
| 2. `.jshdz` auto-resolution for `external` | `src/main/java/org/joshsim/precompute/MultiFormatExternalGetter.java` + `src/main/java/org/joshsim/lang/bridge/MinimalEngineBridge.java` | ~49 + ~197 | ~15-25 LoC | LOW — additive fallback; default case behavior unchanged for already-suffixed names |

Both are independent; can land as two small PRs in parallel. Once merged, rebuild `joshsim-fat-dev.jar` and joshpy's `get-jars` pulls in the fixed JAR for re-validation.

## Artifacts referenced

- joshpy E2E reports: `/tmp/e2e_reports/`
- Failing joshpy tests: `/tmp/test4_jshdz.py`, `/tmp/test5_preprocess_async.py`, `/tmp/test5b_blocking.py`
- joshpy branch: `feat/batch-run` (PRs [#36](https://github.com/SchmidtDSE/joshpy/pull/36) merged, [#37](https://github.com/SchmidtDSE/joshpy/pull/37) merged, [#38](https://github.com/SchmidtDSE/joshpy/pull/38) open)
- Pod entrypoint (reference): [`cloud-img/preprocess-entrypoint.sh`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/preprocess-entrypoint.sh)
