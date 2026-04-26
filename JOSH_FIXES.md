# Josh Fixes for joshpy `feat/batch-run` Integration

This doc tracks josh-side changes needed by joshpy. Some are already shipped; new ones are appended as they're identified during integration. joshpy-side counterparts live in [PR7_PLAN.md](PR7_PLAN.md).

## Status

| # | Fix | Status |
|---|-----|--------|
| 1 | `preprocessBatch` absolute-path propagation | ✅ Merged |
| 2 | `MinimalEngineBridge.getExternal` hardcoded `.jshd` extension | ✅ Merged |
| 3 | Pod run-entrypoint missing `cd $WORK_DIR` before `java -jar` | ✅ Merged |
| 4 | Pod entrypoint DNS readiness + `stageFromMinio` retry | ✅ Merged¹ |
| 5 | `batchRemote` accepts `--custom-tag` and propagates to pod | ✅ Merged² |
| 6 | `run` accepts `--replicate-start=K` + `JOSH_REPLICATE_OFFSET` env var in entrypoint | ✅ Merged² |

All six fixes confirmed against JAR `dcfeb60c124beffcff8ed4d7dd245e2c8c24abe29fbcaf3d4caea9b23ffa4250` on 2026-04-26 (see [RETEST_AFTER_PR7.md](/tmp/e2e_reports/RETEST_AFTER_PR7.md)). Notes per fix:

¹ Fix 4 verified by direct entrypoint inspection: `run-entrypoint.sh` and `preprocess-entrypoint.sh` both contain the 10× `getent hosts storage.googleapis.com` probe (2-second sleep between attempts) before invoking `stageFromMinio`, plus a 3-attempt retry around `stageFromMinio` with exponential backoff (5/10/15 s). The 11 E2E dispatches all succeeded without DNS hiccups; the retry path is wired regardless.

² Fixes 5 and 6 verified end-to-end:
- Fix 5: Test 7 confirmed `{run_hash}` resolves on pod after joshpy emits `--custom-tag run_hash=…` via `BatchRemoteConfig.custom_tags`. The pod entrypoint reads `JOSH_CUSTOM_TAGS` as **newline-delimited** entries (one `key=value` per line) — simpler than the originally-proposed JSON+jq approach and equivalent semantically; joshpy's emission (the `--custom-tag` CLI flag) is unaffected.
- Fix 6: Test 8 confirmed pool-policy dispatch with `replicate_start=2` produced output files at indices 2, 3, 4 — pod entrypoint computes `REPLICATE_INDEX = JOB_COMPLETION_INDEX + JOSH_REPLICATE_OFFSET` and passes `--replicate-index` to the JAR.

Fixes 1, 2, 3 are documented below for historical reference. Fixes 4–6 are the new asks driving this update.

---

## Fix 1: `preprocessBatch` mis-propagates the `<dataFile>` positional argument to the K8s pod ✅ MERGED

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

## Fix 2: `MinimalEngineBridge.getExternal` hardcodes `.jshd` extension — can't reach `.jshdz` files ✅ MERGED

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

## Fix 3: Pod run-entrypoint missing `cd $WORK_DIR` before `java -jar` ✅ MERGED

### Symptom

K8s pod completes `stageFromMinio` (downloads files into `/tmp/work/`), then the simulation fails resolving `external "soil_quality"` with a `FileNotFoundException` for the basename — even though the file is sitting in `/tmp/work/` as expected.

### Root cause

`cloud-img/run-entrypoint.sh` invoked the JAR without changing directory:

```sh
WORK_DIR="/tmp/work"
java -jar "$JAR" stageFromMinio --output-dir="$WORK_DIR"
SCRIPT=$(find "$WORK_DIR" -name '*.josh' -type f | head -1)
java -jar "$JAR" run "$SCRIPT" "$JOSH_SIMULATION" --replicate-index="$REPLICATE_INDEX"
```

The JAR's CWD is the Dockerfile `WORKDIR` (`/app`), so `JvmWorkingDirInputGetter.loadFromWorkingDir(name)` opens `/app/<name>` instead of `/tmp/work/<name>`.

### Fix landed

One-liner: `cd "$WORK_DIR"` immediately before the `run` invocation. Mirrored in `preprocess-entrypoint.sh` for parity.

---

## Fix 4: Pod entrypoint DNS readiness + `stageFromMinio` retry

### Symptom

Roughly 1-in-10 K8s jobs on GKE Autopilot's spot pool fail with:

```
stageFromMinio failed: storage.googleapis.com: Temporary failure in name resolution
```

The pod is scheduled, the image pulled, the container started — and the very first network call from the JAR hits a DNS resolver that isn't fully configured yet. The whole pod terminates because the entrypoint doesn't retry.

Combined with spot-node cold-start scheduling delays (30-120s) and image pull (~35s for a 200MB image), this often exhausts a tight `timeoutSeconds` (e.g., 300s) before the pod can recover. Net effect: failed sweeps that are entirely retry-able.

### Root cause

[`cloud-img/run-entrypoint.sh`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/run-entrypoint.sh) makes a single `stageFromMinio` call with no readiness probe and no retry:

```sh
java -jar "$JAR" stageFromMinio \
  --prefix="$JOSH_MINIO_PREFIX" \
  --output-dir="$WORK_DIR"
```

If DNS fails on the first try (which it occasionally does within the first ~5s of pod start on Autopilot), the script exits non-zero and K8s marks the pod failed.

This is a well-known GKE Autopilot pattern. The canonical fix in production-grade images is to (a) probe DNS readiness before any network call, and (b) retry transient network operations a few times with backoff.

### Proposed fix

Update [`cloud-img/run-entrypoint.sh`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/run-entrypoint.sh) (and apply the same pattern to `preprocess-entrypoint.sh`):

```sh
#!/bin/sh
set -e

JAR="${1:-/app/joshsim-fat.jar}"
WORK_DIR="/tmp/work"

# Wait for DNS resolver to be usable. Cheap (<50ms) on the happy path.
# Most Autopilot DNS hiccups clear within 2-4s of container start.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if getent hosts storage.googleapis.com >/dev/null 2>&1; then
    break
  fi
  echo "DNS not ready (attempt $attempt), waiting..."
  sleep 2
done

# Retry stageFromMinio for transient network failures (3 tries, 5/10/15s backoff).
stage_attempts=0
until java -jar "$JAR" stageFromMinio \
        --prefix="$JOSH_MINIO_PREFIX" \
        --output-dir="$WORK_DIR"; do
  stage_attempts=$((stage_attempts + 1))
  if [ "$stage_attempts" -ge 3 ]; then
    echo "ERROR: stageFromMinio failed after 3 attempts" >&2
    exit 1
  fi
  echo "stageFromMinio attempt $stage_attempts failed, retrying..."
  sleep $((stage_attempts * 5))
done

SCRIPT=$(find "$WORK_DIR" -name '*.josh' -type f | head -1)
if [ -z "$SCRIPT" ]; then
  echo "ERROR: No .josh file found in $WORK_DIR" >&2
  exit 1
fi

REPLICATE_INDEX="${JOB_COMPLETION_INDEX:-0}"

cd "$WORK_DIR"
java -XX:+ExitOnOutOfMemoryError -jar "$JAR" run "$SCRIPT" "$JOSH_SIMULATION" \
  --replicate-index="$REPLICATE_INDEX"
```

`getent hosts` is in `glibc` (already in the eclipse-temurin base image). No new package needed. If the team prefers `nslookup`, that's also fine — just install `dnsutils` in the Dockerfile. `getent` is more portable.

### Test plan

- Existing K8s integration tests should still pass (happy path unchanged).
- New: validate against a forced-flake by introducing a brief `iptables` block on egress within the first second of container start (or by running on a known-flaky cluster window). The retry should kick in and recover.
- Validate the "DNS never recovers" failure mode still fails fast (within ~20s) so we don't leave a pod hanging on `timeoutSeconds`.

### Risk

**LOW.** Pure shell additions. Worst case is small added latency on healthy pods (~50ms for DNS probe). No JVM-level changes.

---

## Fix 5: `batchRemote` accepts `--custom-tag` and propagates to pods

### Why this matters

`run` and `runRemote` accept `--custom-tag k=v` flags, which are resolvable as template variables in `exportFiles.<type>` paths inside the simulation. `batchRemote` dropped this in the post-#423 refactor, breaking parity. Practical consequence: a `.josh` file that references `{run_hash}` (or any other custom tag) in its export path works locally and on Cloud Run-direct, but cannot work on the K8s batch path.

This is the principled way for joshpy to inject `run_hash` into export paths — same mechanism josh already uses for parameter values, labels, and user-defined tags. Without it, joshpy has to resort to pre-interpolating tokens in `sim.josh` before upload, which breaks the "same .josh file works in any mode" invariant.

### Proposed change

Three coordinated edits across the dispatch chain:

#### (a) CLI surface

[`src/main/java/org/joshsim/command/BatchRemoteCommand.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/BatchRemoteCommand.java) — re-add the `--custom-tag` option using the existing picocli pattern from `RunCommand`:

```java
@Option(
    names = {"--custom-tag"},
    description = "Custom tag for template resolution (key=value). Repeatable.",
    paramLabel = "<key=value>"
)
private Map<String, String> customTags = new LinkedHashMap<>();
```

Pass `customTags` into the dispatch object alongside the other `BatchRemoteParams` fields.

#### (b) K8s target propagation

[`src/main/java/org/joshsim/pipeline/target/KubernetesBatchTarget.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/pipeline/target/KubernetesBatchTarget.java) — serialize the tag map as a single env var on the K8s Job spec. JSON keeps it parseable and dodges name-mangling concerns:

```java
if (!customTags.isEmpty()) {
    String json = new ObjectMapper().writeValueAsString(customTags);
    envVars.add(plainEnvVar("JOSH_CUSTOM_TAGS", json));
}
```

(or whichever JSON lib the codebase uses; doesn't need to be Jackson specifically.)

#### (c) HTTP target propagation

[`src/main/java/org/joshsim/pipeline/target/HttpBatchTarget.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/pipeline/target/HttpBatchTarget.java) — include in the form POST. Same shape as `KubernetesBatchTarget`. Server-side handler (`JoshSimBatchHandler`) reads it back and appends `--custom-tag` flags when invoking the inner `run`.

#### (d) Pod entrypoint passthrough

[`cloud-img/run-entrypoint.sh`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/run-entrypoint.sh) — parse `JOSH_CUSTOM_TAGS` and append flags. Requires `jq` in the image:

```sh
TAGS=""
if [ -n "$JOSH_CUSTOM_TAGS" ]; then
  TAGS=$(echo "$JOSH_CUSTOM_TAGS" \
    | jq -r 'to_entries | map("--custom-tag " + .key + "=" + (.value|tostring)) | .[]')
fi

cd "$WORK_DIR"
# shellcheck disable=SC2086
java -XX:+ExitOnOutOfMemoryError -jar "$JAR" run "$SCRIPT" "$JOSH_SIMULATION" \
  --replicate-index="$REPLICATE_INDEX" $TAGS
```

If adding `jq` to the base image is undesirable, an alternative is one env var per tag (`JOSH_CUSTOM_TAG_<key>=<value>`) and a `for` loop in shell. Slightly more brittle on key naming (tag keys with special characters need escaping); JSON is cleaner.

### Test plan

- Unit: `BatchRemoteCommand` parses `--custom-tag a=1 --custom-tag b=2` into a map.
- Unit: `KubernetesBatchTarget` builds Job spec containing `JOSH_CUSTOM_TAGS` env var with the JSON-encoded map.
- Unit: `HttpBatchTarget` includes `customTags` field in the POST body.
- E2E (K8s): dispatch with `--custom-tag run_hash=abc123`, sim's export path uses `{run_hash}`, output CSV lands at `.../abc123/...`.
- E2E (Cloud Run): same as above through `JoshSimBatchHandler`.

### Risk

**LOW-MEDIUM.** Touches three modules + the entrypoint, but each change is small and the patterns are already established by `RunCommand`. The biggest unknown is the JSON-decoding step in shell — worth sanity-testing on values containing spaces or quotes (joshpy's typical tags are simple alphanumeric, but the team should pick the encoding it's comfortable with).

After this lands, joshpy will auto-inject `run_hash` (and any user-supplied tags) at `BatchRemoteConfig` build time. See [PR7_PLAN.md](PR7_PLAN.md) item 4.

---

## Fix 6: `run` accepts `--replicate-start=K`; entrypoint adds `JOSH_REPLICATE_OFFSET`

### Why this matters

joshpy wants to support pool/resume semantics for sweeps with stable run hashes:

- "Dispatched 0–4 yesterday, want 10 total" → dispatch 5–9 only.
- "5 of 10 replicates failed on a flake" → re-dispatch only the missing 5.

Today `run --replicates=N` always means "do indices 0..N-1". K8s indexed-Job parallelism uses `JOB_COMPLETION_INDEX` directly as the replicate index. To pool, we need a way to offset the index range.

### Proposed change

#### (a) `run` CLI

[`src/main/java/org/joshsim/command/RunCommand.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/RunCommand.java) — add an option:

```java
@Option(
    names = {"--replicate-start"},
    description = "Starting replicate index (default: 0). Combined with --replicates "
                + "this selects the half-open range [start, start+count).",
    defaultValue = "0"
)
private int replicateStart = 0;
```

Plumb it into the existing replicate loop: instead of `for rep in 0..replicates-1`, use `for rep in replicateStart..replicateStart + replicates - 1`. The `{replicate}` template resolution uses the actual rep number, so output paths are correct without further changes.

#### (b) K8s pod entrypoint

[`cloud-img/run-entrypoint.sh`](https://github.com/SchmidtDSE/josh/blob/main/cloud-img/run-entrypoint.sh) — add the env var with default 0 so existing dispatches behave unchanged:

```sh
REPLICATE_INDEX=$((${JOB_COMPLETION_INDEX:-0} + ${JOSH_REPLICATE_OFFSET:-0}))

cd "$WORK_DIR"
java ... --replicate-index="$REPLICATE_INDEX" ...
```

#### (c) `batchRemote` CLI passes through

[`BatchRemoteCommand.java`](https://github.com/SchmidtDSE/josh/blob/main/src/main/java/org/joshsim/command/BatchRemoteCommand.java) — accept `--replicate-start=K` and propagate to the K8s target via `JOSH_REPLICATE_OFFSET` env var (and to the HTTP target via a form field that the server-side handler forwards as `--replicate-start` when invoking the inner `run`).

The `--replicates=N` semantics stay unchanged — it remains the *count* of replicates to run. With `--replicate-start=K --replicates=N`, the K8s Job dispatches `N` pods covering indices `[K, K+N)`.

### Test plan

- Unit: `run --replicate-start=5 --replicates=3` produces output for replicates 5, 6, 7.
- Unit: K8s Job spec built with offset=5 and parallelism=3 has each pod resolve to its assigned absolute index when summed with `JOB_COMPLETION_INDEX`.
- E2E: stage and dispatch with offset, verify CSVs land at indices 5, 6, 7 (not 0, 1, 2).

### Risk

**LOW.** Pure additive: default `replicate_start=0` preserves all current behavior. No change for callers that don't pass the flag.

After this lands, joshpy can implement the pre-dispatch MinIO listing + replicate offset computation (collision policy `pool`/`replace`/`skip`/`fail`). See [PR7_PLAN.md](PR7_PLAN.md) item 5.

---

## Summary for josh triage

| Fix | File(s) | Status | Approx LoC | Risk |
|-----|---------|--------|------------|------|
| 1. preprocess absolute-path propagation | `command/PreprocessBatchCommand.java` | ✅ Merged | ~15 | LOW |
| 2. `.jshdz` resolution via `external` | `lang/bridge/MinimalEngineBridge.java` + `precompute/MultiFormatExternalGetter.java` | ✅ Merged | ~15-25 | LOW |
| 3. Pod entrypoint `cd $WORK_DIR` | `cloud-img/run-entrypoint.sh`, `cloud-img/preprocess-entrypoint.sh` | ✅ Merged | ~2 | LOW |
| 4. DNS readiness + `stageFromMinio` retry | `cloud-img/run-entrypoint.sh`, `cloud-img/preprocess-entrypoint.sh` | ⬜ TODO | ~15 | LOW |
| 5. `batchRemote --custom-tag` passthrough | `BatchRemoteCommand.java` + `KubernetesBatchTarget.java` + `HttpBatchTarget.java` + `JoshSimBatchHandler.java` + entrypoints | ⬜ TODO | ~50 | LOW-MED |
| 6. `--replicate-start` + `JOSH_REPLICATE_OFFSET` | `RunCommand.java` + `BatchRemoteCommand.java` + entrypoint | ⬜ TODO | ~25 | LOW |

Fixes 4–6 are independent and can land in any order. After they merge and the JAR is rebuilt, joshpy completes PR7 (auto-inject `run_hash` as custom-tag, MinIO-listing + replicate-offset dispatch, pre-dispatch collision check). See [PR7_PLAN.md](PR7_PLAN.md).

## Artifacts referenced

- joshpy E2E reports: `/tmp/e2e_reports/`
- Failing joshpy tests at the time of original investigation: `/tmp/test4_jshdz.py`, `/tmp/test5_preprocess_async.py`, `/tmp/test5b_blocking.py`
- joshpy branch: `feat/batch-run` (PRs [#36](https://github.com/SchmidtDSE/joshpy/pull/36), [#37](https://github.com/SchmidtDSE/joshpy/pull/37), [#38](https://github.com/SchmidtDSE/joshpy/pull/38) merged; [#39](https://github.com/SchmidtDSE/joshpy/pull/39) open)
