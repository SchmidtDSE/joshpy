"""End-to-end integration tests for bucket-resident remote aggregation.

Exercises REGISTRY_PROVENANCE.md §8 (``get_output_uris`` / ``query_remote`` /
``check_remote_consistency``) against a **real** S3-compatible bucket, driven by
the repo's ``.env`` (``MINIO_ENDPOINT`` / ``MINIO_ACCESS_KEY`` /
``MINIO_SECRET_KEY`` / ``MINIO_BUCKET``). A tiny Josh model is run with the real
JAR, exporting CSVs straight to the bucket; the registry then aggregates them
**without ingesting** anything into ``cell_data``.

Unlike ``test_minio_integration.py`` (which targets a local MinIO at
``localhost:9000`` via ``conftest``), these read the real remote credentials from
``.env`` and skip cleanly when they're absent. All objects are written under an
isolated ``joshpy-phase3-test/<uuid>/`` prefix and deleted on teardown via a
stdlib SigV4 ``DELETE`` (no boto3 dependency).

Run with: pixi run -e dev test-integration tests/test_remote_aggregation_integration.py
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from joshpy.jobs import JobConfig
from joshpy.registry import RunRegistry
from joshpy.sweep import SweepManager

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# .env credential loading (independent of conftest's localhost MinIO)
# ---------------------------------------------------------------------------

def _load_remote_creds() -> dict[str, str] | None:
    """Read MINIO_* creds from the process env, falling back to the repo .env."""
    creds: dict[str, str] = {}
    env_path = Path(__file__).resolve().parent.parent / ".env"
    file_vals: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            file_vals[key.strip()] = val.strip().strip('"').strip("'")
    for key in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET"):
        val = os.environ.get(key) or file_vals.get(key)
        if not val:
            return None
        creds[key] = val
    return creds


REMOTE_CREDS = _load_remote_creds()

requires_remote = pytest.mark.skipif(
    REMOTE_CREDS is None,
    reason="remote bucket creds (MINIO_* in .env or env) not available",
)


def _s3_delete(creds: dict[str, str], key: str, region: str = "auto") -> None:
    """Best-effort SigV4 DELETE of one object (stdlib only; no boto3)."""
    endpoint = creds["MINIO_ENDPOINT"]
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    access_key, secret_key = creds["MINIO_ACCESS_KEY"], creds["MINIO_SECRET_KEY"]
    canonical_uri = f"/{creds['MINIO_BUCKET']}/{key}"
    now = datetime.datetime.now(datetime.timezone.utc)
    amzdate, datestamp = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amzdate}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = (
        f"DELETE\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amzdate}\n{scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def _sign(k: bytes, m: str) -> bytes:
        return hmac.new(k, m.encode(), hashlib.sha256).digest()

    kdate = _sign(("AWS4" + secret_key).encode(), datestamp)
    ksign = _sign(_sign(_sign(kdate, region), "s3"), "aws4_request")
    sig = hmac.new(ksign, string_to_sign.encode(), hashlib.sha256).hexdigest()
    auth = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={sig}"
    )
    req = urllib.request.Request(
        f"https://{host}{canonical_uri}", method="DELETE",
        headers={
            "Host": host, "x-amz-date": amzdate,
            "x-amz-content-sha256": payload_hash, "Authorization": auth,
        },
    )
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except urllib.error.HTTPError:
        pass  # best-effort cleanup


_MODEL = """start simulation Main
  grid.size = 1000 m
  grid.low = 33.7 degrees latitude, -115.4 degrees longitude
  grid.high = 34.0 degrees latitude, -116.4 degrees longitude
  grid.patch = "Default"
  steps.low = 0 count
  steps.high = 3 count
  exportFiles.patch = "minio://{bucket}/{prefix}/output_{{replicate}}.csv"
end simulation
start patch Default
  ForeverTree.init = create 5 count of ForeverTree
  export.treeCount.step = count(ForeverTree)
  export.averageHeight.step = mean(ForeverTree.height)
end patch
start organism ForeverTree
  age.init = 0 year
  age.step = prior.age + 1 year
  height.init = 0 meters
  height.step = prior.height + sample uniform from 0 meters to 1 meters
end organism
start unit year
  alias years
end unit
"""


@requires_remote
class TestRemoteAggregation:
    """Real-bucket §8: aggregate exports without ingest, honoring currency."""

    @pytest.fixture
    def remote_run(self, josh_cli, jar_available, tmp_path):
        """Run a 2-replicate sweep exporting to a fresh bucket prefix.

        Yields ``(registry, prefix, replicates)`` and deletes the objects on
        teardown. Sets MINIO_* in the env so the JAR (subprocess) can write.
        """
        creds = REMOTE_CREDS
        assert creds is not None
        prefix = f"joshpy-phase3-test/{uuid.uuid4().hex[:10]}"
        replicates = 2

        env_backup = {}
        for k in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"):
            env_backup[k] = os.environ.get(k)
            os.environ[k] = creds[k]

        model = _MODEL.format(bucket=creds["MINIO_BUCKET"], prefix=prefix)
        model_path = tmp_path / "remote_model.josh"
        model_path.write_text(model)
        reg_path = str(tmp_path / "remote.duckdb")

        cfg = JobConfig(source_path=model_path, simulation="Main", replicates=replicates)
        manager = (
            SweepManager.builder(cfg)
            .with_registry(reg_path, experiment_name="remote")
            .with_cli(josh_cli)
            .with_label("remote")
            .build()
        )
        run_hash = manager.job_set.jobs[0].run_hash
        try:
            result = manager.run(quiet=True)
            assert result.failed == 0, f"sim failed: {result}"
            # Deliberately NO load_results() — remote aggregation is the point.
        finally:
            manager.cleanup()
            manager.close()

        registry = RunRegistry(reg_path)
        try:
            yield registry, prefix, replicates, run_hash, creds
        finally:
            registry.close()
            for rep in range(replicates + 2):  # a couple extra in case of orphans
                _s3_delete(creds, f"{prefix}/output_{rep}.csv")

        # restore env
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_get_output_uris_are_full_remote_uris(self, remote_run):
        registry, prefix, replicates, run_hash, creds = remote_run
        uris = registry.get_output_uris("remote")
        assert len(uris) == replicates
        assert {u.replicate for u in uris} == set(range(replicates))
        bucket = creds["MINIO_BUCKET"]
        for u in uris:
            assert u.run_hash == run_hash and u.label == "remote"
            # Bucket (host) preserved, not lost to a path-only string.
            assert u.uri == f"minio://{bucket}/{prefix}/output_{u.replicate}.csv"

    def test_query_remote_aggregates_without_ingest(self, remote_run):
        registry, prefix, replicates, run_hash, creds = remote_run
        df = registry.query_remote("averageHeight", agg="mean", group_by=["label", "step"])
        # 4 steps (0..3), one row each, only the current label.
        assert list(df["label"]) == ["remote"] * 4
        assert list(df["step"]) == [0, 1, 2, 3]
        assert (df["value"] >= 0).all()
        # Height accumulates over steps.
        assert df["value"].is_monotonic_increasing
        # Nothing was ingested — this is the load_results=False read path.
        assert registry.conn.execute("SELECT COUNT(*) FROM cell_data").fetchone()[0] == 0

    def test_check_remote_consistency_clean(self, remote_run):
        registry, prefix, replicates, run_hash, creds = remote_run
        assert registry.check_remote_consistency() == []

    def test_bad_run_excluded_from_query_and_orphaned(self, remote_run):
        registry, prefix, replicates, run_hash, creds = remote_run
        registry.mark_run(run_hash, "bad", reason="probe")
        # Current manifest is now empty -> nothing to aggregate.
        with pytest.raises(ValueError):
            registry.query_remote("averageHeight")
        # Its still-present files are now orphans (belong to a non-active run).
        issues = registry.check_remote_consistency()
        kinds = {i.kind for i in issues}
        assert "orphan_remote_output" in kinds
        assert all(i.kind != "missing_remote_output" for i in issues)

    def test_missing_output_detected(self, remote_run):
        registry, prefix, replicates, run_hash, creds = remote_run
        # Delete one replicate's object out from under the registry.
        _s3_delete(creds, f"{prefix}/output_0.csv")
        issues = registry.check_remote_consistency()
        kinds = {i.kind for i in issues}
        assert "missing_remote_output" in kinds
        assert "remote_count_mismatch" in kinds
