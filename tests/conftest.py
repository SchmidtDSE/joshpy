"""Shared fixtures and configuration for joshpy tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pytest marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (MinIO)",
    )


# ---------------------------------------------------------------------------
# MinIO integration test constants (bitnami test image defaults)
# ---------------------------------------------------------------------------

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
TEST_BUCKET = "josh-test-bucket"


# ---------------------------------------------------------------------------
# Session-scoped guards — skip the entire suite when infra is missing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def minio_available():
    """Skip if the MinIO test container is not reachable."""
    import requests

    try:
        resp = requests.get(
            f"http://{MINIO_ENDPOINT}/minio/health/ready", timeout=3
        )
        if resp.status_code != 200:
            pytest.skip(f"MinIO not ready (HTTP {resp.status_code})")
    except requests.ConnectionError:
        pytest.skip("MinIO not available at localhost:9000")


@pytest.fixture(scope="session")
def jar_available():
    """Skip if the Josh JAR has not been downloaded."""
    from joshpy.jar import JarManager, JarMode

    manager = JarManager()
    try:
        manager.get_jar(JarMode.DEV, auto_download=False)
    except FileNotFoundError:
        pytest.skip(
            "Josh JAR not found — run `pixi run get-jars` first"
        )


# ---------------------------------------------------------------------------
# Bucket name
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_bucket():
    return TEST_BUCKET


# ---------------------------------------------------------------------------
# DuckDB connection with S3 configured for the test MinIO
# ---------------------------------------------------------------------------


@pytest.fixture
def minio_conn(minio_available):
    """Fresh DuckDB connection with httpfs configured for test MinIO."""
    import duckdb
    from joshpy.registry import configure_s3

    conn = duckdb.connect(":memory:")
    configure_s3(
        conn,
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        use_ssl=False,
    )
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# RunRegistry with S3 pre-configured
# ---------------------------------------------------------------------------


@pytest.fixture
def minio_registry(minio_available):
    """In-memory RunRegistry whose DuckDB connection can read S3."""
    from joshpy.registry import RunRegistry, configure_s3

    registry = RunRegistry(":memory:")
    configure_s3(
        registry.conn,
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        use_ssl=False,
    )
    yield registry
    registry.close()


# ---------------------------------------------------------------------------
# CSV seeding helper — writes to MinIO via DuckDB COPY
# ---------------------------------------------------------------------------


@pytest.fixture
def seed_csv(minio_conn, test_bucket):
    """Return a callable that writes CSV content to MinIO.

    Usage::

        url = seed_csv("level1/test.csv", "step,replicate,val\\n0,0,1.0\\n")
    """
    cleanup: list[str] = []

    def _seed(key: str, csv_content: str) -> str:
        s3_url = f"s3://{test_bucket}/{key}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            local_path = f.name
        try:
            minio_conn.execute(
                f"COPY (SELECT * FROM read_csv_auto('{local_path}')) "
                f"TO '{s3_url}' (FORMAT CSV, HEADER)"
            )
        finally:
            Path(local_path).unlink(missing_ok=True)
        cleanup.append(s3_url)
        return s3_url

    yield _seed


# ---------------------------------------------------------------------------
# Real JoshCLI (session-scoped — JAR doesn't change)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def josh_cli(jar_available):
    """JoshCLI backed by the real downloaded JAR."""
    from joshpy.cli import JoshCLI
    from joshpy.jar import JarMode

    return JoshCLI(josh_jar=JarMode.DEV)


# ---------------------------------------------------------------------------
# Monkeypatch for configure_s3 → use_ssl=False
# (needed by ingest_results which calls configure_s3 without use_ssl kwarg)
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_s3_no_ssl(monkeypatch):
    """Patch configure_s3 in the sweep module so it uses use_ssl=False."""
    from joshpy.registry import configure_s3 as real_configure_s3

    def _no_ssl(conn, endpoint, access_key, secret_key, **kwargs):
        real_configure_s3(
            conn, endpoint, access_key, secret_key, use_ssl=False
        )

    monkeypatch.setattr("joshpy.sweep.configure_s3", _no_ssl)
