"""MinIO integration tests for joshpy.

Escalating levels of integration testing against a real MinIO service:

- Level 1: DuckDB writes CSV to MinIO and reads it back
- Level 2: Josh JAR runs a simulation that exports to MinIO, Python reads it
- Level 3: CellDataLoader.load_csv() ingests JAR output from S3 into registry
- Level 4: End-to-end ingest_results() from MinIO by label
- Level 5: Partial/interrupted sweep recovery from MinIO
- Edge cases: bad creds, missing bucket, namespace isolation

Requires:
    - MinIO running at localhost:9000 (bitnamilegacy/minio with josh-test-bucket:public)
    - Josh JAR downloaded (pixi run get-jars)

Run with: pixi run -e dev test-integration
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import (
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    TEST_BUCKET,
)

# All tests in this file require MinIO
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test CSV data
# ---------------------------------------------------------------------------

SIMPLE_CSV = "step,replicate,position.x,position.y,treeCount,averageHeight\n"


def _make_csv(replicate: int = 0, steps: int = 5, n_patches: int = 1) -> str:
    """Generate a CSV matching Josh export format."""
    lines = [SIMPLE_CSV.rstrip("\n")]
    for step in range(steps):
        for _ in range(n_patches):
            lines.append(
                f"{step},{replicate},0.0,0.0,{10 + step},{5.0 + step * 0.5}"
            )
    return "\n".join(lines) + "\n"


# ===================================================================
# Level 1: DuckDB httpfs writes to and reads from MinIO
# ===================================================================


class TestMinioWrite:
    """Level 1: Prove DuckDB httpfs can write CSV to MinIO."""

    def test_duckdb_copy_csv_to_s3(self, minio_conn, test_bucket):
        """COPY ... TO 's3://...' should succeed without error."""
        key = f"test-level1/{uuid.uuid4().hex[:8]}/write.csv"
        s3_url = f"s3://{test_bucket}/{key}"

        minio_conn.execute(
            f"COPY (SELECT 1 as step, 0 as replicate, 42.0 as val) "
            f"TO '{s3_url}' (FORMAT CSV, HEADER)"
        )

        # Verify by reading back
        result = minio_conn.execute(
            f"SELECT * FROM read_csv_auto('{s3_url}')"
        ).fetchall()
        assert len(result) == 1
        assert result[0] == (1, 0, 42.0)

    def test_write_then_read_roundtrip(self, seed_csv):
        """seed_csv fixture writes CSV, read it back via DuckDB."""
        csv_data = "a,b,c\n1,hello,3.14\n2,world,2.72\n"
        key = f"test-level1/{uuid.uuid4().hex[:8]}/roundtrip.csv"
        s3_url = seed_csv(key, csv_data)

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
        rows = conn.execute(
            f"SELECT * FROM read_csv_auto('{s3_url}')"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][1] == "hello"
        assert rows[1][1] == "world"


# ===================================================================
# Level 2: Josh JAR writes to MinIO, Python reads
# ===================================================================


class TestMinioJarWrite:
    """Level 2: Run a real simulation that exports to MinIO, verify from Python."""

    SCRIPT = Path(__file__).parent / "fixtures" / "minio_export.josh"

    @pytest.fixture(autouse=True, scope="class")
    def _run_simulation(self, request, josh_cli, minio_available, jar_available):
        """Run the test simulation once for the whole class."""
        env_backup = {}
        for k, v in {
            "MINIO_ENDPOINT": f"http://{MINIO_ENDPOINT}",
            "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
        }.items():
            env_backup[k] = os.environ.get(k)
            os.environ[k] = v

        from joshpy.cli import RunConfig

        result = josh_cli.run(
            RunConfig(
                script=self.SCRIPT,
                simulation="Main",
                replicates=2,
                seed=42,
            )
        )

        # Store result on the class for tests to inspect
        request.cls.jar_result = result

        yield

        # Restore env
        for k, orig in env_backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig

    def test_jar_run_succeeds(self):
        """The Josh JAR should complete the simulation without error."""
        assert self.jar_result.success, (
            f"JAR failed (exit {self.jar_result.exit_code}): "
            f"{self.jar_result.stderr}"
        )

    def test_jar_inspect_exports_minio(self, josh_cli):
        """inspect_exports should parse the minio:// export path."""
        from joshpy.cli import InspectExportsConfig

        exports = josh_cli.inspect_exports(
            InspectExportsConfig(script=self.SCRIPT, simulation="Main")
        )
        patch_info = exports.export_files["patch"]
        assert patch_info is not None
        assert patch_info.protocol == "minio"
        assert patch_info.host == TEST_BUCKET
        assert "{replicate}" in patch_info.path

    def test_jar_output_readable_from_s3(self, minio_conn):
        """CSV written by the JAR should be readable via DuckDB S3."""
        s3_url = f"s3://{TEST_BUCKET}/results/output_0.csv"
        rows = minio_conn.execute(
            f"SELECT * FROM read_csv_auto('{s3_url}')"
        ).fetchall()

        assert len(rows) > 0

        # Check expected columns exist
        cols = [
            desc[0]
            for desc in minio_conn.execute(
                f"SELECT * FROM read_csv_auto('{s3_url}') LIMIT 0"
            ).description
        ]
        assert "step" in cols
        assert "replicate" in cols
        assert "treeCount" in cols
        assert "averageHeight" in cols


# ===================================================================
# Level 3: CellDataLoader loads JAR output from S3
# ===================================================================


class TestMinioCellDataLoader:
    """Level 3: CellDataLoader.load_csv with s3:// URL."""

    def _setup_registry_for_load(self, registry):
        """Register a minimal run so load_csv has a valid run_id."""
        from joshpy.jobs import JobConfig

        config = JobConfig(
            source_path=Path("/tmp/sim.josh"),
            simulation="Main",
            replicates=1,
        )
        session_id = registry.create_session(
            config=config, experiment_name="test"
        )
        registry.register_run(
            session_id=session_id,
            run_hash="load_test_hash",
            josh_path="/tmp/sim.josh",
            config_content="test",
            file_mappings=None,
            parameters={},
        )
        run_id = registry.start_run("load_test_hash", session_id=session_id)
        registry.complete_run(run_id, exit_code=0)
        return run_id

    def test_load_csv_from_s3_url(self, minio_registry, seed_csv, test_bucket):
        """load_csv with an s3:// URL should insert rows into cell_data."""
        from joshpy.cell_data import CellDataLoader

        run_id = self._setup_registry_for_load(minio_registry)
        csv_data = _make_csv(replicate=0, steps=3)
        key = f"test-level3/{uuid.uuid4().hex[:8]}/export.csv"
        s3_url = seed_csv(key, csv_data)

        loader = CellDataLoader(minio_registry)
        rows = loader.load_csv(
            csv_path=s3_url,
            run_id=run_id,
            run_hash="load_test_hash",
        )

        assert rows == 3

        # Verify data in registry
        result = minio_registry.conn.execute(
            "SELECT step, replicate, \"treeCount\", \"averageHeight\" "
            "FROM cell_data ORDER BY step"
        ).fetchall()
        assert len(result) == 3
        assert result[0][0] == 0  # step
        assert result[0][1] == 0  # replicate
        assert result[0][2] == 10  # treeCount at step 0

    def test_load_csv_creates_variable_columns(
        self, minio_registry, seed_csv, test_bucket
    ):
        """Variable columns from the S3 CSV should be auto-created."""
        from joshpy.cell_data import CellDataLoader

        run_id = self._setup_registry_for_load(minio_registry)
        csv_data = _make_csv(replicate=0, steps=2)
        key = f"test-level3/{uuid.uuid4().hex[:8]}/vars.csv"
        s3_url = seed_csv(key, csv_data)

        CellDataLoader(minio_registry).load_csv(
            csv_path=s3_url, run_id=run_id, run_hash="load_test_hash"
        )

        var_cols = minio_registry.list_variable_columns()
        assert "treeCount" in var_cols
        assert "averageHeight" in var_cols

    def test_load_csv_s3_nonexistent_key(self, minio_registry):
        """Missing S3 object should raise a recognizable error."""
        from joshpy.cell_data import CellDataLoader

        run_id = self._setup_registry_for_load(minio_registry)
        loader = CellDataLoader(minio_registry)

        with pytest.raises(Exception, match="HTTP|404|NoSuchKey|IOException"):
            loader.load_csv(
                csv_path=f"s3://{TEST_BUCKET}/nonexistent/{uuid.uuid4()}.csv",
                run_id=run_id,
                run_hash="load_test_hash",
            )

    def test_load_csv_s3_missing_required_columns(
        self, minio_registry, seed_csv
    ):
        """CSV without step/replicate should raise ValueError even from S3."""
        from joshpy.cell_data import CellDataLoader

        run_id = self._setup_registry_for_load(minio_registry)
        bad_csv = "a,b,c\n1,2,3\n"
        key = f"test-level3/{uuid.uuid4().hex[:8]}/bad.csv"
        s3_url = seed_csv(key, bad_csv)

        loader = CellDataLoader(minio_registry)
        with pytest.raises(ValueError, match="step.*replicate"):
            loader.load_csv(
                csv_path=s3_url,
                run_id=run_id,
                run_hash="load_test_hash",
            )


# ===================================================================
# Level 4: End-to-end ingest_results() from MinIO
# ===================================================================


def _make_ingest_registry(minio_registry, josh_content, replicates=2):
    """Set up registry metadata for ingest_results() tests.

    Creates session, registers run with josh_content, labels it,
    and creates completed job_runs.  Returns (run_hash, run_id).
    """
    from joshpy.jobs import JobConfig

    run_hash = f"ingest_{uuid.uuid4().hex[:8]}"

    config = JobConfig(
        source_path=Path("/tmp/sim.josh"),
        simulation="Main",
        replicates=replicates,
    )
    session_id = minio_registry.create_session(
        config=config, experiment_name="ingest-test"
    )
    minio_registry.register_run(
        session_id=session_id,
        run_hash=run_hash,
        josh_path="/tmp/sim.josh",
        config_content="test",
        file_mappings=None,
        parameters={},
        josh_content=josh_content,
    )
    minio_registry.label_run(run_hash, f"label-{run_hash}")

    run_id = None
    for _ in range(replicates):
        run_id = minio_registry.start_run(run_hash, session_id=session_id)
        minio_registry.complete_run(run_id, exit_code=0)

    return run_hash, run_id


class TestMinioIngestResults:
    """Level 4: Full ingest_results() reading real CSVs from MinIO."""

    JOSH_CONTENT = (Path(__file__).parent / "fixtures" / "minio_export.josh").read_text()

    def test_ingest_all_replicates(
        self,
        minio_registry,
        seed_csv,
        test_bucket,
        patch_s3_no_ssl,
        monkeypatch,
    ):
        """ingest_results() should load all replicates from S3."""
        from joshpy.cli import ExportFileInfo, ExportPaths
        from joshpy.sweep import ingest_results

        run_hash, _ = _make_ingest_registry(
            minio_registry, self.JOSH_CONTENT, replicates=3
        )
        label = f"label-{run_hash}"

        # Seed 3 replicate CSVs
        prefix = f"test-level4/{run_hash}"
        for rep in range(3):
            csv_data = _make_csv(replicate=rep, steps=4)
            seed_csv(f"{prefix}/output_{rep}.csv", csv_data)

        # Mock CLI — only inspect_exports needs the JAR
        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw=f"minio://{test_bucket}/{prefix}/output_{{replicate}}.csv",
                    protocol="minio",
                    host=test_bucket,
                    path=f"/{prefix}/output_{{replicate}}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )

        monkeypatch.setenv("MINIO_ENDPOINT", MINIO_ENDPOINT)
        monkeypatch.setenv("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY)
        monkeypatch.setenv("MINIO_SECRET_KEY", MINIO_SECRET_KEY)

        rows = ingest_results(mock_cli, minio_registry, label, quiet=True)

        # 3 replicates x 4 steps x 1 patch = 12 rows
        assert rows == 12

        # Verify data is queryable
        result = minio_registry.conn.execute(
            "SELECT DISTINCT replicate FROM cell_data ORDER BY replicate"
        ).fetchall()
        assert [r[0] for r in result] == [0, 1, 2]

    def test_ingest_results_queryable(
        self,
        minio_registry,
        seed_csv,
        test_bucket,
        patch_s3_no_ssl,
        monkeypatch,
    ):
        """After ingest, cell_data should be queryable with aggregates."""
        from joshpy.cli import ExportFileInfo, ExportPaths
        from joshpy.sweep import ingest_results

        run_hash, _ = _make_ingest_registry(
            minio_registry, self.JOSH_CONTENT, replicates=2
        )
        label = f"label-{run_hash}"

        prefix = f"test-level4-query/{run_hash}"
        for rep in range(2):
            seed_csv(f"{prefix}/output_{rep}.csv", _make_csv(replicate=rep, steps=5))

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw=f"minio://{test_bucket}/{prefix}/output_{{replicate}}.csv",
                    protocol="minio",
                    host=test_bucket,
                    path=f"/{prefix}/output_{{replicate}}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )

        monkeypatch.setenv("MINIO_ENDPOINT", MINIO_ENDPOINT)
        monkeypatch.setenv("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY)
        monkeypatch.setenv("MINIO_SECRET_KEY", MINIO_SECRET_KEY)

        ingest_results(mock_cli, minio_registry, label, quiet=True)

        # Aggregate query
        avg = minio_registry.conn.execute(
            'SELECT AVG("treeCount") FROM cell_data WHERE run_hash = ?',
            [run_hash],
        ).fetchone()[0]
        assert avg is not None
        assert avg > 0


# ===================================================================
# Level 5: Partial / interrupted sweep recovery
# ===================================================================


class TestMinioPartialRecovery:
    """Level 5: Graceful recovery when some replicates are missing."""

    JOSH_CONTENT = (Path(__file__).parent / "fixtures" / "minio_export.josh").read_text()

    def _run_ingest(
        self,
        minio_registry,
        seed_csv,
        test_bucket,
        monkeypatch,
        *,
        replicates_registered: int,
        replicates_seeded: list[int],
        steps: int = 3,
    ) -> tuple[int, str]:
        """Helper: set up registry, seed some replicates, call ingest_results."""
        from joshpy.cli import ExportFileInfo, ExportPaths
        from joshpy.sweep import ingest_results

        run_hash, _ = _make_ingest_registry(
            minio_registry, self.JOSH_CONTENT, replicates=replicates_registered
        )
        label = f"label-{run_hash}"

        prefix = f"test-level5/{run_hash}"
        for rep in replicates_seeded:
            seed_csv(
                f"{prefix}/output_{rep}.csv",
                _make_csv(replicate=rep, steps=steps),
            )

        mock_cli = MagicMock()
        mock_cli.inspect_exports.return_value = ExportPaths(
            simulation="Main",
            export_files={
                "patch": ExportFileInfo(
                    raw=f"minio://{test_bucket}/{prefix}/output_{{replicate}}.csv",
                    protocol="minio",
                    host=test_bucket,
                    path=f"/{prefix}/output_{{replicate}}.csv",
                    file_type="csv",
                ),
                "meta": None,
                "entity": None,
            },
            debug_files={
                "organism": None,
                "patch": None,
                "agent": None,
                "disturbance": None,
            },
        )

        monkeypatch.setenv("MINIO_ENDPOINT", MINIO_ENDPOINT)
        monkeypatch.setenv("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY)
        monkeypatch.setenv("MINIO_SECRET_KEY", MINIO_SECRET_KEY)

        rows = ingest_results(mock_cli, minio_registry, label, quiet=True)
        return rows, run_hash

    def test_partial_replicates_graceful(
        self, minio_registry, seed_csv, test_bucket, patch_s3_no_ssl, monkeypatch
    ):
        """Only 2 of 3 replicates exist — should load 2, skip 1, no error."""
        rows, run_hash = self._run_ingest(
            minio_registry,
            seed_csv,
            test_bucket,
            monkeypatch,
            replicates_registered=3,
            replicates_seeded=[0, 2],  # replicate 1 missing
            steps=4,
        )

        # 2 replicates x 4 steps = 8 rows
        assert rows == 8

        # Verify only replicates 0 and 2 present
        reps = minio_registry.conn.execute(
            "SELECT DISTINCT replicate FROM cell_data "
            "WHERE run_hash = ? ORDER BY replicate",
            [run_hash],
        ).fetchall()
        assert [r[0] for r in reps] == [0, 2]

    def test_zero_replicates_available(
        self, minio_registry, seed_csv, test_bucket, patch_s3_no_ssl, monkeypatch
    ):
        """No CSVs in MinIO — should return 0 rows, no exception."""
        rows, _ = self._run_ingest(
            minio_registry,
            seed_csv,
            test_bucket,
            monkeypatch,
            replicates_registered=3,
            replicates_seeded=[],  # nothing written
        )
        assert rows == 0

    def test_single_replicate_of_many(
        self, minio_registry, seed_csv, test_bucket, patch_s3_no_ssl, monkeypatch
    ):
        """1 of 10 replicates available — should load only that one."""
        rows, run_hash = self._run_ingest(
            minio_registry,
            seed_csv,
            test_bucket,
            monkeypatch,
            replicates_registered=10,
            replicates_seeded=[7],
            steps=3,
        )
        assert rows == 3

        reps = minio_registry.conn.execute(
            "SELECT DISTINCT replicate FROM cell_data WHERE run_hash = ?",
            [run_hash],
        ).fetchall()
        assert [r[0] for r in reps] == [7]


# ===================================================================
# Edge cases
# ===================================================================


class TestMinioEdgeCases:
    """Edge cases: bad credentials, missing bucket, namespace isolation."""

    def test_bad_credentials_clear_error(self, minio_available, test_bucket):
        """Wrong credentials should produce an actionable error."""
        import duckdb
        from joshpy.registry import configure_s3

        conn = duckdb.connect(":memory:")
        configure_s3(
            conn,
            endpoint=MINIO_ENDPOINT,
            access_key="WRONG_KEY",
            secret_key="WRONG_SECRET",
            use_ssl=False,
        )

        with pytest.raises(Exception, match="403|AccessDenied|Forbidden|signature"):
            conn.execute(
                f"SELECT * FROM read_csv_auto('s3://{test_bucket}/results/output_0.csv')"
            ).fetchall()

        conn.close()

    def test_nonexistent_bucket_clear_error(self, minio_conn):
        """Reading from a missing bucket should raise a clear error."""
        with pytest.raises(Exception, match="404|NoSuchBucket|NoSuchKey|not found"):
            minio_conn.execute(
                "SELECT * FROM read_csv_auto("
                "'s3://this-bucket-does-not-exist/file.csv')"
            ).fetchall()

    def test_namespace_isolation(
        self, minio_registry, seed_csv, test_bucket
    ):
        """Two run_hashes should not leak data into each other."""
        from joshpy.cell_data import CellDataLoader
        from joshpy.jobs import JobConfig

        config = JobConfig(
            source_path=Path("/tmp/sim.josh"),
            simulation="Main",
            replicates=1,
        )
        session_id = minio_registry.create_session(
            config=config, experiment_name="isolation-test"
        )

        # Register two runs
        for rh in ("hash_AAA", "hash_BBB"):
            minio_registry.register_run(
                session_id=session_id,
                run_hash=rh,
                josh_path="/tmp/sim.josh",
                config_content="test",
                file_mappings=None,
                parameters={},
            )

        run_id_a = minio_registry.start_run("hash_AAA", session_id=session_id)
        minio_registry.complete_run(run_id_a, exit_code=0)
        run_id_b = minio_registry.start_run("hash_BBB", session_id=session_id)
        minio_registry.complete_run(run_id_b, exit_code=0)

        # Seed different CSVs
        prefix = f"test-isolation/{uuid.uuid4().hex[:8]}"
        csv_a = "step,replicate,position.x,position.y,val\n0,0,0.0,0.0,111\n"
        csv_b = "step,replicate,position.x,position.y,val\n0,0,0.0,0.0,999\n"
        url_a = seed_csv(f"{prefix}/a.csv", csv_a)
        url_b = seed_csv(f"{prefix}/b.csv", csv_b)

        loader = CellDataLoader(minio_registry)
        loader.load_csv(csv_path=url_a, run_id=run_id_a, run_hash="hash_AAA")
        loader.load_csv(csv_path=url_b, run_id=run_id_b, run_hash="hash_BBB")

        # Query by hash — should be isolated
        val_a = minio_registry.conn.execute(
            'SELECT val FROM cell_data WHERE run_hash = ?', ["hash_AAA"]
        ).fetchone()[0]
        val_b = minio_registry.conn.execute(
            'SELECT val FROM cell_data WHERE run_hash = ?', ["hash_BBB"]
        ).fetchone()[0]

        assert val_a == 111
        assert val_b == 999
