"""End-to-end integration tests for the run provenance model.

Exercises status / supersession (REGISTRY_PROVENANCE.md Phase 1) against a
**real** Josh JAR run: two sims are executed and their results ingested into a
registry, then the provenance API is verified against the real ``cell_data``
that landed — including that the ``cell_data_current`` view actually filters out
a superseded (and a ``bad``) run's rows.

No joshpy/JAR mocking: real JAR, real disk I/O, real ingestion.

Requires: Josh JAR (``pixi run get-jars``). Does NOT require MinIO.

Run with::

    pixi run -e dev test-integration tests/test_provenance_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from joshpy.jobs import JobConfig
from joshpy.registry import RunRegistry
from joshpy.sweep import SweepManager

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SOURCE_PATH = EXAMPLES / "tutorial_sweep.josh"
BASELINE_CONFIG = EXAMPLES / "configs" / "baseline.jshc"
HIGH_GROWTH_CONFIG = EXAMPLES / "configs" / "high_growth.jshc"


def _run(cli, registry_path, config, *, experiment, label, on_collision=None,
         reason=None):
    """Run a sim through SweepManager, ingest results, return its run_hash."""
    builder = (
        SweepManager.builder(config)
        .with_registry(registry_path, experiment_name=experiment)
        .with_cli(cli)
    )
    if on_collision is not None:
        builder = builder.with_label(label, on_collision=on_collision, reason=reason)
    else:
        builder = builder.with_label(label)
    manager = builder.build()
    run_hash = manager.job_set.jobs[0].run_hash
    try:
        results = manager.run()
        assert results.failed == 0, f"sim failed: {results}"
        manager.load_results()
    finally:
        manager.cleanup()
        manager.close()
    return run_hash


@pytest.mark.skipif(
    not SOURCE_PATH.exists() or not HIGH_GROWTH_CONFIG.exists(),
    reason="tutorial example files not present",
)
class TestSupersessionEndToEnd:
    """Real-JAR supersession over ingested cell_data."""

    def test_supersede_filters_real_cell_data(self, josh_cli, tmp_path):
        registry_path = str(tmp_path / "provenance.duckdb")

        baseline_cfg = JobConfig(
            source_path=SOURCE_PATH,
            config_path=BASELINE_CONFIG,
            simulation="Main",
            replicates=2,
        )
        old_hash = _run(
            josh_cli, registry_path, baseline_cfg,
            experiment="baseline", label="baseline",
        )

        high_cfg = JobConfig(
            source_path=SOURCE_PATH,
            config_path=HIGH_GROWTH_CONFIG,
            simulation="Main",
            replicates=2,
        )
        new_hash = _run(
            josh_cli, registry_path, high_cfg,
            experiment="high_growth", label="baseline",
            on_collision="supersede", reason="doubled maxGrowth",
        )

        assert old_hash != new_hash

        registry = RunRegistry(registry_path)
        try:
            # Label handover + supersession provenance recorded.
            assert registry.resolve_label("baseline") == new_hash
            assert registry.list_labels() == [("baseline", new_hash)]

            old_status = registry.get_run_status(old_hash)
            assert old_status.status == "superseded"
            assert old_status.superseded_by == new_hash
            assert old_status.reason == "doubled maxGrowth"

            # Both runs really landed rows in cell_data.
            raw = registry.conn.execute(
                "SELECT run_hash, COUNT(*) FROM cell_data GROUP BY run_hash"
            ).fetchall()
            counts = {h: n for h, n in raw}
            assert counts.get(old_hash, 0) > 0
            assert counts.get(new_hash, 0) > 0

            # The current view excludes the superseded run's real rows and keeps
            # exactly the active run's rows.
            view = registry.conn.execute(
                "SELECT run_hash, COUNT(*) FROM cell_data_current GROUP BY run_hash"
            ).fetchall()
            view_counts = {h: n for h, n in view}
            assert view_counts == {new_hash: counts[new_hash]}, view_counts

            # Lineage is walkable, current first.
            assert [d.run_hash for d in registry.run_history("baseline")] == [
                new_hash,
                old_hash,
            ]
        finally:
            registry.close()

    def test_mark_bad_excludes_from_current_view(self, josh_cli, tmp_path):
        registry_path = str(tmp_path / "bad.duckdb")

        cfg = JobConfig(
            source_path=SOURCE_PATH,
            config_path=BASELINE_CONFIG,
            simulation="Main",
            replicates=2,
        )
        run_hash = _run(
            josh_cli, registry_path, cfg, experiment="baseline", label="baseline",
        )

        registry = RunRegistry(registry_path)
        try:
            before = registry.conn.execute(
                "SELECT COUNT(*) FROM cell_data_current WHERE run_hash = ?",
                [run_hash],
            ).fetchone()[0]
            assert before > 0

            registry.mark_run(run_hash, "bad", reason="corrupt forcing data")
            assert registry.get_run_status(run_hash).status == "bad"

            after = registry.conn.execute(
                "SELECT COUNT(*) FROM cell_data_current WHERE run_hash = ?",
                [run_hash],
            ).fetchone()[0]
            assert after == 0

            # Raw cell_data is untouched — marking bad hides, never deletes.
            raw = registry.conn.execute(
                "SELECT COUNT(*) FROM cell_data WHERE run_hash = ?", [run_hash]
            ).fetchone()[0]
            assert raw == before
        finally:
            registry.close()


@pytest.mark.skipif(
    not SOURCE_PATH.exists() or not HIGH_GROWTH_CONFIG.exists(),
    reason="tutorial example files not present",
)
class TestAttributeCurrencyEndToEnd:
    """Real-JAR attribute currency + coverage (REGISTRY_PROVENANCE.md §7, §12)."""

    def _two_runs(self, josh_cli, registry_path):
        baseline_cfg = JobConfig(
            source_path=SOURCE_PATH, config_path=BASELINE_CONFIG,
            simulation="Main", replicates=2,
        )
        old_hash = _run(
            josh_cli, registry_path, baseline_cfg,
            experiment="baseline", label="spin",
        )
        high_cfg = JobConfig(
            source_path=SOURCE_PATH, config_path=HIGH_GROWTH_CONFIG,
            simulation="Main", replicates=2,
        )
        new_hash = _run(
            josh_cli, registry_path, high_cfg,
            experiment="high_growth", label="nospin",
        )
        return old_hash, new_hash

    def test_jobconfig_attributes_persist_through_dispatch(self, josh_cli, tmp_path):
        """Attributes declared on the JobConfig land in the registry with no
        post-run tagging step, and drive coverage directly."""
        from joshpy.registry import Requirement, TargetDesign

        registry_path = str(tmp_path / "declared_attrs.duckdb")
        spin_cfg = JobConfig(
            source_path=SOURCE_PATH, config_path=BASELINE_CONFIG,
            simulation="Main", replicates=1,
            attributes={"scenario": "historical", "treatment": "spinup"},
        )
        spin_hash = _run(
            josh_cli, registry_path, spin_cfg, experiment="spin", label="spin",
        )
        nospin_cfg = JobConfig(
            source_path=SOURCE_PATH, config_path=HIGH_GROWTH_CONFIG,
            simulation="Main", replicates=1,
            attributes={"scenario": "historical", "treatment": "no_spinup"},
        )
        nospin_hash = _run(
            josh_cli, registry_path, nospin_cfg, experiment="nospin", label="nospin",
        )

        registry = RunRegistry(registry_path)
        try:
            # Persisted verbatim, no set_attributes() call anywhere above.
            assert registry.get_attributes(spin_hash) == {
                "scenario": "historical", "treatment": "spinup",
            }
            assert registry.get_attributes(nospin_hash) == {
                "scenario": "historical", "treatment": "no_spinup",
            }
            design = TargetDesign(
                "declared",
                [
                    Requirement({"scenario": "historical", "treatment": "spinup"}),
                    Requirement({"scenario": "historical", "treatment": "no_spinup"}),
                ],
            )
            registry.register_design(design)
            assert registry.check_design("declared").complete
        finally:
            registry.close()

    def test_find_by_attribute_honors_currency(self, josh_cli, tmp_path):
        from joshpy.registry import AttributeMatch

        registry_path = str(tmp_path / "attrs.duckdb")
        old_hash, new_hash = self._two_runs(josh_cli, registry_path)

        registry = RunRegistry(registry_path)
        try:
            registry.set_attributes(old_hash, scenario="historical")
            registry.set_attributes(new_hash, scenario="historical")

            # Both match until one is retired.
            assert set(registry.find_by_attribute("scenario", "historical")) == {
                old_hash, new_hash,
            }
            registry.mark_run(old_hash, "bad", reason="wrong forcing data")

            assert registry.find_by_attribute("scenario", "historical") == [new_hash]
            assert set(
                registry.find_by_attribute("scenario", "historical", current_only=False)
            ) == {old_hash, new_hash}
            assert registry.find_current_by_attribute("scenario", "historical") == [
                AttributeMatch(new_hash, "nospin", {"scenario": "historical"}),
            ]
        finally:
            registry.close()

    def test_check_design_over_real_runs(self, josh_cli, tmp_path):
        from joshpy.registry import Requirement, TargetDesign

        registry_path = str(tmp_path / "coverage.duckdb")
        spin_hash, nospin_hash = self._two_runs(josh_cli, registry_path)

        registry = RunRegistry(registry_path)
        try:
            registry.set_attributes(
                spin_hash, scenario="historical", treatment="spinup"
            )
            registry.set_attributes(
                nospin_hash, scenario="historical", treatment="no_spinup"
            )
            design = TargetDesign(
                "jotr",
                [
                    Requirement({"scenario": "historical", "treatment": "spinup"}),
                    Requirement({"scenario": "historical", "treatment": "no_spinup"}),
                ],
            )
            registry.register_design(design)
            assert registry.check_design("jotr").complete

            # Marking one cell's only run bad makes the design incomplete, but the
            # bad run is surfaced so the cell reads "present but bad".
            registry.mark_run(nospin_hash, "bad", reason="corrupt")
            report = registry.check_design("jotr")
            assert not report.complete
            unmet = report.unmet
            assert len(unmet) == 1
            assert unmet[0].attributes["treatment"] == "no_spinup"
            assert unmet[0].non_active_matches == [(nospin_hash, "bad")]
        finally:
            registry.close()


@pytest.mark.skipif(
    not SOURCE_PATH.exists() or not BASELINE_CONFIG.exists(),
    reason="tutorial example files not present",
)
class TestBadRedispatchEndToEnd:
    """A run marked bad is dropped and re-dispatched fresh (REGISTRY_PROVENANCE §11.1)."""

    def test_bad_run_is_redone_on_rerun(self, josh_cli, tmp_path):
        registry_path = str(tmp_path / "redo.duckdb")
        cfg = JobConfig(
            source_path=SOURCE_PATH, config_path=BASELINE_CONFIG,
            simulation="Main", replicates=2,
        )
        first_hash = _run(
            josh_cli, registry_path, cfg, experiment="baseline", label="baseline",
        )

        # Capture the original executions, then condemn the run.
        registry = RunRegistry(registry_path)
        try:
            original_run_ids = {
                r.run_id for r in registry.get_runs_for_hash(first_hash)
            }
            assert original_run_ids
            registry.mark_run(first_hash, "bad", reason="transient failure")
        finally:
            registry.close()

        # Re-running the identical config yields the same hash; the bad run is
        # dropped and re-dispatched, so it comes back active with fresh rows.
        second_hash = _run(
            josh_cli, registry_path, cfg, experiment="baseline", label="baseline",
        )
        assert second_hash == first_hash

        registry = RunRegistry(registry_path)
        try:
            status = registry.get_run_status(first_hash)
            assert status.effective_status == "active", status
            # Fresh executions replaced the dropped ones (no lingering bad rows).
            new_run_ids = {r.run_id for r in registry.get_runs_for_hash(first_hash)}
            assert new_run_ids and new_run_ids.isdisjoint(original_run_ids)
            current_rows = registry.conn.execute(
                "SELECT COUNT(*) FROM cell_data_current WHERE run_hash = ?",
                [first_hash],
            ).fetchone()[0]
            assert current_rows > 0
        finally:
            registry.close()
