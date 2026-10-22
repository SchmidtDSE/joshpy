"""Known-truth studies: test an inference setup on simulated data first.

**Experimental.** The API may change while the Inference docs are written
against it.

A known-truth study picks parameter values (the *truth*), simulates
pseudo-observed data there, and checks whether an inference method recovers
them. Do this with any model before fitting field data: it tests whether the
chosen summary statistics pin down the parameters.

You supply what is specific to your model:

- ``model``: a :class:`~joshpy.jobs.JobConfig` binding the model, its config
  template and data, without a sweep;
- ``prior`` and ``truth``, in :class:`~joshpy.strategies.SbiStrategy`'s prior
  format;
- ``observe(registry, run_hash, replicate)``: what a field study would record
  from one run, read from its exports (for example, a survey of fixed sites);
- ``statistics(data)``: candidate summary statistics of observed data;
- ``compare``: the statistics ABC compares.

joshpy supplies the workflow: the truth run and the prior simulations, the
reference table, z-scored statistics and distances, correlations, replicates at
the truth, sbi searches, posterior summaries and leave-one-out coverage. A
:class:`PropertyFilter` asks a different question of the same prior: which
parameters produce a stated behaviour, with no observed data.

Everything runs through :class:`~joshpy.sweep.SweepManager` and is stored in one
:class:`~joshpy.registry.RunRegistry`, keyed by ``run_hash``. Runs already in
the registry are reused, so :meth:`KnownTruthStudy.load` simulates only what is
missing. A study is one experiment family: its runs are recorded under
experiment names prefixed with ``name`` (``{name}_truth``, ``{name}_prior``,
``{name}_replicates``, ``{name}_filter``), and searches reuse them.

Example:
    from joshpy.inference import KnownTruthStudy

    study = KnownTruthStudy(
        name="forevertree",
        model=JobConfig(template_path=..., source_path=..., file_mappings=...),
        prior={"maxNewGrowth": {"uniform": [0.2, 2.0]}, ...},
        truth={"maxNewGrowth": 1.0, ...},
        observe=survey,
        statistics=statistics,
        compare=["mean_height", "height_cv"],
    )
    loaded = study.load(registry, cli)  # the truth run and 200 prior simulations
    result, session_id = loaded.search(registry, cli, "smcabc", {...}, "forevertree_smcabc")
    loaded.posterior(result)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from joshpy.cli import JoshCLI
from joshpy.jobs import ConfigSweepParameter, ExpandedJob, JobConfig, SweepConfig
from joshpy.registry import RunRegistry
from joshpy.strategies import SbiResult, SbiStrategy
from joshpy.sweep import SweepManager

Observe = Callable[[RunRegistry, str, int], Any]
Summaries = Callable[[RunRegistry, str, Any], dict[str, float]]
Measure = Callable[[RunRegistry, str, int], Mapping[str, float]]


def prior_interval(spec: Mapping[str, Any], level: float) -> tuple[float, float]:
    """Central interval of a prior that holds ``level`` of its mass.

    Args:
        spec: One parameter's prior, ``{"uniform": [low, high]}`` or
            ``{"normal": [mean, std]}``.
        level: Probability mass inside the interval, in (0, 1].

    Raises:
        ValueError: If the spec is not one of the supported kinds.
    """
    if len(spec) != 1:
        raise ValueError(f"A prior spec has one kind, got {dict(spec)}")
    ((kind, (a, b)),) = spec.items()
    tail = (1 - level) / 2
    if kind == "uniform":
        return a + tail * (b - a), b - tail * (b - a)
    if kind == "normal":
        dist = NormalDist(a, b)
        return dist.inv_cdf(tail), dist.inv_cdf(1 - tail)
    raise ValueError(f"Unsupported prior kind {kind!r}; use 'uniform' or 'normal'")


def weighted_quantile(values: Any, weights: Any, q: float) -> float:
    """The q-quantile of weighted values (the smallest value with cumulative weight >= q)."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cumulative, q), len(values) - 1)])


def effective_sample_size(weights: Any) -> float:
    """Kish's effective sample size of importance weights."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (weights**2).sum())


@dataclass
class PropertyFilter:
    """A rejection filter on a model property: which parameters produce a behaviour.

    The filter states criteria the model's output must meet, with no observed
    data [Hartig et al. 2011, option (d)]. Each criterion bounds one
    measurement of a replicate. A parameter set passes a criterion when at
    least ``min_share`` of its replicates meet it, and passes the filter when it
    passes every criterion. Prior draws that pass make up the accepted region.
    It is not a posterior, because no data entered it.

    Attributes:
        measure: ``measure(registry, run_hash, replicate)`` returns the
            property's measurements of one replicate, as ``{name: value}``.
        criteria: ``{criterion: (measurement, low, high)}``. A replicate meets
            a criterion if ``low <= value <= high``; ``None`` leaves an end
            open. A NaN measurement meets no criterion.
        replicates: Replicates of each parameter set.
        min_share: Share of replicates that must meet a criterion.

    Example:
        PropertyFilter(
            measure=gain,
            criteria={"fast": ("gain", 0.75, 0.87), "steady": ("gain_sd", None, 0.017)},
        )
    """

    measure: Measure
    criteria: dict[str, tuple[str, float | None, float | None]]
    replicates: int = 10
    min_share: float = 0.9

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("criteria must name at least one criterion")
        for name, (_, low, high) in self.criteria.items():
            if low is None and high is None:
                raise ValueError(f"Criterion {name!r} needs a low or a high bound")
            if low is not None and high is not None and low > high:
                raise ValueError(f"Criterion {name!r} has low {low} above high {high}")
        if self.replicates < 1:
            raise ValueError("replicates must be at least 1")
        if not 0 < self.min_share <= 1:
            raise ValueError(f"min_share must be in (0, 1], got {self.min_share}")

    def measurements(self, registry: RunRegistry, run_hashes: Any) -> pd.DataFrame:
        """One row per run and replicate: ``run_hash``, ``replicate`` and every measurement."""
        return pd.DataFrame(
            [
                {
                    "run_hash": run_hash,
                    "replicate": replicate,
                    **self.measure(registry, run_hash, replicate),
                }
                for run_hash in run_hashes
                for replicate in range(self.replicates)
            ]
        )

    def meets(self, measurements: pd.DataFrame) -> pd.DataFrame:
        """Whether each row of ``measurements`` meets each criterion (one column per criterion)."""
        met = {}
        for name, (measurement, low, high) in self.criteria.items():
            values = measurements[measurement]
            inside = values.notna()
            if low is not None:
                inside &= values >= low
            if high is not None:
                inside &= values <= high
            met[name] = inside
        return pd.DataFrame(met)

    def evaluate(self, registry: RunRegistry, run_hashes: Any) -> pd.DataFrame:
        """Which runs pass the filter; each must hold ``replicates`` loaded replicates.

        Returns:
            One row per run, indexed by ``run_hash``: the share of its
            replicates meeting each criterion, and ``passes``.
        """
        measurements = self.measurements(registry, run_hashes)
        shares = self.meets(measurements).groupby(measurements["run_hash"], sort=False).mean()
        shares["passes"] = (shares[list(self.criteria)] >= self.min_share).all(axis=1)
        return shares


@dataclass
class KnownTruthStudy:
    """The definition of a known-truth study: what to simulate and how to observe it.

    Like a :class:`~joshpy.jobs.JobConfig`, it is a specification. It runs
    nothing until :meth:`load` (or another method) is called with a registry
    and a CLI.

    Attributes:
        name: Prefix of the experiment names the study's runs are recorded under.
        model: The model, config template and data, without a sweep. The study
            adds its own sweep, ``seed`` and ``replicates``.
        prior: Each parameter's prior, as for :class:`~joshpy.strategies.SbiStrategy`.
        truth: The parameter values of the pseudo-observed run.
        observe: ``observe(registry, run_hash, replicate)`` returns what a field
            study would record from that run.
        statistics: ``statistics(data)`` returns the candidate summary
            statistics of ``observe``'s output, as ``{name: value}``.
        compare: The statistics ABC compares, a subset of ``statistics``' keys.
        truth_seed: Josh seed of the pseudo-observed run, so it is reproducible.
        sbi_seed: Seed of sbi's proposals. With the same seed, a search's first
            draws are the prior simulations, so it reuses those runs.
    """

    name: str
    model: JobConfig
    prior: dict[str, dict[str, list[float]]]
    truth: dict[str, float]
    observe: Observe
    statistics: Callable[[Any], dict[str, float]]
    compare: list[str]
    truth_seed: int = 1
    sbi_seed: int = 0

    def __post_init__(self) -> None:
        if self.model.sweep is not None:
            raise ValueError("model must not have a sweep; the study adds its own")
        if set(self.truth) != set(self.prior):
            raise ValueError(
                f"truth and prior must name the same parameters: "
                f"{sorted(self.truth)} vs {sorted(self.prior)}"
            )
        for spec in self.prior.values():
            prior_interval(spec, 1.0)
        if not self.compare:
            raise ValueError("compare must name at least one statistic")

    @property
    def parameters(self) -> list[str]:
        """Parameter names, in prior order."""
        return list(self.prior)

    def run_at(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        params: Mapping[str, float],
        experiment: str,
        **job: Any,
    ) -> ExpandedJob:
        """Run the model once at fixed parameters, unless the registry already holds that run.

        Args:
            registry: Where the run is recorded and looked up.
            cli: How the model runs.
            params: A value for every parameter.
            experiment: Experiment name of the session.
            **job: Other :class:`~joshpy.jobs.JobConfig` fields, such as
                ``seed`` and ``replicates``.

        Returns:
            The expanded job; its ``run_hash`` keys the results in the registry.
        """
        sweep = SweepConfig(
            config_parameters=[
                ConfigSweepParameter(name=name, values=[value]) for name, value in params.items()
            ]
        )
        manager = (
            SweepManager.builder(replace(self.model, sweep=sweep, **job))
            .with_registry(registry, experiment_name=experiment)
            .with_cli(cli)
            .build()
        )
        try:
            expanded = manager.job_set.jobs[0]
            if len(registry.loaded_replicates(expanded.run_hash)) < expanded.replicates:
                manager.run(quiet=True)
                manager.load_results(quiet=True)
            return expanded
        finally:
            manager.cleanup()

    def summaries(
        self, registry: RunRegistry, run_hash: str, replicate: int = 0
    ) -> dict[str, float]:
        """Every candidate statistic of one run's observed data."""
        return dict(self.statistics(self.observe(registry, run_hash, replicate)))

    def search(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        summaries: Summaries,
        observed: Mapping[str, float],
        method: str,
        method_kwargs: Mapping[str, Any],
        experiment: str,
    ) -> tuple[SbiResult, str]:
        """Run sbi with joshpy as the simulator; returns ``(SbiResult, session_id)``.

        Rerunning with the same arguments against the same registry replays the
        search, reusing every completed run. Failed runs are recorded and
        rejected instead of stopping the search.

        Args:
            registry: Where runs are recorded and reused from.
            cli: How the model runs.
            summaries: ``summaries(registry, run_hash, job)`` for
                :class:`~joshpy.strategies.SbiStrategy`.
            observed: S_obs, in the same units as ``summaries``.
            method: ``"smcabc"`` or ``"mcabc"``.
            method_kwargs: Passed through to sbi.
            experiment: Experiment name of the session.
        """
        strategy = SbiStrategy(
            prior=self.prior,
            summaries=summaries,
            observed=dict(observed),
            method=method,
            method_kwargs=dict(method_kwargs),
            distance="l2",
            seed=self.sbi_seed,
        )
        manager = (
            SweepManager.builder(replace(self.model, sweep=SweepConfig(strategy=strategy)))
            .with_registry(registry, experiment_name=experiment)
            .with_cli(cli)
            .build()
        )
        try:
            result = manager.run(quiet=True, stop_on_failure=False)
        finally:
            manager.cleanup()
        if not isinstance(result, SbiResult):  # SweepManager.run is typed for every strategy
            raise TypeError(f"Expected an SbiResult, got {type(result).__name__}")
        return result, manager.session_id

    def load(self, registry: RunRegistry, cli: JoshCLI, n_prior: int = 200) -> LoadedStudy:
        """The truth run and ``n_prior`` prior simulations, simulating only what is missing.

        The prior simulations are sbi's rejection ABC keeping every draw
        (``quantile=1.0``), which makes it a plain prior simulation.

        Raises:
            RuntimeError: If any prior simulation failed.
        """
        truth_job = self.run_at(
            registry, cli, self.truth, f"{self.name}_truth", seed=self.truth_seed
        )
        truth_data = self.observe(registry, truth_job.run_hash, 0)
        candidates = dict(self.statistics(truth_data))

        def compared(registry: RunRegistry, run_hash: str, job: Any) -> dict[str, float]:
            values = self.summaries(registry, run_hash)
            return {name: values[name] for name in self.compare}

        prior = self._prior_draws(
            registry,
            cli,
            n_prior,
            compared,
            {name: candidates[name] for name in self.compare},
            f"{self.name}_prior",
        )
        reference = pd.DataFrame(
            [
                {
                    "run_hash": job.run_hash,
                    **job.parameters,
                    **self.summaries(registry, job.run_hash),
                }
                for job, _ in prior.job_results
            ]
        )
        return LoadedStudy(self, truth_job, truth_data, reference)

    def filter_prior(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        property_filter: PropertyFilter,
        n: int = 200,
        experiment: str | None = None,
    ) -> pd.DataFrame:
        """Rejection sampling from the prior: which of n prior draws pass a filter.

        The draws are :meth:`load`'s prior simulations (the same ``sbi_seed``),
        so their runs are reused. Each runs with ``property_filter.replicates``
        replicates: a run the registry holds with fewer is rerun, and the
        replicates the registry already holds are kept, not replaced.

        Args:
            registry: Where runs are recorded and reused from.
            cli: How the model runs.
            property_filter: The criteria, and the replicates per draw.
            n: Number of prior draws.
            experiment: Experiment name of the session; defaults to ``{name}_filter``.

        Returns:
            One row per draw: ``run_hash``, the parameters, the share of
            replicates meeting each criterion, and ``passes``.

        Raises:
            RuntimeError: If any simulation failed.
        """

        def no_data(registry: RunRegistry, run_hash: str, job: Any) -> dict[str, float]:
            return {"none": 0.0}  # a filter compares no data; every draw is kept

        study = replace(self, model=replace(self.model, replicates=property_filter.replicates))
        draws = study._prior_draws(
            registry, cli, n, no_data, {"none": 0.0}, experiment or f"{self.name}_filter"
        )
        jobs = [job for job, _ in draws.job_results]
        parameters = pd.DataFrame([{"run_hash": job.run_hash, **job.parameters} for job in jobs])
        return parameters.join(
            property_filter.evaluate(registry, [job.run_hash for job in jobs]), on="run_hash"
        )

    def _prior_draws(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        n: int,
        summaries: Summaries,
        observed: Mapping[str, float],
        experiment: str,
    ) -> SbiResult:
        """n prior draws: sbi's rejection ABC keeping every draw (``quantile=1.0``).

        Every caller gets the same draws (the same ``sbi_seed``), so their runs
        are shared.

        Raises:
            RuntimeError: If any simulation failed.
        """
        draws, _ = self.search(
            registry,
            cli,
            summaries,
            observed,
            "mcabc",
            {"num_simulations": n, "quantile": 1.0},
            experiment,
        )
        if draws.failed:
            raise RuntimeError(f"{draws.failed} prior simulation(s) failed")
        return draws

    def replicates(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        n: int,
        seed: int,
        observe: Observe | None = None,
    ) -> pd.DataFrame:
        """Simulate n replicates at the truth, as one seeded job, and compute their statistics.

        Args:
            registry: Where the job is recorded and reused from.
            cli: How the model runs.
            n: Number of replicates.
            seed: Josh seed of the job; keep it different from ``truth_seed``.
            observe: Observe the replicates differently from ``self.observe``
                (for example, without measurement error).

        Returns:
            One row per replicate: ``replicate`` and every candidate statistic.
        """
        job = self.run_at(
            registry, cli, self.truth, f"{self.name}_replicates", seed=seed, replicates=n
        )
        observe = observe or self.observe
        return pd.DataFrame(
            [
                {
                    "replicate": replicate,
                    **self.statistics(observe(registry, job.run_hash, replicate)),
                }
                for replicate in range(n)
            ]
        )


@dataclass
class LoadedStudy:
    """A known-truth study with its truth run and prior simulations: what a method starts from.

    Statistics are compared as **z-scores**: each statistic minus its mean, over
    its standard deviation, both taken over the prior simulations. That puts
    every statistic on the same scale, so none dominates the Euclidean distance.
    The mean cancels in every difference, so distances are those of statistics
    divided by their standard deviation [Beaumont et al. 2002].

    Attributes:
        study: The study's definition.
        truth_job: The pseudo-observed run.
        truth_data: Its observed data, as ``study.observe`` returned it.
        reference: The ABC reference table: one row per prior simulation, with
            ``run_hash``, the parameters and every candidate statistic.
        center: Mean of each candidate statistic over the prior simulations.
        scale: Standard deviation of each candidate statistic over the prior
            simulations.
    """

    study: KnownTruthStudy
    truth_job: ExpandedJob
    truth_data: Any
    reference: pd.DataFrame
    center: dict[str, float] = field(init=False)
    scale: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        names = list(self.candidates)
        self.center = {name: float(self.reference[name].mean()) for name in names}
        self.scale = {name: float(self.reference[name].std()) for name in names}

    @property
    def candidates(self) -> dict[str, float]:
        """Every candidate statistic of the pseudo-observed data."""
        return dict(self.study.statistics(self.truth_data))

    @property
    def observed(self) -> dict[str, float]:
        """S_obs: the compared statistics of the pseudo-observed data."""
        candidates = self.candidates
        return {name: candidates[name] for name in self.study.compare}

    def zscore(self, values: Any, names: list[str] | None = None) -> Any:
        """z-scores of statistics: a ``{name: value}`` mapping, or a DataFrame's columns.

        Args:
            values: One set of statistics, or a table with a column per statistic.
            names: Statistics to score; defaults to ``study.compare``.
        """
        names = list(names or self.study.compare)
        if isinstance(values, pd.DataFrame):
            return pd.DataFrame(
                {name: (values[name] - self.center[name]) / self.scale[name] for name in names}
            )
        return {
            name: (float(values[name]) - self.center[name]) / self.scale[name] for name in names
        }

    def distance(self, table: pd.DataFrame, names: list[str] | None = None) -> pd.Series:
        """Euclidean distance (sbi's ``"l2"``) from S_obs of each row's z-scored statistics."""
        offsets = self.zscore(table, names) - pd.Series(self.zscore(self.candidates, names))
        return np.sqrt((offsets**2).sum(axis=1))

    def correlations(self, names: list[str] | None = None) -> pd.DataFrame:
        """Spearman correlation of each statistic with each parameter, over the prior simulations.

        Args:
            names: Statistics (rows); defaults to every candidate.
        """
        names = list(names or self.candidates)
        parameters = self.study.parameters
        table = self.reference[parameters + names].corr(method="spearman")
        return table.loc[names, parameters]

    def abc_target(self, names: list[str] | None = None) -> tuple[Summaries, dict[str, float]]:
        """``(summaries, observed)`` for :class:`~joshpy.strategies.SbiStrategy`, as z-scores.

        Args:
            names: Statistics to compare; defaults to ``study.compare``.
        """
        names = list(names or self.study.compare)
        study, center, scale = self.study, dict(self.center), dict(self.scale)

        def zscored_summaries(registry: RunRegistry, run_hash: str, job: Any) -> dict[str, float]:
            values = study.summaries(registry, run_hash)
            return {name: (values[name] - center[name]) / scale[name] for name in names}

        return zscored_summaries, self.zscore(self.candidates, names)

    def search(
        self,
        registry: RunRegistry,
        cli: JoshCLI,
        method: str,
        method_kwargs: Mapping[str, Any],
        experiment: str,
        names: list[str] | None = None,
    ) -> tuple[SbiResult, str]:
        """Search for the truth with sbi on z-scored statistics; returns ``(SbiResult, session_id)``.

        See :meth:`KnownTruthStudy.search`; ``names`` defaults to ``study.compare``.
        """
        summaries, observed = self.abc_target(names)
        return self.study.search(
            registry, cli, summaries, observed, method, method_kwargs, experiment
        )

    def posterior(self, result: SbiResult, level: float = 0.9) -> pd.DataFrame:
        """Weighted posterior mean and central interval of each parameter, against the truth.

        Args:
            result: A search's result; its final population is the posterior sample.
            level: Probability mass of the central interval.

        Returns:
            One row per parameter: ``truth``, ``mean``, the interval's ends,
            ``width`` (the interval's width over the prior's central interval of
            the same level) and ``truth inside``. The interval is marginal.
        """
        particles = result.to_dataframe()
        weights = particles["weight"].to_numpy()
        tail = (1 - level) / 2
        low_label, high_label = f"{tail:.0%}", f"{1 - tail:.0%}"
        rows = {}
        for name in self.study.parameters:
            values = particles[name].to_numpy()
            low, high = (weighted_quantile(values, weights, q) for q in (tail, 1 - tail))
            prior_low, prior_high = prior_interval(self.study.prior[name], level)
            truth = self.study.truth[name]
            rows[name] = {
                "truth": truth,
                "mean": float(np.average(values, weights=weights)),
                low_label: low,
                high_label: high,
                "width": (high - low) / (prior_high - prior_low),
                "truth inside": low <= truth <= high,
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def leave_one_out(self, names: list[str] | None = None, k: int = 20) -> pd.DataFrame:
        """Coverage of rejection ABC on the reference table, holding out each simulation in turn.

        Each prior simulation plays the observed data once: the ``k`` other
        simulations closest to it are accepted, and the held-out parameter value
        is scored as covered if it falls strictly inside the accepted range. If
        the method is calibrated, that happens (k - 1) / (k + 1) of the time.

        Args:
            names: Statistics to compare; defaults to ``study.compare``.
            k: Simulations accepted per held-out one.

        Returns:
            One row per parameter: ``coverage``, and ``width``, the mean accepted
            range over the prior's central interval holding (k - 1) / (k + 1) of
            its mass. Uninformative statistics give a width near 1.
        """
        reference = self.reference
        z = self.zscore(reference, names).to_numpy()
        parameters = self.study.parameters
        level = (k - 1) / (k + 1)
        spans = {}
        for name in parameters:
            low, high = prior_interval(self.study.prior[name], level)
            spans[name] = high - low
        inside: dict[str, list[bool]] = {name: [] for name in parameters}
        width: dict[str, list[float]] = {name: [] for name in parameters}
        for i in range(len(reference)):
            distance = np.sqrt(((z - z[i]) ** 2).sum(axis=1))
            distance[i] = np.inf
            accepted = reference.iloc[np.argsort(distance, kind="stable")[:k]]
            for name in parameters:
                low, high = accepted[name].min(), accepted[name].max()
                inside[name].append(bool(low < reference[name].iloc[i] < high))
                width[name].append((high - low) / spans[name])
        return pd.DataFrame(
            {
                "coverage": {name: float(np.mean(inside[name])) for name in parameters},
                "width": {name: float(np.mean(width[name])) for name in parameters},
            }
        )
