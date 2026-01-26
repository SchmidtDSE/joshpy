"""Job orchestration for parameter sweeps with Jinja templating.

This module enables Python users to programmatically generate .jshc configuration
files using Jinja templating, expand parameter sweeps into concrete job combinations,
and execute them against the Josh runtime.

Example usage:
    from pathlib import Path
    from joshpy.jobs import JobConfig, SweepConfig, SweepParameter, JobExpander, JobRunner

    config = JobConfig(
        template_path=Path("templates/forestsim.jshc.j2"),
        simulation="ForestSim",
        replicates=3,
        sweep=SweepConfig(
            parameters=[
                SweepParameter(name="survivalProbAdult", values=[85, 90, 95]),
                SweepParameter(name="seedPerTree", values=[500, 1000, 2000]),
            ]
        ),
    )

    expander = JobExpander()
    job_set = expander.expand(config)

    runner = JobRunner(josh_jar=Path("joshsim.jar"))
    results = runner.run_all(job_set)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _check_jinja2() -> None:
    """Raise ImportError if jinja2 is not available."""
    if not HAS_JINJA2:
        raise ImportError(
            "jinja2 is required for job templating. "
            "Install with: pip install joshpy[jobs]"
        )


def _check_yaml() -> None:
    """Raise ImportError if pyyaml is not available."""
    if not HAS_YAML:
        raise ImportError(
            "pyyaml is required for YAML config files. "
            "Install with: pip install joshpy[jobs]"
        )


def compute_config_hash(config_content: str) -> str:
    """Compute MD5 hash of config content for unique identification.

    Returns first 12 characters of hex digest for reasonable uniqueness
    while keeping paths manageable.

    Args:
        config_content: The rendered configuration file content.

    Returns:
        12-character hex string hash.
    """
    return hashlib.md5(config_content.encode('utf-8')).hexdigest()[:12]


def _normalize_values(values: Any) -> list[Any]:
    """Normalize parameter values to a list.

    Handles:
    - Plain lists/tuples
    - NumPy arrays
    - Range specification dicts: {"start": 0, "stop": 10, "step": 2}
                              or {"start": 0, "stop": 10, "num": 5}

    Args:
        values: Input values in any supported format.

    Returns:
        A plain Python list of values.
    """
    # NumPy array
    if HAS_NUMPY and isinstance(values, np.ndarray):
        return values.tolist()

    # Range specification dict
    if isinstance(values, dict):
        start = values.get('start', 0)
        stop = values['stop']  # Required

        if 'num' in values:
            # linspace-style
            if not HAS_NUMPY:
                # Fallback without numpy
                num = values['num']
                if num == 1:
                    return [start]
                step = (stop - start) / (num - 1)
                return [start + i * step for i in range(num)]
            return np.linspace(start, stop, values['num']).tolist()
        elif 'step' in values:
            # arange-style
            if not HAS_NUMPY:
                # Fallback without numpy
                result = []
                current = start
                while current < stop:
                    result.append(current)
                    current += values['step']
                return result
            return np.arange(start, stop, values['step']).tolist()
        else:
            raise ValueError(f"Range spec must have 'step' or 'num': {values}")

    # Already a sequence
    if isinstance(values, (list, tuple)):
        # Recursively normalize in case of nested numpy arrays
        return [
            v.tolist() if HAS_NUMPY and isinstance(v, np.ndarray) else v
            for v in values
        ]

    # Single value - wrap in list
    return [values]


@dataclass
class SweepParameter:
    """A parameter to sweep over in the job expansion.

    Accepts values as:
    - Explicit list: [85, 90, 95]
    - NumPy array: np.arange(80, 99, 2)
    - Range specification dict: {"start": 80, "stop": 99, "step": 2}
                             or {"start": 80, "stop": 99, "num": 10}

    Attributes:
        name: Parameter name (used in Jinja template).
        values: List of values to sweep over.
    """

    name: str
    values: list[Any] = field(default_factory=list)

    # Store original range spec for YAML round-tripping
    _range_spec: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalize values after initialization."""
        # Store range spec if provided
        if isinstance(self.values, dict):
            self._range_spec = self.values.copy()

        # Normalize to list
        self.values = _normalize_values(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for YAML/JSON serialization.

        Preserves range specs when possible for cleaner YAML output.
        """
        result: dict[str, Any] = {"name": self.name}
        if self._range_spec:
            result["range"] = self._range_spec
        else:
            result["values"] = self.values
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepParameter:
        """Create from dict (YAML/JSON deserialization)."""
        name = data["name"]
        if "range" in data:
            return cls(name=name, values=data["range"])
        return cls(name=name, values=data.get("values", []))


@dataclass
class SweepConfig:
    """Configuration for parameter sweep expansion.

    Attributes:
        parameters: List of parameters to sweep over.
    """

    parameters: list[SweepParameter] = field(default_factory=list)

    def expand(self) -> list[dict[str, Any]]:
        """Generate cartesian product of all parameter values.

        Returns:
            List of dicts, each representing one parameter combination.
            For example, with parameters A=[1,2] and B=[x,y], returns:
            [{"A": 1, "B": "x"}, {"A": 1, "B": "y"},
             {"A": 2, "B": "x"}, {"A": 2, "B": "y"}]
        """
        if not self.parameters:
            return [{}]

        names = [p.name for p in self.parameters]
        value_lists = [p.values for p in self.parameters]

        combinations = []
        for combo in itertools.product(*value_lists):
            combinations.append(dict(zip(names, combo)))

        return combinations

    def __len__(self) -> int:
        """Return total number of parameter combinations."""
        if not self.parameters:
            return 1
        count = 1
        for p in self.parameters:
            count *= len(p.values)
        return count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "parameters": [p.to_dict() for p in self.parameters]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepConfig:
        """Create from dict."""
        parameters = [
            SweepParameter.from_dict(p)
            for p in data.get("parameters", [])
        ]
        return cls(parameters=parameters)


@dataclass
class JobConfig:
    """Configuration for a job (may be template for expansion).

    Attributes:
        template_path: Path to Jinja template file (.jshc.j2).
        template_string: Jinja template as string (alternative to template_path).
        simulation: Name of simulation to run.
        replicates: Number of replicates per job.
        sweep: Parameter sweep configuration.
        source_path: Path to .josh source file.
        file_mappings: Map of data file names to paths.
        upload_source_path: Template for uploading source files.
        upload_config_path: Template for uploading config files.
        upload_data_path: Template for uploading data files.
        output_steps: Step range to output (e.g., "0-10,50,100").
        seed: Random seed for reproducibility.
        crs: Coordinate reference system.
        use_float64: Use double precision instead of BigDecimal.
    """

    # Template source (one of these required for sweep)
    template_path: Path | None = None
    template_string: str | None = None

    # Core simulation settings
    simulation: str = "Main"
    replicates: int = 1
    source_path: Path | None = None

    # Parameter sweep config (optional)
    sweep: SweepConfig | None = None

    # File mappings (editor.jshc, data.jshd, etc.)
    file_mappings: dict[str, Path] = field(default_factory=dict)

    # Upload path templates
    upload_source_path: str | None = None
    upload_config_path: str | None = None
    upload_data_path: str | None = None

    # Additional CLI options
    output_steps: str | None = None
    seed: int | None = None
    crs: str | None = None
    use_float64: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        result: dict[str, Any] = {}

        if self.template_path:
            result["template_path"] = str(self.template_path)
        if self.template_string:
            result["template_string"] = self.template_string
        if self.simulation != "Main":
            result["simulation"] = self.simulation
        if self.replicates != 1:
            result["replicates"] = self.replicates
        if self.source_path:
            result["source_path"] = str(self.source_path)
        if self.sweep:
            result["sweep"] = self.sweep.to_dict()
        if self.file_mappings:
            result["file_mappings"] = {k: str(v) for k, v in self.file_mappings.items()}
        if self.upload_source_path:
            result["upload_source_path"] = self.upload_source_path
        if self.upload_config_path:
            result["upload_config_path"] = self.upload_config_path
        if self.upload_data_path:
            result["upload_data_path"] = self.upload_data_path
        if self.output_steps:
            result["output_steps"] = self.output_steps
        if self.seed is not None:
            result["seed"] = self.seed
        if self.crs:
            result["crs"] = self.crs
        if self.use_float64:
            result["use_float64"] = self.use_float64

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobConfig:
        """Create from dict."""
        kwargs: dict[str, Any] = {}

        if "template_path" in data:
            kwargs["template_path"] = Path(data["template_path"])
        if "template_string" in data:
            kwargs["template_string"] = data["template_string"]
        if "simulation" in data:
            kwargs["simulation"] = data["simulation"]
        if "replicates" in data:
            kwargs["replicates"] = data["replicates"]
        if "source_path" in data:
            kwargs["source_path"] = Path(data["source_path"])
        if "sweep" in data:
            kwargs["sweep"] = SweepConfig.from_dict(data["sweep"])
        if "file_mappings" in data:
            kwargs["file_mappings"] = {
                k: Path(v) for k, v in data["file_mappings"].items()
            }
        if "upload_source_path" in data:
            kwargs["upload_source_path"] = data["upload_source_path"]
        if "upload_config_path" in data:
            kwargs["upload_config_path"] = data["upload_config_path"]
        if "upload_data_path" in data:
            kwargs["upload_data_path"] = data["upload_data_path"]
        if "output_steps" in data:
            kwargs["output_steps"] = data["output_steps"]
        if "seed" in data:
            kwargs["seed"] = data["seed"]
        if "crs" in data:
            kwargs["crs"] = data["crs"]
        if "use_float64" in data:
            kwargs["use_float64"] = data["use_float64"]

        return cls(**kwargs)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        _check_yaml()
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> JobConfig:
        """Create from YAML string."""
        _check_yaml()
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: Path) -> JobConfig:
        """Load from YAML file."""
        _check_yaml()
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: Path) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            f.write(self.to_yaml())


@dataclass
class ExpandedJob:
    """A fully-expanded job with concrete parameter values.

    Attributes:
        config_content: Rendered .jshc configuration content.
        config_path: Path to written config file.
        config_hash: MD5 hash of config content.
        parameters: Parameter values used for this job.
        simulation: Simulation name.
        replicates: Number of replicates.
        source_path: Path to .josh source file.
        file_mappings: Data file mappings.
        custom_tags: Tags for CLI (derived from parameters).
        upload_source_path: Resolved upload path for source.
        upload_config_path: Resolved upload path for config.
        upload_data_path: Resolved upload path for data.
        output_steps: Step range to output.
        seed: Random seed.
        crs: Coordinate reference system.
        use_float64: Use double precision.
    """

    config_content: str
    config_path: Path
    config_hash: str
    parameters: dict[str, Any]
    simulation: str
    replicates: int
    source_path: Path | None = None
    file_mappings: dict[str, Path] = field(default_factory=dict)
    custom_tags: dict[str, str] = field(default_factory=dict)
    upload_source_path: str | None = None
    upload_config_path: str | None = None
    upload_data_path: str | None = None
    output_steps: str | None = None
    seed: int | None = None
    crs: str | None = None
    use_float64: bool = False


@dataclass
class JobSet:
    """A collection of jobs to execute (from expanding a JobConfig).

    Attributes:
        jobs: List of expanded jobs.
        temp_dir: Temporary directory containing config files.
        template_path: Original template path (if any).
    """

    jobs: list[ExpandedJob] = field(default_factory=list)
    temp_dir: Path | None = None
    template_path: Path | None = None

    def cleanup(self) -> None:
        """Remove temporary config files."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __len__(self) -> int:
        return len(self.jobs)

    def __iter__(self) -> Iterator[ExpandedJob]:
        return iter(self.jobs)


@dataclass
class JobExpander:
    """Expands JobConfig with sweeps into concrete ExpandedJobs.

    Attributes:
        jinja_env: Jinja2 Environment for template rendering.
    """

    jinja_env: Any = field(default=None)  # jinja2.Environment

    def __post_init__(self) -> None:
        """Initialize Jinja environment if not provided."""
        _check_jinja2()
        if self.jinja_env is None:
            self.jinja_env = Environment(autoescape=False)

    def expand(
        self,
        config: JobConfig,
        output_dir: Path | None = None,
        config_name: str = "editor.jshc"
    ) -> JobSet:
        """Expand a JobConfig into a JobSet with one ExpandedJob per parameter combo.

        Args:
            config: The job configuration to expand.
            output_dir: Directory to write configs to (uses temp dir if None).
            config_name: Name for the generated config file.

        Returns:
            JobSet containing all expanded jobs.
        """
        _check_jinja2()

        # Load template
        if config.template_path:
            template_dir = config.template_path.parent
            template_name = config.template_path.name
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=False
            )
            template = env.get_template(template_name)
        elif config.template_string:
            template = self.jinja_env.from_string(config.template_string)
        else:
            raise ValueError("Either template_path or template_string must be provided")

        # Create output directory
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = None
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="josh_sweep_"))
            output_dir = temp_dir

        # Get parameter combinations
        if config.sweep:
            combinations = config.sweep.expand()
        else:
            combinations = [{}]

        # Generate jobs
        jobs: list[ExpandedJob] = []
        for i, params in enumerate(combinations):
            # Render template
            rendered = template.render(**params)

            # Compute hash
            config_hash = compute_config_hash(rendered)

            # Write config file
            config_subdir = output_dir / f"job_{i:04d}_{config_hash}"
            config_subdir.mkdir(parents=True, exist_ok=True)
            config_path = config_subdir / config_name
            config_path.write_text(rendered)

            # Build custom tags from parameters
            custom_tags = {
                str(k): str(v) for k, v in params.items()
            }
            # Add config_hash as a custom tag
            custom_tags["config_hash"] = config_hash

            # Create expanded job
            job = ExpandedJob(
                config_content=rendered,
                config_path=config_path,
                config_hash=config_hash,
                parameters=params,
                simulation=config.simulation,
                replicates=config.replicates,
                source_path=config.source_path,
                file_mappings=config.file_mappings.copy(),
                custom_tags=custom_tags,
                upload_source_path=config.upload_source_path,
                upload_config_path=config.upload_config_path,
                upload_data_path=config.upload_data_path,
                output_steps=config.output_steps,
                seed=config.seed,
                crs=config.crs,
                use_float64=config.use_float64,
            )
            jobs.append(job)

        return JobSet(
            jobs=jobs,
            temp_dir=temp_dir,
            template_path=config.template_path,
        )


@dataclass
class JobResult:
    """Result from executing a single job.

    Attributes:
        job: The executed job.
        exit_code: Process exit code.
        stdout: Standard output.
        stderr: Standard error.
        command: The command that was executed.
    """

    job: ExpandedJob
    exit_code: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Return True if job completed successfully."""
        return self.exit_code == 0


@dataclass
class JobRunner:
    """Executes ExpandedJobs via subprocess (calls Josh CLI).

    Attributes:
        josh_jar: Path to joshsim.jar file.
        java_path: Path to java executable.
        working_dir: Working directory for job execution.
    """

    josh_jar: Path
    java_path: str = "java"
    working_dir: Path | None = None

    def build_command(self, job: ExpandedJob) -> list[str]:
        """Build the CLI command for a job.

        Args:
            job: The job to build command for.

        Returns:
            List of command arguments.
        """
        cmd = [
            self.java_path,
            "-jar",
            str(self.josh_jar),
            "run",
        ]

        # Source file
        if job.source_path:
            cmd.append(str(job.source_path))

        # Simulation name
        cmd.append(job.simulation)

        # Replicates
        if job.replicates > 1:
            cmd.extend(["--replicates", str(job.replicates)])

        # Config file - use --data flag with the config file
        # The config path parent directory is used as working dir
        # so the config file is found by name
        cmd.extend(["--data", f"editor={job.config_path}"])

        # File mappings
        for name, path in job.file_mappings.items():
            cmd.extend(["--data", f"{name}={path}"])

        # Custom tags (for path template resolution)
        for name, value in job.custom_tags.items():
            cmd.extend(["--custom-tag", f"{name}={value}"])

        # Upload paths
        if job.upload_source_path:
            cmd.extend(["--upload-source-path", job.upload_source_path])
        if job.upload_config_path:
            cmd.extend(["--upload-config-path", job.upload_config_path])
        if job.upload_data_path:
            cmd.extend(["--upload-data-path", job.upload_data_path])

        # Other options
        if job.output_steps:
            cmd.extend(["--output-steps", job.output_steps])
        if job.seed is not None:
            cmd.extend(["--seed", str(job.seed)])
        if job.crs:
            cmd.extend(["--crs", job.crs])
        if job.use_float64:
            cmd.append("--use-float-64")

        return cmd

    def run(
        self,
        job: ExpandedJob,
        timeout: float | None = None,
        capture_output: bool = True
    ) -> JobResult:
        """Execute a single job.

        Args:
            job: The job to execute.
            timeout: Timeout in seconds (None for no timeout).
            capture_output: Whether to capture stdout/stderr.

        Returns:
            JobResult with execution details.
        """
        cmd = self.build_command(job)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            return JobResult(
                job=job,
                exit_code=proc.returncode,
                stdout=proc.stdout if capture_output else "",
                stderr=proc.stderr if capture_output else "",
                command=cmd,
            )
        except subprocess.TimeoutExpired:
            return JobResult(
                job=job,
                exit_code=-1,
                stdout="",
                stderr=f"Job timed out after {timeout} seconds",
                command=cmd,
            )
        except Exception as e:
            return JobResult(
                job=job,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                command=cmd,
            )

    def run_all(
        self,
        job_set: JobSet,
        *,
        cleanup: bool = True,
        timeout: float | None = None,
        on_complete: Callable[[JobResult], None] | None = None
    ) -> list[JobResult]:
        """Execute all jobs in a JobSet sequentially.

        Args:
            job_set: Collection of jobs to execute.
            cleanup: Whether to cleanup temp files after completion.
            timeout: Timeout per job in seconds.
            on_complete: Callback called after each job completes.

        Returns:
            List of JobResults in same order as jobs.
        """
        results = []

        try:
            for job in job_set.jobs:
                result = self.run(job, timeout=timeout)
                results.append(result)
                if on_complete:
                    on_complete(result)
        finally:
            if cleanup:
                job_set.cleanup()

        return results


# Convenience function for simple sweeps
def run_sweep(
    template: str | Path,
    source: Path,
    parameters: dict[str, Sequence[Any]],
    josh_jar: Path,
    simulation: str = "Main",
    replicates: int = 1,
    **kwargs: Any
) -> list[JobResult]:
    """Convenience function to run a parameter sweep.

    Args:
        template: Jinja template string or path to template file.
        source: Path to .josh source file.
        parameters: Dict mapping parameter names to lists of values.
        josh_jar: Path to joshsim.jar.
        simulation: Simulation name.
        replicates: Number of replicates per job.
        **kwargs: Additional arguments passed to JobConfig.

    Returns:
        List of JobResults.

    Example:
        results = run_sweep(
            template="maxGrowth = {{ maxGrowth }} meters",
            source=Path("hello.josh"),
            parameters={"maxGrowth": [1, 5, 10]},
            josh_jar=Path("joshsim.jar"),
        )
    """
    # Build sweep config
    sweep_params = [
        SweepParameter(name=name, values=list(values))
        for name, values in parameters.items()
    ]
    sweep = SweepConfig(parameters=sweep_params)

    # Build job config
    if isinstance(template, Path):
        config = JobConfig(
            template_path=template,
            source_path=source,
            simulation=simulation,
            replicates=replicates,
            sweep=sweep,
            **kwargs
        )
    else:
        config = JobConfig(
            template_string=template,
            source_path=source,
            simulation=simulation,
            replicates=replicates,
            sweep=sweep,
            **kwargs
        )

    # Expand and run
    expander = JobExpander()
    job_set = expander.expand(config)

    runner = JobRunner(josh_jar=josh_jar)
    return runner.run_all(job_set)
