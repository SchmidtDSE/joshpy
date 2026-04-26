"""Thin CLI wrapper for Josh CLI commands.

This module provides a clean interface to Josh CLI commands via subprocess.
Each command has a corresponding Config dataclass that maps 1:1 to CLI arguments.

Example usage:
    from pathlib import Path
    from joshpy.cli import (
        JoshCLI, RunConfig,
        NetcdfPreprocessConfig, GeotiffPreprocessConfig, CsvPreprocessConfig,
    )

    cli = JoshCLI()

    # Run a simulation
    result = cli.run(RunConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        replicates=5,
    ))

    # Preprocess NetCDF data
    result = cli.preprocess(NetcdfPreprocessConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        data_file=Path("temperature.nc"),
        variable="temp",
        units="K",
        output=Path("temperature.jshd"),
    ))

    # Preprocess GeoTIFF/COG data
    result = cli.preprocess(GeotiffPreprocessConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        data_file=Path("cover.tif"),
        band=0,
        units="percent",
        output=Path("cover.jshd"),
        timestep=0,
    ))

    # Preprocess CSV point data
    result = cli.preprocess(CsvPreprocessConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        data_file=Path("stations.csv"),
        variable="precipitation",
        units="mm",
        output=Path("precip.jshd"),
        timestep=0,
    ))
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from joshpy.jar import JarManager, JarMode


@dataclass
class CLIResult:
    """Result of any CLI command execution.

    Attributes:
        exit_code: Process exit code (0 = success).
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        command: The command that was executed.
    """

    exit_code: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Return True if command completed successfully."""
        return self.exit_code == 0


@dataclass(frozen=True)
class JfrConfig:
    """Java Flight Recorder configuration for profiling JVM operations.

    JFR is built into OpenJDK 11+ and captures detailed runtime profiling data
    (CPU, GC, memory, threads, I/O) with minimal overhead. Use this to diagnose
    performance issues in simulations or preprocessing.

    The resulting .jfr file can be analyzed with:
    - JDK Mission Control (GUI)
    - ``jfr`` CLI tool (bundled with JDK): ``jfr summary recording.jfr``
    - IntelliJ IDEA profiler
    - :meth:`JoshCLI.summarize_jfr` for a quick text summary

    Example::

        from joshpy.cli import JoshCLI, RunConfig, JfrConfig

        cli = JoshCLI()
        result = cli.run(
            RunConfig(script=Path("sim.josh"), simulation="Main"),
            jfr=JfrConfig(output=Path("recording.jfr")),
        )
        # Get text summary for GitHub issue
        summary = cli.summarize_jfr(Path("recording.jfr"))
        print(summary.stdout)

    Attributes:
        output: Path for the .jfr output file.
        settings: JFR settings profile. ``"profile"`` captures more detail
            (~2% overhead), ``"default"`` is lighter (~1% overhead).
            Default is ``"profile"`` since JFR is typically enabled for debugging.
        maxsize: Optional maximum recording file size (e.g., ``"500m"``, ``"1g"``).
            If None, no size limit is applied.
    """

    output: Path
    settings: str = "profile"
    maxsize: str | None = None


# -----------------------------------------------------------------------------
# Config Dataclasses - each maps 1:1 to a CLI command's arguments
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Arguments for 'java -jar joshsim.jar run' command.

    See: https://joshsim.org/cli#run

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation to execute.
        replicates: Number of replicates to run.
        data: Map of data file names to paths (--data name=path).
        custom_tags: Custom tags for template resolution (--custom-tag name=value).
        crs: Coordinate Reference System.
        output_format: Output format (csv, netcdf, geotiff).
        output: Output file path.
        parallel: Enable parallel patch processing.
        use_float64: Use double precision instead of BigDecimal.
        verbose: Enable verbose output.
        upload_source: Upload source .josh file to MinIO after completion.
        upload_config: Upload configuration .jshc files to MinIO after completion.
        upload_data: Upload data .jshd files to MinIO after completion.
        output_steps: Step range to output (e.g., "0-10,50,100").
        seed: Random seed for reproducibility.
        enable_profiler: Enable Josh evaluation profiler (--enable-profiler).
    """

    script: Path
    simulation: str
    replicates: int = 1
    data: dict[str, Path] = field(default_factory=dict)
    custom_tags: dict[str, str] = field(default_factory=dict)
    crs: str | None = None
    output_format: str | None = None
    output: Path | None = None
    parallel: bool = False
    use_float64: bool = False
    verbose: bool = False
    upload_source: bool = False
    upload_config: bool = False
    upload_data: bool = False
    output_steps: str | None = None
    seed: int | None = None
    enable_profiler: bool = False


@dataclass(frozen=True)
class RunRemoteConfig:
    """Arguments for 'java -jar joshsim.jar runRemote' command.

    Executes simulations on remote Josh infrastructure (Josh Cloud or local server).

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation to execute.
        api_key: API key for authentication (optional for local servers).
        replicates: Number of replicates to run.
        endpoint: Custom endpoint URL (e.g., local server or Josh Cloud).
        data: Map of data file names to paths.
        custom_tags: Custom tags for template resolution.
    """

    script: Path
    simulation: str
    api_key: str | None = None
    replicates: int = 1
    endpoint: str | None = None
    data: dict[str, Path] = field(default_factory=dict)
    custom_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NetcdfPreprocessConfig:
    """Preprocess NetCDF files into Josh's .jshd format.

    For files with time dimensions. Use x_coord, y_coord, and time_coord
    to specify dimension names if they differ from defaults.

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation for grid extraction.
        data_file: Path to NetCDF file (.nc, .nc4, .netcdf).
        variable: NetCDF variable name to extract (e.g., "tas", "pr").
        units: Units of the data for simulation use.
        output: Path for output .jshd file.
        x_coord: Name of X/longitude dimension (default: "lon").
        y_coord: Name of Y/latitude dimension (default: "lat").
        time_coord: Name of time dimension (default: "time").
        timestep: Extract specific time slice (optional).
        amend: Append to existing JSHD file.
        crs: Coordinate Reference System if not embedded.
        parallel: Enable parallel processing (~Nx speedup on N cores).
    """

    script: Path
    simulation: str
    data_file: Path
    variable: str
    units: str
    output: Path
    x_coord: str = "lon"
    y_coord: str = "lat"
    time_coord: str = "time"
    timestep: int | None = None
    amend: bool = False
    crs: str | None = None
    parallel: bool = False

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        suffix = str(self.data_file).lower()
        if not any(suffix.endswith(ext) for ext in (".nc", ".nc4", ".netcdf")):
            raise ValueError(
                f"NetcdfPreprocessConfig expects .nc/.nc4/.netcdf file, "
                f"got: {self.data_file.suffix}"
            )


@dataclass(frozen=True)
class GeotiffPreprocessConfig:
    """Preprocess GeoTIFF/COG files into Josh's .jshd format.

    For single-band rasters without time dimension. The `timestep` parameter
    is required to specify which simulation timestep the data maps to.

    Note: GeoTIFF spatial coordinates are implicit in the file format,
    so x_coord/y_coord options are not needed.

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation for grid extraction.
        data_file: Path to GeoTIFF file (.tif, .tiff).
        band: Band index to extract (0-based).
        units: Units of the data for simulation use.
        output: Path for output .jshd file.
        timestep: Simulation timestep this data maps to (required).
        amend: Append to existing JSHD file.
        crs: Coordinate Reference System if not embedded in TIF.
        parallel: Enable parallel processing (~Nx speedup on N cores).
    """

    script: Path
    simulation: str
    data_file: Path
    band: int
    units: str
    output: Path
    timestep: int  # Required, no default
    amend: bool = False
    crs: str | None = None
    parallel: bool = False

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if self.band < 0:
            raise ValueError(f"band must be >= 0, got: {self.band}")
        if self.timestep < 0:
            raise ValueError(f"timestep must be >= 0, got: {self.timestep}")
        suffix = str(self.data_file).lower()
        if not any(suffix.endswith(ext) for ext in (".tif", ".tiff", ".geotiff")):
            raise ValueError(
                f"GeotiffPreprocessConfig expects .tif/.tiff file, "
                f"got: {self.data_file.suffix}"
            )


@dataclass(frozen=True)
class CsvPreprocessConfig:
    """Preprocess CSV point data into Josh's .jshd format.

    CSV must have columns named exactly "longitude" and "latitude".
    All other columns are available as variables. The `timestep` parameter
    is required to specify which simulation timestep the data maps to.

    Note: CSV preprocessing uses brute-force nearest-neighbor lookup,
    which may be slow for large files. For large datasets, consider
    converting to GeoTIFF first.

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation for grid extraction.
        data_file: Path to CSV file (.csv).
        variable: Column name to extract.
        units: Units of the data for simulation use.
        output: Path for output .jshd file.
        timestep: Simulation timestep this data maps to (required).
        amend: Append to existing JSHD file.
        crs: Coordinate Reference System.
        parallel: Enable parallel processing (~Nx speedup on N cores).
    """

    script: Path
    simulation: str
    data_file: Path
    variable: str
    units: str
    output: Path
    timestep: int  # Required, no default
    amend: bool = False
    crs: str | None = None
    parallel: bool = False

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if self.timestep < 0:
            raise ValueError(f"timestep must be >= 0, got: {self.timestep}")
        if not str(self.data_file).lower().endswith(".csv"):
            raise ValueError(
                f"CsvPreprocessConfig expects .csv file, "
                f"got: {self.data_file.suffix}"
            )


# Type alias for preprocess method parameter
PreprocessConfig = Union[NetcdfPreprocessConfig, GeotiffPreprocessConfig, CsvPreprocessConfig]


@dataclass(frozen=True)
class ValidateConfig:
    """Arguments for 'java -jar joshsim.jar validate' command.

    Validates Josh script syntax and reports parsing errors.

    Attributes:
        script: Path to Josh script file to validate.
        verbose: Enable verbose validation output.
        upload_source: Upload source .josh file to MinIO after validation.
    """

    script: Path
    verbose: bool = False
    upload_source: bool = False


@dataclass(frozen=True)
class DiscoverConfigConfig:
    """Arguments for 'java -jar joshsim.jar discoverConfig' command.

    Analyzes Josh scripts to discover all configuration variables used.

    Attributes:
        script: Path to Josh script file to analyze.
    """

    script: Path


@dataclass(frozen=True)
class InspectJshdConfig:
    """Arguments for 'java -jar joshsim.jar inspectJshd' command.

    Inspects values in preprocessed JSHD files for debugging.

    Attributes:
        jshd_file: Path to JSHD file to inspect.
        variable: Variable name to examine.
        timestep: Time step to inspect.
        x: X coordinate (grid space).
        y: Y coordinate (grid space).
    """

    jshd_file: Path
    variable: str
    timestep: int
    x: int
    y: int


@dataclass(frozen=True)
class StageFromMinioConfig:
    """Arguments for 'java -jar joshsim.jar stageFromMinio' command.

    Downloads all objects under a MinIO prefix to a local directory.
    MinIO credentials are optional -- joshsim falls back to environment
    variables (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_BUCKET) via its HierarchyConfig.

    Attributes:
        output_dir: Local directory to download files into.
        prefix: MinIO object prefix to download from.
        minio_endpoint: MinIO endpoint URL (optional).
        minio_access_key: MinIO access key (optional).
        minio_secret_key: MinIO secret key (optional).
        minio_bucket: MinIO bucket name (optional).
        config_file: Path to JSON configuration file (optional).
        ensure_bucket_exists: Ensure the bucket exists before downloading.
        minio_path: Base object name/path within bucket (optional).
    """

    output_dir: Path
    prefix: str
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None
    config_file: Path | None = None
    ensure_bucket_exists: bool = False
    minio_path: str | None = None


@dataclass(frozen=True)
class StageToMinioConfig:
    """Arguments for 'java -jar joshsim.jar stageToMinio' command.

    Uploads a local directory to MinIO under a given prefix.
    MinIO credentials are optional -- joshsim falls back to environment
    variables via its HierarchyConfig.

    Attributes:
        input_dir: Local directory to upload.
        prefix: MinIO object prefix to upload to.
        minio_endpoint: MinIO endpoint URL (optional).
        minio_access_key: MinIO access key (optional).
        minio_secret_key: MinIO secret key (optional).
        minio_bucket: MinIO bucket name (optional).
        config_file: Path to JSON configuration file (optional).
        ensure_bucket_exists: Ensure the bucket exists before uploading.
        minio_path: Base object name/path within bucket (optional).
    """

    input_dir: Path
    prefix: str
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None
    config_file: Path | None = None
    ensure_bucket_exists: bool = False
    minio_path: str | None = None


@dataclass(frozen=True)
class BatchRemoteConfig:
    """Arguments for 'java -jar joshsim.jar batchRemote' command.

    Mirrors josh's CLI surface. joshpy's staging model is always explicit
    two-step: caller uploads inputs via :meth:`JoshCLI.stage_to_minio`, then
    dispatches via :meth:`JoshCLI.batch_remote` with ``require_prestaged=True``
    to verify the sentinel before the remote target starts.

    Attributes:
        simulation: Name of simulation to run.
        target: Target profile name (required).
        minio_prefix: MinIO object prefix where inputs already live (e.g.
            ``batch-jobs/my-run/inputs/``). Must have the staging sentinel
            present if ``require_prestaged=True``.
        replicates: Number of replicates (default: 1).
        no_wait: If True, dispatch and exit without polling (default: False).
        poll_interval: Polling interval in seconds (optional).
        timeout: Job timeout in seconds (optional).
        require_prestaged: Fail fast unless the staging sentinel at
            ``minio_prefix`` reports ``complete``. Recommended for sweeps;
            ``run_sweep`` sets this to True by default via
            :func:`joshpy.jobs.to_batch_remote_config`.
        custom_tags: Custom tags for template resolution (``--custom-tag
            name=value``). Pods receive these via the runtime's custom-tag
            propagation mechanism and resolve ``{name}`` references in
            export paths. joshpy's :func:`joshpy.jobs.to_batch_remote_config`
            auto-injects ``run_hash`` so users can write
            ``exportFiles.patch = "minio://bucket/{run_hash}/output_{replicate}.csv"``
            for deterministic per-simulation paths.
        replicate_start: Offset added to each pod's replicate index. When K,
            pods run as replicates K..K+replicates-1 (instead of 0..replicates-1).
            Used by the pool collision policy to append new replicates to an
            existing MinIO prefix without overwriting prior CSVs.
    """

    simulation: str
    target: str
    minio_prefix: str
    replicates: int = 1
    no_wait: bool = False
    poll_interval: int | None = None
    timeout: int | None = None
    require_prestaged: bool = False
    custom_tags: dict[str, str] = field(default_factory=dict)
    replicate_start: int = 0


@dataclass(frozen=True)
class PreprocessBatchConfig:
    """Arguments for 'java -jar joshsim.jar preprocessBatch' command.

    Preprocesses geospatial data on a remote target, downloads the
    resulting ``.jshd`` file.

    Attributes:
        script: Path to .josh file.
        simulation: Name of simulation.
        data_file: Input data file (e.g. .nc).
        variable: Variable name in the data file.
        units: Units of the data.
        output: Output .jshd file path.
        target: Target profile name (required).
        crs: CRS to use when reading the file.
        x_coord: Name of X coordinate dimension.
        y_coord: Name of Y coordinate dimension.
        time_dim: Name of time dimension.
        timestep: Single timestep to process.
        default_value: Default value to fill grid spaces before copying data.
        parallel: Enable parallel processing of patches within each timestep.
        amend: Amend existing output file rather than overwriting.
        no_wait: If True, dispatch and exit without polling for completion.
        poll_interval: Polling interval in seconds (optional).
        timeout: Maximum seconds to wait for completion (optional).
    """

    script: Path
    simulation: str
    data_file: Path
    variable: str
    units: str
    output: Path
    target: str
    crs: str | None = None
    x_coord: str | None = None
    y_coord: str | None = None
    time_dim: str | None = None
    timestep: int | None = None
    default_value: float | None = None
    parallel: bool = False
    amend: bool = False
    no_wait: bool = False
    poll_interval: int | None = None
    timeout: int | None = None


@dataclass(frozen=True)
class PollBatchConfig:
    """Arguments for 'java -jar joshsim.jar pollBatch' command.

    Single-shot status check for a dispatched batch job.

    Exit codes from the JAR:
        0 — complete (job succeeded)
        1 — error (job failed)
        2 — running / pending
        100 — poll failure (transient, retry)

    Attributes:
        job_id: Job ID returned by ``batchRemote --no-wait``.
        target: Target profile name (loads ``~/.josh/targets/<name>.json``).
    """

    job_id: str
    target: str


@dataclass(frozen=True)
class InspectExportsConfig:
    """Arguments for 'java -jar joshsim.jar inspect-exports' command.

    Parses a Josh script and extracts export/debug file paths without running.

    Attributes:
        script: Path to Josh script file.
        simulation: Name of simulation to inspect.
        json_output: Output in JSON format (default: True).
    """

    script: Path
    simulation: str
    json_output: bool = True


@dataclass
class ExportFileInfo:
    """Parsed export file information.

    Attributes:
        raw: Original path string as specified in script.
        protocol: URL protocol (e.g., "file").
        host: URL host (empty for local files).
        path: File path component.
        file_type: File extension/type (e.g., "csv").
    """

    raw: str
    protocol: str
    host: str
    path: str
    file_type: str


@dataclass
class ExportPaths:
    """Parsed export paths from inspect-exports output.

    Attributes:
        simulation: Simulation name.
        export_files: Dict of export type -> ExportFileInfo or None.
                      Keys: "patch", "meta", "entity"
        debug_files: Dict of debug type -> ExportFileInfo or None.
                     Keys: "organism", "patch", "agent", "disturbance"
    """

    simulation: str
    export_files: dict[str, ExportFileInfo | None]
    debug_files: dict[str, ExportFileInfo | None]

    def get_patch_path(self) -> str | None:
        """Get the patch export file path, if configured."""
        if self.export_files.get("patch"):
            return self.export_files["patch"].path
        return None

    def resolve_path(self, path_template: str, **kwargs: Any) -> Path:
        """Resolve template variables in a path.

        Template variables: {replicate}, {step}, {variable}, {timestamp},
        and any custom job parameters.

        Args:
            path_template: Path with template variables.
            **kwargs: Values to substitute (e.g., replicate=0, maxGrowth=50).

        Returns:
            Resolved Path object.

        Examples:
            >>> path = export_paths.resolve_path(
            ...     "/tmp/output_{maxGrowth}_{replicate}.csv",
            ...     maxGrowth=50,
            ...     replicate=0,
            ... )
            >>> # Returns: Path("/tmp/output_50_0.csv")
        """
        resolved = path_template.format(**kwargs)
        return Path(resolved)


# -----------------------------------------------------------------------------
# JoshCLI - The executor class
# -----------------------------------------------------------------------------


@dataclass
class JoshCLI:
    """Executes Josh CLI commands via subprocess.

    This is a thin wrapper that builds command lines and runs them.
    No business logic - just CLI invocation.

    Supports three jar modes:
    - Path: Use a specific jar file
    - JarMode.PROD: Use production jar from joshsim.org (default)
    - JarMode.DEV: Use development jar from joshsim.org
    - JarMode.LOCAL: Use local jar from jar/joshsim-fat.jar

    Attributes:
        josh_jar: Path to jar, JarMode enum, or None for default (PROD).
        java_path: Path to java executable.
        auto_download: If True, download jars automatically if needed.
        working_dir: Working directory for command execution.

    Examples:
        >>> cli = JoshCLI(josh_jar=Path("joshsim-fat.jar"))

        >>> # Run a simulation
        >>> result = cli.run(RunConfig(
        ...     script=Path("sim.josh"),
        ...     simulation="Main",
        ...     replicates=5,
        ... ))

        >>> # Preprocess data
        >>> result = cli.preprocess(PreprocessConfig(
        ...     script=Path("sim.josh"),
        ...     simulation="Main",
        ...     data_file=Path("temp.nc"),
        ...     variable="temperature",
        ...     units="K",
        ...     output=Path("temp.jshd"),
        ... ))
    """

    josh_jar: Path | JarMode | None = None
    java_path: str = "java"
    auto_download: bool = True
    working_dir: Path | None = None
    jar_dir: Path | None = None

    # Resolved jar path (set in __post_init__)
    _resolved_jar: Path = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Resolve jar path based on josh_jar setting."""
        manager = JarManager(jar_dir=self.jar_dir, java_path=self.java_path)

        if self.josh_jar is None:
            # Default to PROD
            self._resolved_jar = manager.get_jar(JarMode.PROD, auto_download=self.auto_download)
        elif isinstance(self.josh_jar, JarMode):
            # JarMode enum
            self._resolved_jar = manager.get_jar(self.josh_jar, auto_download=self.auto_download)
        elif isinstance(self.josh_jar, (str, Path)):
            # Explicit path
            self._resolved_jar = Path(self.josh_jar)
            if not self._resolved_jar.exists():
                raise FileNotFoundError(f"JAR file not found: {self._resolved_jar}")
        else:
            raise TypeError(f"josh_jar must be Path, JarMode, or None, got {type(self.josh_jar)}")

        # Ensure jar path is absolute since we may change working directories
        self._resolved_jar = self._resolved_jar.resolve()

    def _execute(
        self,
        args: list[str],
        timeout: float | None = None,
        capture_output: bool = True,
        jfr: JfrConfig | None = None,
        stream_output: bool = False,
    ) -> CLIResult:
        """Execute a CLI command.

        Args:
            args: Command arguments (without java -jar prefix).
            timeout: Timeout in seconds (None for no timeout).
            capture_output: Whether to capture stdout/stderr.
            jfr: Optional JFR configuration for profiling. When provided,
                adds ``-XX:StartFlightRecording`` JVM flag before ``-jar``.
            stream_output: If True, stream stdout/stderr to the terminal
                in real time while still capturing them in CLIResult.
                Overrides capture_output when True.

        Returns:
            CLIResult with execution details.
        """
        jvm_flags: list[str] = []
        if jfr is not None:
            jfr_opts = (
                f"filename={jfr.output.resolve()},"
                f"settings={jfr.settings},"
                f"dumponexit=true"
            )
            if jfr.maxsize is not None:
                jfr_opts += f",maxsize={jfr.maxsize}"
            jvm_flags.append(f"-XX:StartFlightRecording={jfr_opts}")

        cmd = [self.java_path] + jvm_flags + ["-jar", str(self._resolved_jar)] + args

        if stream_output:
            return self._execute_streaming(cmd, timeout=timeout)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            return CLIResult(
                exit_code=proc.returncode,
                stdout=proc.stdout if capture_output else "",
                stderr=proc.stderr if capture_output else "",
                command=cmd,
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                command=cmd,
            )
        except Exception as e:
            return CLIResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                command=cmd,
            )

    def stage_from_minio(
        self,
        config: StageFromMinioConfig,
        timeout: float | None = None,
    ) -> CLIResult:
        """Download files from MinIO to a local directory.

        Args:
            config: Stage-from-MinIO configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "stageFromMinio",
            "--output-dir", str(config.output_dir.resolve()),
            "--prefix", config.prefix,
        ]

        if config.minio_endpoint:
            args.extend(["--minio-endpoint", config.minio_endpoint])
        if config.minio_access_key:
            args.extend(["--minio-access-key", config.minio_access_key])
        if config.minio_secret_key:
            args.extend(["--minio-secret-key", config.minio_secret_key])
        if config.minio_bucket:
            args.extend(["--minio-bucket", config.minio_bucket])
        if config.config_file is not None:
            args.append(f"--config-file={config.config_file.resolve()}")
        if config.ensure_bucket_exists:
            args.append("--ensure-bucket-exists")
        if config.minio_path is not None:
            args.append(f"--minio-path={config.minio_path}")

        return self._execute(args, timeout=timeout)

    def stage_to_minio(
        self,
        config: StageToMinioConfig,
        timeout: float | None = None,
    ) -> CLIResult:
        """Upload a local directory to MinIO.

        Args:
            config: Stage-to-MinIO configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "stageToMinio",
            "--input-dir", str(config.input_dir.resolve()),
            "--prefix", config.prefix,
        ]

        if config.minio_endpoint:
            args.extend(["--minio-endpoint", config.minio_endpoint])
        if config.minio_access_key:
            args.extend(["--minio-access-key", config.minio_access_key])
        if config.minio_secret_key:
            args.extend(["--minio-secret-key", config.minio_secret_key])
        if config.minio_bucket:
            args.extend(["--minio-bucket", config.minio_bucket])
        if config.config_file is not None:
            args.append(f"--config-file={config.config_file.resolve()}")
        if config.ensure_bucket_exists:
            args.append("--ensure-bucket-exists")
        if config.minio_path is not None:
            args.append(f"--minio-path={config.minio_path}")

        return self._execute(args, timeout=timeout)

    def batch_remote(
        self,
        config: BatchRemoteConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
        stream_output: bool = False,
    ) -> CLIResult:
        """Dispatch a simulation to a remote target via MinIO staging.

        Args:
            config: Batch remote configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.
            stream_output: Stream JAR output to terminal in real time.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "batchRemote",
            f"--target={config.target}",
            f"--minio-prefix={config.minio_prefix}",
        ]

        if config.replicates > 1:
            args.append(f"--replicates={config.replicates}")
        if config.no_wait:
            args.append("--no-wait")
        if config.poll_interval is not None:
            args.append(f"--poll-interval={config.poll_interval}")
        if config.timeout is not None:
            args.append(f"--timeout={config.timeout}")
        if config.require_prestaged:
            args.append("--require-prestaged")
        if config.replicate_start > 0:
            args.append(f"--replicate-start={config.replicate_start}")
        for name, value in config.custom_tags.items():
            args.extend(["--custom-tag", f"{name}={value}"])

        args.append(config.simulation)

        return self._execute(
            args, timeout=timeout, jfr=jfr, stream_output=stream_output,
        )

    def preprocess_batch(
        self,
        config: PreprocessBatchConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> CLIResult:
        """Preprocess geospatial data on a remote target.

        Args:
            config: Preprocess-batch configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "preprocessBatch",
            str(config.script.resolve()),
            config.simulation,
            str(config.data_file.resolve()),
            config.variable,
            config.units,
            str(config.output.resolve()),
            f"--target={config.target}",
        ]

        if config.crs is not None:
            args.append(f"--crs={config.crs}")
        if config.x_coord is not None:
            args.append(f"--x-coord={config.x_coord}")
        if config.y_coord is not None:
            args.append(f"--y-coord={config.y_coord}")
        if config.time_dim is not None:
            args.append(f"--time-dim={config.time_dim}")
        if config.timestep is not None:
            args.append(f"--timestep={config.timestep}")
        if config.default_value is not None:
            args.append(f"--default-value={config.default_value}")
        if config.parallel:
            args.append("--parallel")
        if config.amend:
            args.append("--amend")
        if config.no_wait:
            args.append("--no-wait")
        if config.poll_interval is not None:
            args.append(f"--poll-interval={config.poll_interval}")
        if config.timeout is not None:
            args.append(f"--timeout={config.timeout}")

        return self._execute(args, timeout=timeout, jfr=jfr)

    def poll_batch(
        self,
        config: PollBatchConfig,
        timeout: float | None = None,
    ) -> CLIResult:
        """Check the status of a dispatched batch job.

        Exit codes:
            0 — complete (job succeeded, stdout has JSON status)
            1 — error (job failed, stdout has JSON with error details)
            2 — running / pending
            100 — poll failure (transient error, caller should retry)

        Args:
            config: Poll-batch configuration with job_id and target.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with exit_code indicating job state.
        """
        args = [
            "pollBatch",
            config.job_id,
            f"--target={config.target}",
        ]

        return self._execute(args, timeout=timeout)

    def _execute_streaming(
        self,
        cmd: list[str],
        timeout: float | None = None,
    ) -> CLIResult:
        """Execute a command, streaming stdout/stderr while capturing both.

        Uses Popen with reader threads to tee output to the terminal
        in real time while accumulating it for CLIResult.

        Args:
            cmd: Full command list to execute.
            timeout: Timeout in seconds (None for no timeout).

        Returns:
            CLIResult with captured output.
        """
        import sys
        import threading

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _reader(stream, sink: list[str], dest):
            """Read lines from *stream*, write to *dest*, accumulate in *sink*."""
            for line in stream:
                dest.write(line)
                dest.flush()
                sink.append(line)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
            )

            out_thread = threading.Thread(
                target=_reader, args=(proc.stdout, stdout_lines, sys.stdout),
            )
            err_thread = threading.Thread(
                target=_reader, args=(proc.stderr, stderr_lines, sys.stderr),
            )
            out_thread.start()
            err_thread.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                out_thread.join()
                err_thread.join()
                return CLIResult(
                    exit_code=-1,
                    stdout="".join(stdout_lines),
                    stderr=f"Command timed out after {timeout} seconds",
                    command=cmd,
                )

            out_thread.join()
            err_thread.join()

            return CLIResult(
                exit_code=proc.returncode,
                stdout="".join(stdout_lines),
                stderr="".join(stderr_lines),
                command=cmd,
            )
        except Exception as e:
            return CLIResult(
                exit_code=-1,
                stdout="".join(stdout_lines),
                stderr=str(e),
                command=cmd,
            )

    def run(
        self,
        config: RunConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
        stream_output: bool = False,
    ) -> CLIResult:
        """Execute a simulation.

        Args:
            config: Run configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.
            stream_output: If True, stream JAR stdout/stderr to the
                terminal in real time while still capturing them.

        Returns:
            CLIResult with execution details.
        """
        args = ["run", str(config.script.resolve()), config.simulation]

        # Replicates
        if config.replicates > 1:
            args.extend(["--replicates", str(config.replicates)])

        # Data files
        for name, path in config.data.items():
            data_name = name if '.' in name else f"{name}{path.suffix}"
            args.extend(["--data", f"{data_name}={path.resolve()}"])

        # Custom tags
        for name, value in config.custom_tags.items():
            args.extend(["--custom-tag", f"{name}={value}"])

        # Optional arguments
        if config.crs:
            args.extend(["--crs", config.crs])
        if config.output_format:
            args.extend(["--output-format", config.output_format])
        if config.output:
            args.extend(["--output", str(config.output.resolve())])
        if config.parallel:
            args.append("--parallel")
        if config.use_float64:
            args.append("--use-float-64")
        if config.verbose:
            args.append("--verbose")
        if config.upload_source:
            args.append("--upload-source")
        if config.upload_config:
            args.append("--upload-config")
        if config.upload_data:
            args.append("--upload-data")
        if config.output_steps:
            args.extend(["--output-steps", config.output_steps])
        if config.seed is not None:
            args.extend(["--seed", str(config.seed)])
        if config.enable_profiler:
            args.append("--enable-profiler")

        return self._execute(args, timeout=timeout, jfr=jfr, stream_output=stream_output)

    def run_remote(
        self,
        config: RunRemoteConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
        stream_output: bool = False,
    ) -> CLIResult:
        """Execute a simulation on Josh Cloud.

        Args:
            config: Run remote configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.
            stream_output: If True, stream JAR stdout/stderr to the
                terminal in real time while still capturing them.

        Returns:
            CLIResult with execution details.
        """
        args = ["runRemote", str(config.script.resolve()), config.simulation]

        # API key (optional - not needed for local servers)
        if config.api_key is not None:
            args.extend(["--api-key", config.api_key])

        # Replicates
        if config.replicates > 1:
            args.extend(["--replicates", str(config.replicates)])

        # Endpoint
        if config.endpoint:
            args.extend(["--endpoint", config.endpoint])

        # Data files
        for name, path in config.data.items():
            data_name = name if '.' in name else f"{name}{path.suffix}"
            args.extend(["--data", f"{data_name}={path.resolve()}"])

        # Custom tags
        for name, value in config.custom_tags.items():
            args.extend(["--custom-tag", f"{name}={value}"])

        return self._execute(args, timeout=timeout, jfr=jfr, stream_output=stream_output)

    def preprocess(
        self,
        config: PreprocessConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> CLIResult:
        """Preprocess geospatial data into JSHD format.

        Supports NetCDF, GeoTIFF/COG, and CSV input formats. Use the
        appropriate config class for your input format:

        - NetcdfPreprocessConfig: For .nc files with time dimensions
        - GeotiffPreprocessConfig: For .tif/.tiff/COG rasters
        - CsvPreprocessConfig: For .csv point data

        Args:
            config: Format-specific preprocess configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            CLIResult with execution details.
        """
        # Build common args
        args = [
            "preprocess",
            str(config.script.resolve()),
            config.simulation,
            str(config.data_file.resolve()),
        ]

        # Variable/band differs by format
        if isinstance(config, GeotiffPreprocessConfig):
            args.append(str(config.band))  # Band index as string
        else:
            args.append(config.variable)  # Variable name for NetCDF/CSV

        args.extend([
            config.units,
            str(config.output.resolve()),
        ])

        # Format-specific optional arguments
        if isinstance(config, NetcdfPreprocessConfig):
            # NetCDF has coordinate dimension names
            if config.x_coord:
                args.extend(["--x-coord", config.x_coord])
            if config.y_coord:
                args.extend(["--y-coord", config.y_coord])
            if config.time_coord:
                args.extend(["--time-dim", config.time_coord])
            if config.timestep is not None:
                args.extend(["--timestep", str(config.timestep)])
        else:
            # GeoTIFF and CSV require timestep (validated in __post_init__)
            args.extend(["--timestep", str(config.timestep)])

        # Common optional arguments
        if config.amend:
            args.append("--amend")
        if config.crs:
            args.extend(["--crs", config.crs])
        if config.parallel:
            args.append("--parallel")

        return self._execute(args, timeout=timeout, jfr=jfr)

    def validate(
        self,
        config: ValidateConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> CLIResult:
        """Validate a Josh script.

        Args:
            config: Validate configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            CLIResult with execution details.
        """
        args = ["validate", str(config.script.resolve())]

        if config.verbose:
            args.append("--verbose")
        if config.upload_source:
            args.append("--upload-source")

        return self._execute(args, timeout=timeout, jfr=jfr)

    def discover_config(
        self,
        config: DiscoverConfigConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> CLIResult:
        """Discover configuration variables in a Josh script.

        Args:
            config: Discover config configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            CLIResult with execution details.
        """
        args = ["discoverConfig", str(config.script.resolve())]
        return self._execute(args, timeout=timeout, jfr=jfr)

    def inspect_jshd(
        self,
        config: InspectJshdConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> CLIResult:
        """Inspect values in a JSHD file.

        Args:
            config: Inspect JSHD configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "inspectJshd",
            str(config.jshd_file.resolve()),
            config.variable,
            str(config.timestep),
            str(config.x),
            str(config.y),
        ]
        return self._execute(args, timeout=timeout, jfr=jfr)

    def inspect_exports(
        self,
        config: InspectExportsConfig,
        timeout: float | None = None,
        jfr: JfrConfig | None = None,
    ) -> ExportPaths:
        """Inspect export paths in a Josh script.

        Args:
            config: Inspect exports configuration.
            timeout: Timeout in seconds.
            jfr: Optional JFR profiling configuration.

        Returns:
            ExportPaths with parsed export file information.

        Raises:
            RuntimeError: If command fails (includes exit code context).

        """
        args = ["inspect-exports", str(config.script.resolve()), config.simulation]

        # Note: --json is a toggle flag with default=true. Passing --json toggles it OFF.
        # So we only add --json when json_output is False (to get human-readable output).
        if not config.json_output:
            args.append("--json")

        result = self._execute(args, timeout=timeout, jfr=jfr)

        if not result.success:
            raise RuntimeError(
                f"inspect-exports failed (exit code {result.exit_code}): {result.stderr}"
            )

        # Parse JSON output
        data = json.loads(result.stdout)

        def parse_file_info(info: dict[str, Any] | None) -> ExportFileInfo | None:
            if info is None:
                return None
            return ExportFileInfo(
                raw=info["raw"],
                protocol=info["protocol"],
                host=info["host"],
                path=info["path"],
                file_type=info["fileType"],
            )

        return ExportPaths(
            simulation=data["simulation"],
            export_files={
                "patch": parse_file_info(data["exportFiles"].get("patch")),
                "meta": parse_file_info(data["exportFiles"].get("meta")),
                "entity": parse_file_info(data["exportFiles"].get("entity")),
            },
            debug_files={
                "organism": parse_file_info(data["debugFiles"].get("organism")),
                "patch": parse_file_info(data["debugFiles"].get("patch")),
                "agent": parse_file_info(data["debugFiles"].get("agent")),
                "disturbance": parse_file_info(data["debugFiles"].get("disturbance")),
            },
        )

    def diagnose_jfr(
        self,
        jfr_path: Path,
        timeout: float | None = None,
    ) -> "ResourceProfile":
        """Parse a JFR recording into user-friendly resource diagnostics.

        Runs ``jfr print`` and ``jfr summary`` to extract CPU, memory, GC,
        I/O, and thread contention metrics, then classifies the likely
        bottleneck.

        Args:
            jfr_path: Path to the ``.jfr`` recording file.
            timeout: Timeout in seconds for each jfr subprocess call.

        Returns:
            :class:`~joshpy.jfr.ResourceProfile` with parsed diagnostics.

        Raises:
            RuntimeError: If jfr commands fail.
        """
        from joshpy.jfr import build_resource_profile
        from joshpy.jfr.__main__ import _EVENTS

        java = Path(self.java_path)
        if java.parent.name == "bin":
            jfr_bin = str(java.parent / "jfr")
        else:
            jfr_bin = "jfr"

        resolved = str(jfr_path.resolve())

        # Get event data
        try:
            print_proc = subprocess.run(
                [jfr_bin, "print", "--events", _EVENTS, resolved],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"jfr print timed out after {timeout}s") from e

        if print_proc.returncode != 0:
            raise RuntimeError(f"jfr print failed: {print_proc.stderr}")

        # Get summary for recording duration
        summary_result = self.summarize_jfr(jfr_path, timeout=timeout)
        if not summary_result.success:
            raise RuntimeError(f"jfr summary failed: {summary_result.stderr}")

        return build_resource_profile(print_proc.stdout, summary_result.stdout)

    def summarize_jfr(self, jfr_path: Path, timeout: float | None = None) -> CLIResult:
        """Get a text summary of a JFR recording.

        Uses the ``jfr`` CLI tool bundled with the JDK to produce a human-readable
        summary of a flight recording. Users can paste this into GitHub issues
        without needing JDK Mission Control or IntelliJ.

        The ``jfr`` binary is resolved relative to :attr:`java_path`: if
        ``java_path`` points into a JDK ``bin/`` directory, the sibling ``jfr``
        binary is used; otherwise ``jfr`` is expected to be on ``PATH``.

        Args:
            jfr_path: Path to the ``.jfr`` recording file.
            timeout: Timeout in seconds.

        Returns:
            CLIResult where stdout contains the text summary.
        """
        java = Path(self.java_path)
        if java.parent.name == "bin":
            jfr_bin = str(java.parent / "jfr")
        else:
            jfr_bin = "jfr"

        cmd = [jfr_bin, "summary", str(jfr_path.resolve())]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CLIResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                command=cmd,
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                command=cmd,
            )
        except Exception as e:
            return CLIResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                command=cmd,
            )
