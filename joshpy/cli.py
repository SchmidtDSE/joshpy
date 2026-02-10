"""Thin CLI wrapper for Josh CLI commands.

This module provides a clean interface to Josh CLI commands via subprocess.
Each command has a corresponding Config dataclass that maps 1:1 to CLI arguments.

Example usage:
    from pathlib import Path
    from joshpy.cli import JoshCLI, RunConfig, PreprocessConfig

    cli = JoshCLI()

    # Run a simulation
    result = cli.run(RunConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        replicates=5,
    ))

    # Preprocess data
    result = cli.preprocess(PreprocessConfig(
        script=Path("simulation.josh"),
        simulation="Main",
        data_file=Path("temperature.nc"),
        variable="temp",
        units="K",
        output=Path("temperature.jshd"),
    ))
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class RunRemoteConfig:
    """Arguments for 'java -jar joshsim.jar runRemote' command.

    Executes simulations on Josh Cloud infrastructure.

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation to execute.
        api_key: Josh Cloud API key.
        replicates: Number of replicates to run.
        endpoint: Custom Josh Cloud endpoint URL.
        data: Map of data file names to paths.
        custom_tags: Custom tags for template resolution.
    """

    script: Path
    simulation: str
    api_key: str
    replicates: int = 1
    endpoint: str | None = None
    data: dict[str, Path] = field(default_factory=dict)
    custom_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessConfig:
    """Arguments for 'java -jar joshsim.jar preprocess' command.

    Converts geospatial data (NetCDF, GeoTIFF) into Josh's .jshd format.

    Attributes:
        script: Path to Josh simulation file (.josh).
        simulation: Name of simulation for preprocessing.
        data_file: Path to input data file (NetCDF, GeoTIFF, etc.).
        variable: Variable name or band number to extract.
        units: Units of the data for simulation use.
        output: Path for output preprocessed file (.jshd).
        amend: Amend existing JSHD file rather than overwriting.
        crs: Coordinate Reference System for reading input file.
        x_coord: Name of X coordinate dimension.
        y_coord: Name of Y coordinate dimension.
        time_coord: Name of time dimension.
        verbose: Enable verbose output.
    """

    script: Path
    simulation: str
    data_file: Path
    variable: str
    units: str
    output: Path
    amend: bool = False
    crs: str | None = None
    x_coord: str | None = None
    y_coord: str | None = None
    time_coord: str | None = None
    verbose: bool = False


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

        Example:
            path = export_paths.resolve_path(
                "/tmp/output_{maxGrowth}_{replicate}.csv",
                maxGrowth=50,
                replicate=0,
            )
            # Returns: Path("/tmp/output_50_0.csv")
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

    Example:
        cli = JoshCLI(josh_jar=Path("joshsim-fat.jar"))

        # Run a simulation
        result = cli.run(RunConfig(
            script=Path("sim.josh"),
            simulation="Main",
            replicates=5,
        ))

        # Preprocess data
        result = cli.preprocess(PreprocessConfig(
            script=Path("sim.josh"),
            simulation="Main",
            data_file=Path("temp.nc"),
            variable="temperature",
            units="K",
            output=Path("temp.jshd"),
        ))
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
        self, args: list[str], timeout: float | None = None, capture_output: bool = True
    ) -> CLIResult:
        """Execute a CLI command.

        Args:
            args: Command arguments (without java -jar prefix).
            timeout: Timeout in seconds (None for no timeout).
            capture_output: Whether to capture stdout/stderr.

        Returns:
            CLIResult with execution details.
        """
        cmd = [self.java_path, "-jar", str(self._resolved_jar)] + args

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

    def run(self, config: RunConfig, timeout: float | None = None) -> CLIResult:
        """Execute a simulation.

        Args:
            config: Run configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = ["run", str(config.script.resolve()), config.simulation]

        # Replicates
        if config.replicates > 1:
            args.extend(["--replicates", str(config.replicates)])

        # Data files
        for name, path in config.data.items():
            args.extend(["--data", f"{name}={path.resolve()}"])

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

        return self._execute(args, timeout=timeout)

    def run_remote(self, config: RunRemoteConfig, timeout: float | None = None) -> CLIResult:
        """Execute a simulation on Josh Cloud.

        Args:
            config: Run remote configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = ["runRemote", str(config.script.resolve()), config.simulation]

        # API key (required)
        args.extend(["--api-key", config.api_key])

        # Replicates
        if config.replicates > 1:
            args.extend(["--replicates", str(config.replicates)])

        # Endpoint
        if config.endpoint:
            args.extend(["--endpoint", config.endpoint])

        # Data files
        for name, path in config.data.items():
            args.extend(["--data", f"{name}={path.resolve()}"])

        # Custom tags
        for name, value in config.custom_tags.items():
            args.extend(["--custom-tag", f"{name}={value}"])

        return self._execute(args, timeout=timeout)

    def preprocess(self, config: PreprocessConfig, timeout: float | None = None) -> CLIResult:
        """Preprocess geospatial data into JSHD format.

        Args:
            config: Preprocess configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = [
            "preprocess",
            str(config.script.resolve()),
            config.simulation,
            str(config.data_file.resolve()),
            config.variable,
            config.units,
            str(config.output.resolve()),
        ]

        # Optional arguments
        if config.amend:
            args.append("--amend")
        if config.crs:
            args.extend(["--crs", config.crs])
        if config.x_coord:
            args.extend(["--x-coord", config.x_coord])
        if config.y_coord:
            args.extend(["--y-coord", config.y_coord])
        if config.time_coord:
            args.extend(["--time-coord", config.time_coord])
        if config.verbose:
            args.append("--verbose")

        return self._execute(args, timeout=timeout)

    def validate(self, config: ValidateConfig, timeout: float | None = None) -> CLIResult:
        """Validate a Josh script.

        Args:
            config: Validate configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = ["validate", str(config.script.resolve())]

        if config.verbose:
            args.append("--verbose")
        if config.upload_source:
            args.append("--upload-source")

        return self._execute(args, timeout=timeout)

    def discover_config(
        self, config: DiscoverConfigConfig, timeout: float | None = None
    ) -> CLIResult:
        """Discover configuration variables in a Josh script.

        Args:
            config: Discover config configuration.
            timeout: Timeout in seconds.

        Returns:
            CLIResult with execution details.
        """
        args = ["discoverConfig", str(config.script.resolve())]
        return self._execute(args, timeout=timeout)

    def inspect_jshd(self, config: InspectJshdConfig, timeout: float | None = None) -> CLIResult:
        """Inspect values in a JSHD file.

        Args:
            config: Inspect JSHD configuration.
            timeout: Timeout in seconds.

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
        return self._execute(args, timeout=timeout)

    def inspect_exports(
        self, config: InspectExportsConfig, timeout: float | None = None
    ) -> ExportPaths:
        """Inspect export paths in a Josh script.

        Args:
            config: Inspect exports configuration.
            timeout: Timeout in seconds.

        Returns:
            ExportPaths with parsed export file information.

        Raises:
            RuntimeError: If command fails (includes exit code context).

        Note:
            Currently only available in local jar (joshsim-fat.jar),
            not yet in prod/dev jars.
        """
        args = ["inspect-exports", str(config.script.resolve()), config.simulation]

        # Note: --json is a toggle flag with default=true. Passing --json toggles it OFF.
        # So we only add --json when json_output is False (to get human-readable output).
        if not config.json_output:
            args.append("--json")

        result = self._execute(args, timeout=timeout)

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
