"""Python library for interacting with Josh ecological simulations.

joshpy provides a Python interface for:
- Triggering Josh simulations (local or cloud execution)
- Tracking simulation runs with DuckDB-backed registry
- Collecting and analyzing simulation outputs
- Parameter sweeps with Jinja templating

License: BSD-3-Clause
"""

# Jar management (always available)
from joshpy.jar import (
    JarMode,
    JarManager,
    get_jar,
    download_jars,
    get_jar_hash,
    get_jar_version,
)

# CLI module (always available)
from joshpy.cli import (
    JoshCLI,
    CLIResult,
    JfrConfig,
    RunConfig,
    RunRemoteConfig,
    # Format-specific preprocess configs
    NetcdfPreprocessConfig,
    GeotiffPreprocessConfig,
    CsvPreprocessConfig,
    PreprocessConfig,  # Type alias for Union of the above
    ValidateConfig,
    DiscoverConfigConfig,
    InspectJshdConfig,
    InspectExportsConfig,
    ExportFileInfo,
    ExportPaths,
    StageFromMinioConfig,
)

# JFR diagnostics (always available, no external deps)
from joshpy.jfr import (
    ResourceProfile,
    CpuProfile,
    GcProfile,
    MemoryProfile,
    IoProfile,
    ContentionProfile,
    build_resource_profile,
)

# JSHD loading module (always available)
from joshpy.jshd import (
    JshdMetadata,
    JshdData,
    load_jshd,
    plot_jshd,
)

# Config parser (always available, no external deps)
from joshpy.config_parser import parse_jshc, parse_jshc_content

# Debug log parsing module (always available)
from joshpy.debug import (
    DebugMessage,
    DebugMessageStore,
    DebugSummary,
    parse_debug_line,
    load_debug_file,
    load_debug_from_script,
    format_message,
)

# Optional jobs module (requires jinja2 and pyyaml)
try:
    from joshpy.jobs import (
        ConfigSweepParameter,
        FileSweepParameter,
        CompoundSweepParameter,
        SweepConfig,
        JobConfig,
        ExpandedJob,
        JobSet,
        JobExpander,
        SweepResult,
        AdaptiveSweepResult,
        to_run_config,
        to_run_remote_config,
        run_sweep,
        discover_jshd_files,
    )
    from joshpy.strategies import (
        SweepStrategy,
        CartesianStrategy,
        OptunaStrategy,
        ObjectiveFn,
        strategy_from_dict,
        sample_params_from_trial,
        run_adaptive_sweep,
        SweepExecutionError,
        EXIT_CODE_DIAGNOSTICS,
        get_exit_code_diagnostic,
        cv_objective,
    )
    from joshpy.grid import GridSpec

    HAS_JOBS = True
except ImportError:
    HAS_JOBS = False

# Optional registry module (requires duckdb)
try:
    from joshpy.registry import (
        RunRegistry,
        RegistryCallback,
        SessionInfo,
        ConfigInfo,
        RunInfo,
        SessionSummary,
        DataSummary,
        configure_s3,
    )
    from joshpy.cell_data import (
        CellDataLoader,
        DiagnosticQueries,
    )
    from joshpy.diagnostics import SimulationDiagnostics
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

# Optional sweep module (requires jobs + registry)
try:
    from joshpy.sweep import (
        SweepManager,
        SweepManagerBuilder,
        recover_sweep_results,
        load_job_results,
        ingest_results,
        LoadConfig,
        ResultLoadError,
    )
    HAS_SWEEP = True
except ImportError:
    HAS_SWEEP = False

# Optional catalog module (requires duckdb)
try:
    from joshpy.catalog import (
        ProjectCatalog,
        ExperimentInfo,
        ModelInfo,
        DataManifestInfo,
        compute_model_hash,
        compute_config_hash,
        compute_data_manifest_hash,
    )
    HAS_CATALOG = True
except ImportError:
    HAS_CATALOG = False

__all__ = [
    # Jar management
    "JarMode",
    "JarManager",
    "get_jar",
    "download_jars",
    "get_jar_hash",
    "get_jar_version",
    # CLI
    "JoshCLI",
    "CLIResult",
    "JfrConfig",
    "RunConfig",
    "RunRemoteConfig",
    "NetcdfPreprocessConfig",
    "GeotiffPreprocessConfig",
    "CsvPreprocessConfig",
    "PreprocessConfig",
    "ValidateConfig",
    "DiscoverConfigConfig",
    "InspectJshdConfig",
    "InspectExportsConfig",
    "ExportFileInfo",
    "ExportPaths",
    "StageFromMinioConfig",
    # JFR diagnostics
    "ResourceProfile",
    "CpuProfile",
    "GcProfile",
    "MemoryProfile",
    "IoProfile",
    "ContentionProfile",
    "build_resource_profile",
    # JSHD loading
    "JshdMetadata",
    "JshdData",
    "load_jshd",
    "plot_jshd",
    # Config parser
    "parse_jshc",
    "parse_jshc_content",
    # Debug log parsing
    "DebugMessage",
    "DebugMessageStore",
    "DebugSummary",
    "parse_debug_line",
    "load_debug_file",
    "load_debug_from_script",
    "format_message",
    # Jobs (optional)
    "ConfigSweepParameter",
    "FileSweepParameter",
    "CompoundSweepParameter",
    "SweepConfig",
    "JobConfig",
    "ExpandedJob",
    "JobSet",
    "JobExpander",
    "SweepResult",
    "AdaptiveSweepResult",
    "to_run_config",
    "to_run_remote_config",
    "run_sweep",
    "discover_jshd_files",
    "GridSpec",
    "HAS_JOBS",
    # Strategies (optional, part of jobs)
    "SweepStrategy",
    "CartesianStrategy",
    "OptunaStrategy",
    "ObjectiveFn",
    "strategy_from_dict",
    "sample_params_from_trial",
    "run_adaptive_sweep",
    "SweepExecutionError",
    "EXIT_CODE_DIAGNOSTICS",
    "get_exit_code_diagnostic",
    "cv_objective",
    # Registry (optional)
    "RunRegistry",
    "RegistryCallback",
    "SessionInfo",
    "ConfigInfo",
    "RunInfo",
    "SessionSummary",
    "DataSummary",
    "configure_s3",
    "CellDataLoader",
    "DiagnosticQueries",
    "SimulationDiagnostics",
    "HAS_REGISTRY",
    # Sweep (optional)
    "SweepManager",
    "SweepManagerBuilder",
    "recover_sweep_results",
    "load_job_results",
    "ingest_results",
    "LoadConfig",
    "ResultLoadError",
    "HAS_SWEEP",
    # Catalog (optional)
    "ProjectCatalog",
    "ExperimentInfo",
    "ModelInfo",
    "DataManifestInfo",
    "compute_model_hash",
    "compute_config_hash",
    "compute_data_manifest_hash",
    "HAS_CATALOG",
]
