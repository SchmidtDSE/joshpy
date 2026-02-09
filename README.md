# joshpy

Python client for [Josh](https://joshsim.org) ecological simulation runtime.

## Installation

```bash
pip install joshpy
```

For parameter sweep functionality with Jinja templating:

```bash
pip install joshpy[jobs]
```

For all optional dependencies:

```bash
pip install joshpy[all]
```

## Quick Start

### Running Simulations via HTTP

```python
from joshpy import Josh

client = Josh(server="https://your-josh-server.com", api_key="your-key")

code = """
start simulation Main
  grid.size = 1000 m
  grid.low = 33.7 degrees latitude, -115.4 degrees longitude
  grid.high = 34.0 degrees latitude, -116.4 degrees longitude
  steps.low = 0 count
  steps.high = 10 count
end simulation
"""

# Check for errors
error = client.get_error(code)
if error:
    print(f"Error: {error}")

# Get simulation metadata
metadata = client.get_metadata(code, "Main")
print(f"Grid size: {metadata.patch_size}")

# Run simulation
results = client.run_simulation(code, "Main", virtual_files=[], replicates=3)
```

### Parameter Sweeps with Jinja Templating

Generate multiple configuration files from a single template:

```python
from pathlib import Path
from joshpy.jobs import JobConfig, SweepConfig, SweepParameter, JobExpander, run_sweep
from joshpy.cli import JoshCLI
from joshpy.registry import RunRegistry, RegistryCallback

# Define sweep configuration
config = JobConfig(
    template_path=Path("templates/sweep_config.jshc.j2"),
    source_path=Path("simulation.josh"),
    simulation="Main",
    sweep=SweepConfig(
        parameters=[
            SweepParameter(name="survivalProb", values=[85, 90, 95]),
            SweepParameter(name="seedCount", values=[1000, 2000, 4000]),
        ]
    ),
)

# Expand to 3x3 = 9 concrete jobs
expander = JobExpander()
job_set = expander.expand(config)
print(f"Will run {job_set.total_jobs} jobs ({job_set.total_replicates} replicates)")

# Setup tracking (optional)
registry = RunRegistry("experiment.duckdb")
session_id = registry.create_session(experiment_name="my_sweep")
callback = RegistryCallback(registry, session_id)

# Execute with run_sweep()
cli = JoshCLI()
results = run_sweep(cli, job_set, callback=callback.record)
print(f"Completed: {results.succeeded} succeeded, {results.failed} failed")
```

### Direct CLI Usage

Use the CLI wrapper directly for individual commands:

```python
from pathlib import Path
from joshpy.cli import JoshCLI, RunConfig, PreprocessConfig, ValidateConfig

cli = JoshCLI()

# Validate a Josh script
result = cli.validate(ValidateConfig(script=Path("simulation.josh")))
if not result.success:
    print(f"Validation failed: {result.stderr}")

# Preprocess external data
result = cli.preprocess(PreprocessConfig(
    script=Path("simulation.josh"),
    simulation="Main",
    data_file=Path("temperature.nc"),
    variable="temp",
    units="K",
    output=Path("temperature.jshd"),
))

# Run a simulation
result = cli.run(RunConfig(
    script=Path("simulation.josh"),
    simulation="Main",
    replicates=5,
    data={"climate": Path("temperature.jshd")},
    output=Path("results.csv"),
    output_format="csv",
))
```

### YAML Configuration

Define sweeps in YAML for easier management:

```yaml
# sweep_config.yaml
template_path: templates/sweep_config.jshc.j2
source_path: simulation.josh
simulation: Main
replicates: 3

sweep:
  parameters:
    # Explicit values
    - name: scenario
      values: [baseline, optimistic, pessimistic]

    # Range with step (like numpy.arange)
    - name: survivalProb
      range: {start: 80, stop: 100, step: 5}

    # Range with count (like numpy.linspace)
    - name: seedCount
      range: {start: 1000, stop: 5000, num: 5}
```

Load and run:

```python
from joshpy.jobs import JobExpander, to_run_config
from joshpy.cli import JoshCLI

config = JobConfig.from_yaml_file(Path("sweep_config.yaml"))
job_set = JobExpander().expand(config)

cli = JoshCLI()
for job in job_set:
    result = cli.run(to_run_config(job))
    print(f"[{'OK' if result.success else 'FAIL'}] {job.parameters}")
```

### Diagnostic Plotting

Quick visualization for simulation sanity checks:

```python
from joshpy.registry import RunRegistry
from joshpy.cell_data import CellDataLoader
from joshpy.diagnostics import SimulationDiagnostics

# Load simulation outputs into registry
registry = RunRegistry("experiment.duckdb")
loader = CellDataLoader(registry)
loader.load_csv(Path("output.csv"), run_id, config_hash)

# Discover what's available
print(registry.get_data_summary())
print(f"Export variables: {registry.list_export_variables()}")
print(f"Config parameters: {registry.list_config_parameters()}")

# Create diagnostics helper
diag = SimulationDiagnostics(registry)

# Time series with uncertainty bands
diag.plot_timeseries("averageAge", config_hash="abc123")

# Compare across parameter sweep (group_by auto-detects config params vs export vars)
diag.plot_comparison("averageAge", group_by="maxGrowth")  # config parameter
diag.plot_comparison("height", group_by="FIRE_REGIME")    # export variable

# Spatial snapshot
diag.plot_spatial("treeCount", step=50, config_hash="abc123")

# Show SQL for learning/debugging
diag.plot_comparison("averageAge", group_by="maxGrowth", show_sql=True)
```

## Development

### Using DevContainer (Recommended)

Open in VS Code with the Dev Containers extension, or use GitHub Codespaces.

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/SchmidtDSE/joshpy.git
cd joshpy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with development dependencies
pip install -e ".[all,dev]"

# Run tests
python -m pytest joshpy/test_*.py -v

# Type checking
mypy joshpy/

# Linting
ruff check joshpy/
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

## Links

- [Josh Language Documentation](https://joshsim.org)
- [Josh GitHub Repository](https://github.com/SchmidtDSE/josh)
- [Issue Tracker](https://github.com/SchmidtDSE/joshpy/issues)
