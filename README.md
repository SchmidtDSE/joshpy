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
from joshpy.jobs import JobConfig, SweepConfig, SweepParameter, JobExpander, JobRunner

# Define sweep configuration
config = JobConfig(
    template_path=Path("templates/editor.jshc.j2"),
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

# Execute via Josh CLI
runner = JobRunner(josh_jar=Path("joshsim-fat.jar"))
results = runner.run_all(job_set)

for result in results:
    status = "OK" if result.success else "FAIL"
    print(f"[{status}] {result.job.parameters}")
```

### YAML Configuration

Define sweeps in YAML for easier management:

```yaml
# sweep_config.yaml
template_path: templates/editor.jshc.j2
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
config = JobConfig.from_yaml_file(Path("sweep_config.yaml"))
job_set = JobExpander().expand(config)
results = JobRunner(josh_jar=Path("joshsim.jar")).run_all(job_set)
```

### Convenience Function

For simple sweeps:

```python
from joshpy.jobs import run_sweep

results = run_sweep(
    template="maxGrowth = {{ maxGrowth }} meters",
    source=Path("simulation.josh"),
    parameters={"maxGrowth": [1, 5, 10]},
    josh_jar=Path("joshsim.jar"),
)
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
