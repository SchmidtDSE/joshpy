#!/bin/bash
# Render documentation in correct order
# Tutorials must run sequentially since analysis.qmd depends on data from earlier tutorials

set -e

# Setup for R/reticulate
echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

# Clean up any corrupted DuckDB files from previous runs
echo "=== Cleaning up previous artifacts ==="
rm -f docs/tutorials/demo_registry.duckdb*

# Build API reference first
echo "=== Building API reference ==="
python -m quartodoc build --config docs/_quarto.yml

# Render quickstart (standalone, no dependencies)
echo "=== Rendering quickstart.qmd ==="
quarto render docs/getting-started/quickstart.qmd

# Render tutorials in order (they create demo_registry.duckdb)
echo "=== Rendering manual-workflow.qmd ==="
quarto render docs/tutorials/manual-workflow.qmd

echo "=== Rendering sweep-manager.qmd ==="
quarto render docs/tutorials/sweep-manager.qmd

echo "=== Rendering analysis.qmd ==="
quarto render docs/tutorials/analysis.qmd

# Render the rest of the site
echo "=== Rendering full site ==="
quarto render docs

echo "=== Done ==="
