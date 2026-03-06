#!/bin/bash
# Render documentation
# This script does a FULL clean build by default.
# For fast cached builds, use: pixi run docs-preview-fast

set -e

# Setup for R/reticulate
echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

# Clean previous build artifacts to ensure fresh state
echo "=== Cleaning previous build artifacts ==="
rm -rf docs/_site docs/_freeze
rm -f docs/tutorials/*.duckdb docs/tutorials/*.duckdb.wal
rm -f docs/reference/*.qmd

# Build API reference first
echo "=== Building API reference ==="
python -m quartodoc build --config docs/_quarto.yml

# Render the full site
echo "=== Rendering full site ==="
quarto render docs

# Ensure .nojekyll exists for GitHub Pages (prevents ignoring _ prefixed dirs)
touch docs/_site/.nojekyll

echo "=== Done ==="
