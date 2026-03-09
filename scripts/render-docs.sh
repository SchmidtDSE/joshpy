#!/bin/bash
# Render documentation
# This script uses Quarto's freeze cache by default for fast incremental builds.
# For a full clean build, run: pixi run docs-clean && pixi run docs-build

set -e

# Setup for R/reticulate
echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

# Clean only output site (preserve _freeze for caching)
echo "=== Cleaning output site ==="
rm -rf docs/_site
rm -f docs/tutorials/*.duckdb docs/tutorials/*.duckdb.wal

# Regenerate API reference (these are generated files)
echo "=== Building API reference ==="
rm -f docs/reference/*.qmd
python -m quartodoc build --config docs/_quarto.yml

# Render the full site (uses _freeze cache when available)
echo "=== Rendering site ==="
quarto render docs

# Ensure .nojekyll exists for GitHub Pages (prevents ignoring _ prefixed dirs)
touch docs/_site/.nojekyll

echo "=== Done ==="
