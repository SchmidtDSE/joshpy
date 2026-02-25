#!/bin/bash
# Render documentation
# With freeze: true in _quarto.yml, cached results from _freeze/ are used

set -e

# Setup for R/reticulate
echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

# Build API reference first
echo "=== Building API reference ==="
python -m quartodoc build --config docs/_quarto.yml

# Render the full site (freeze: true means cached results are used)
echo "=== Rendering full site ==="
quarto render docs

# Ensure .nojekyll exists for GitHub Pages (prevents ignoring _ prefixed dirs)
touch docs/_site/.nojekyll

echo "=== Done ==="
