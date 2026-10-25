#!/bin/bash
# Render (execute) the Inference pages, in order.
#
# These pages need joshpy[sbi] and the Josh JAR, so docs CI only uses their
# committed freezes (docs/_freeze/inference/). They share the registry
# docs/inference/forevertree.duckdb: later pages reuse the case study's
# simulations instead of re-simulating. Delete it to start from scratch.

set -e

echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

PAGES=(
  docs/inference/forevertree.qmd
  docs/inference/smc-abc.qmd
)

for page in "${PAGES[@]}"; do
  echo "=== Rendering $page ==="
  quarto render "$page"
done
