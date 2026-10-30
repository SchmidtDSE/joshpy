#!/bin/bash
# Render (execute) the Inference pages, in order.
#
# These pages need joshpy[sbi] and the Josh JAR, so docs CI only uses their
# committed freezes (docs/_freeze/inference/). The first model's pages share
# the registry docs/inference/forevertree.duckdb, and the fire model's pages
# docs/inference/forevertree_fire.duckdb: later pages reuse earlier pages'
# simulations instead of re-simulating. Delete a registry to start from scratch.
#
# The fire model's pages run thousands of simulations. Set JOSH_BATCH_TARGET to
# a batch target profile (for example gke-test) to run them there; they read
# MinIO keys from the repository's .env. With the registry in place, they
# replay without a target.

set -e

echo '# cmd: conda' > $CONDA_PREFIX/conda-meta/history
export RETICULATE_PYTHON=$CONDA_PREFIX/bin/python

PAGES=(
  docs/inference/forevertree.qmd
  docs/inference/smc-abc.qmd
  docs/inference/fire-model.qmd
)

for page in "${PAGES[@]}"; do
  echo "=== Rendering $page ==="
  quarto render "$page"
done
