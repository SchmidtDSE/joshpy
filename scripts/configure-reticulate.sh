#!/bin/bash
# Configure R's reticulate package to use pixi's Python environment
# This enables Quarto docs with Python code blocks to work correctly

set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX not set. Run this via 'pixi run configure-reticulate'"
    exit 1
fi

# Create conda-meta/history file that reticulate checks to identify conda envs
mkdir -p "$CONDA_PREFIX/conda-meta"
echo '# cmd: conda' > "$CONDA_PREFIX/conda-meta/history"

# Add RETICULATE_PYTHON to bashrc if not already present
if ! grep -q 'RETICULATE_PYTHON' ~/.bashrc 2>/dev/null; then
    echo "export RETICULATE_PYTHON=\"$CONDA_PREFIX/bin/python\"" >> ~/.bashrc
    echo "Added RETICULATE_PYTHON to ~/.bashrc"
else
    echo "RETICULATE_PYTHON already in ~/.bashrc"
fi

echo "Reticulate configured to use: $CONDA_PREFIX/bin/python"
echo "Restart your shell or run: source ~/.bashrc"
