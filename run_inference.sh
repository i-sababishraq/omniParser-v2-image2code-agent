#!/bin/bash

# Set PYTHONPATH to project root
export PYTHONPATH=/anvil/projects/x-soc250046/x-sishraq/omniparser-v2

# Use the full path to conda env python
CONDA_PYTHON=/anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni/bin/python

echo "Using Python: $CONDA_PYTHON"
$CONDA_PYTHON --version
echo "PYTHONPATH: $PYTHONPATH"

# Run the inference script
$CONDA_PYTHON scripts/run_samples.py
