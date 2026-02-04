#!/bin/bash
# Setup conda environment on PACE ICE
# Run once: bash scripts/hpc/setup_env.sh

set -e

echo "Setting up agent-sim environment on PACE ICE..."

# Load modules
module load anaconda3 cuda

# Create conda environment
conda create -n agent-sim python=3.10 -y
conda activate agent-sim

# Install dependencies
pip install numpy pyyaml pytest

# Install Ray for distributed execution
pip install 'ray[default]>=2.9.0'

# Install vLLM for LLM inference (Phase 2)
# Note: Requires CUDA 12.1+ on H100
pip install vllm

# Install sentence-transformers for embeddings (Phase 3)
pip install sentence-transformers lancedb

echo ""
echo "Environment setup complete!"
echo "Activate with: conda activate agent-sim"
