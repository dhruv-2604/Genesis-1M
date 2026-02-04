#!/bin/bash
#SBATCH -J agent-bench
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -o logs/bench_%j.out
#SBATCH -e logs/bench_%j.err

# Agent Simulation Benchmark
# Submit: sbatch scripts/hpc/run_benchmark.sbatch

set -e

echo "=========================================="
echo "Agent Simulation Benchmark - Job $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

module load anaconda3 cuda
conda activate agent-sim

cd $SLURM_SUBMIT_DIR

# Run benchmark
python scripts/benchmark.py

echo ""
echo "Benchmark complete!"
