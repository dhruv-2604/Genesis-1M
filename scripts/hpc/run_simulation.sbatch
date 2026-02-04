#!/bin/bash
#SBATCH -J agent-sim
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/sim_%j.out
#SBATCH -e logs/sim_%j.err

# Agent Simulation - Single Node Run
# Submit: sbatch scripts/hpc/run_simulation.sbatch

set -e

echo "=========================================="
echo "Agent Simulation - Job $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module load anaconda3 cuda
conda activate agent-sim

# Create output directories
mkdir -p logs
mkdir -p ~/scratch/agent-sim/checkpoints
mkdir -p ~/scratch/agent-sim/events

# Copy code to fast local storage
echo "Copying code to $TMPDIR..."
cp -r $SLURM_SUBMIT_DIR $TMPDIR/agent-sim
cd $TMPDIR/agent-sim

# Configuration
AGENTS=${AGENTS:-100000}
TICKS=${TICKS:-10000}
SEED=${SEED:-42}

echo ""
echo "Configuration:"
echo "  Agents: $AGENTS"
echo "  Ticks: $TICKS"
echo "  Seed: $SEED"
echo ""

# Run simulation
python run_simulation.py \
    --agents $AGENTS \
    --ticks $TICKS \
    --seed $SEED \
    --config config/hpc.yaml

# Copy results back to scratch
echo ""
echo "Copying results to ~/scratch/agent-sim..."
cp -r checkpoints/* ~/scratch/agent-sim/checkpoints/ 2>/dev/null || true
cp -r logs/events/* ~/scratch/agent-sim/events/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
