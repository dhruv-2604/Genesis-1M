#!/bin/bash
#SBATCH -J agent-sim-multi
#SBATCH -p ice-gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -o logs/multi_%j.out
#SBATCH -e logs/multi_%j.err

# Agent Simulation - Multi-Node Ray Cluster
# Submit: sbatch scripts/hpc/run_multinode.sbatch
#
# This runs a Ray cluster across 8 nodes with 4 H100s each (32 GPUs total)

set -e

echo "=========================================="
echo "Multi-Node Agent Simulation - Job $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module load anaconda3 cuda
conda activate agent-sim

# Get node list
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODELIST)
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

echo ""
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Worker nodes: ${NODES_ARRAY[@]:1}"
echo ""

# Start Ray head node
echo "Starting Ray head on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address=$HEAD_NODE_IP \
    --port=6379 --num-cpus=32 --num-gpus=4 \
    --dashboard-host=0.0.0.0 --block &

sleep 10

# Start Ray workers on other nodes
for NODE in "${NODES_ARRAY[@]:1}"; do
    echo "Starting Ray worker on $NODE..."
    srun --nodes=1 --ntasks=1 -w "$NODE" \
        ray start --address=$HEAD_NODE_IP:6379 \
        --num-cpus=32 --num-gpus=4 --block &
done

sleep 30

# Run distributed simulation
echo ""
echo "Starting distributed simulation..."

# Configuration
AGENTS=${AGENTS:-1000000}
TICKS=${TICKS:-100000}

# Copy and run on head node
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    cd $SLURM_SUBMIT_DIR
    python scripts/hpc/run_distributed.py \
        --agents $AGENTS \
        --ticks $TICKS \
        --ray-address $HEAD_NODE_IP:6379 \
        --output-dir ~/scratch/agent-sim/run_$SLURM_JOB_ID
"

# Shutdown Ray
echo ""
echo "Shutting down Ray cluster..."
ray stop

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
