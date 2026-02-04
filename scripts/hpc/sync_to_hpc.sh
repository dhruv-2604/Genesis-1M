#!/bin/bash
# Sync code to PACE ICE
# Usage: bash scripts/hpc/sync_to_hpc.sh <username>

USERNAME=${1:-$USER}
HPC_HOST="login-ice.pace.gatech.edu"
REMOTE_DIR="~/scratch/agent-sim-code"

echo "Syncing to $USERNAME@$HPC_HOST:$REMOTE_DIR"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'checkpoints' \
    --exclude 'logs/events' \
    --exclude '.pytest_cache' \
    --exclude '*.egg-info' \
    . $USERNAME@$HPC_HOST:$REMOTE_DIR/

echo ""
echo "Sync complete!"
echo ""
echo "Next steps on HPC:"
echo "  1. ssh $USERNAME@$HPC_HOST"
echo "  2. cd $REMOTE_DIR"
echo "  3. bash scripts/hpc/setup_env.sh  # First time only"
echo "  4. sbatch scripts/hpc/run_simulation.sbatch"
