#!/bin/bash
# Deploy simulation to Ray cluster
# Usage: ./deploy_cluster.sh [NUM_GPUS]

set -e

NUM_GPUS=${1:-8}

echo "======================================"
echo "Agent Simulation Cluster Deployment"
echo "======================================"
echo "GPUs requested: $NUM_GPUS"
echo ""

# Check Ray is installed
if ! command -v ray &> /dev/null; then
    echo "Error: Ray is not installed"
    echo "Install with: pip install 'ray[default]>=2.9.0'"
    exit 1
fi

# Check if Ray cluster is running
if ray status 2>/dev/null | grep -q "Ray is running"; then
    echo "Ray cluster already running"
else
    echo "Starting Ray cluster..."
    ray start --head --num-gpus=$NUM_GPUS --dashboard-host=0.0.0.0
fi

# Display cluster info
echo ""
echo "Cluster status:"
ray status

echo ""
echo "To run simulation on cluster:"
echo "  python run_simulation.py --config config/cluster.yaml"
echo ""
echo "To monitor:"
echo "  Ray Dashboard: http://localhost:8265"
echo ""
echo "To stop cluster:"
echo "  ray stop"
