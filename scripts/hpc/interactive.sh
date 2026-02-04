#!/bin/bash
# Start an interactive GPU session for development/debugging
# Usage: bash scripts/hpc/interactive.sh [gpu_type]
#   gpu_type: v100, a100, h100 (default: h100)

GPU_TYPE=${1:-h100}
MEM="32G"
TIME="02:00:00"

case $GPU_TYPE in
    v100)
        GRES="gpu:v100:1"
        ;;
    a100)
        GRES="gpu:a100:1"
        ;;
    h100)
        GRES="gpu:h100:1"
        MEM="64G"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        echo "Options: v100, a100, h100"
        exit 1
        ;;
esac

echo "Requesting interactive session:"
echo "  GPU: $GPU_TYPE"
echo "  Memory: $MEM"
echo "  Time: $TIME"
echo ""

srun -p ice-gpu --gres=$GRES --mem=$MEM --time=$TIME --cpus-per-task=8 --pty bash
