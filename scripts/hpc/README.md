# PACE ICE HPC Deployment Guide

## Quick Start

### 1. Sync code to HPC
```bash
# From local machine
bash scripts/hpc/sync_to_hpc.sh YOUR_GT_USERNAME
```

### 2. Setup environment (first time only)
```bash
ssh YOUR_USERNAME@login-ice.pace.gatech.edu
cd ~/scratch/agent-sim-code
bash scripts/hpc/setup_env.sh
```

### 3. Run simulation
```bash
# Single node (1 H100, 100K agents)
sbatch scripts/hpc/run_simulation.sbatch

# Multi-node (8 nodes × 4 H100s = 32 GPUs, 1M agents)
sbatch scripts/hpc/run_multinode.sbatch

# Custom agent count
AGENTS=500000 TICKS=50000 sbatch scripts/hpc/run_simulation.sbatch
```

### 4. Monitor jobs
```bash
squeue -u $USER              # Check job status
tail -f logs/sim_<job_id>.out  # Watch output
scancel <job_id>             # Cancel job
```

## Job Scripts

| Script | GPUs | Agents | Use Case |
|--------|------|--------|----------|
| `run_simulation.sbatch` | 1 H100 | 100K | Testing, small runs |
| `run_multinode.sbatch` | 32 H100 | 1M | Full-scale simulation |
| `run_benchmark.sbatch` | 1 H100 | Various | Performance testing |

## Interactive Development
```bash
bash scripts/hpc/interactive.sh h100  # Get interactive H100 session
```

## Storage

- **Code**: `~/scratch/agent-sim-code/`
- **Checkpoints**: `~/scratch/agent-sim/checkpoints/`
- **Event logs**: `~/scratch/agent-sim/events/`
- **Job logs**: `logs/` (in code directory)

## Scaling Guide

| Nodes | GPUs | Recommended Agents | Expected TPS |
|-------|------|-------------------|--------------|
| 1 | 1 H100 | 100K | ~50-100 |
| 1 | 4 H100 | 250K | ~100-200 |
| 8 | 32 H100 | 1M | ~200-500 |

## Troubleshooting

**Job pending forever**
```bash
squeue -u $USER -t PENDING  # Check why
# Usually: waiting for GPU resources
```

**Out of memory**
- Reduce `AGENTS` count
- Increase `--mem` in SBATCH script

**Module not found**
```bash
module load anaconda3 cuda
conda activate agent-sim
```

**Ray connection failed**
- Check firewall between nodes
- Verify head node IP is correct
