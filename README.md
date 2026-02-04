# Genesis-1M: The Million-Agent Civilization Engine

![Status](https://img.shields.io/badge/Status-Phase_1_Complete-yellow) ![Agents](https://img.shields.io/badge/Scale-1_Million_Agents-blue) ![Compute](https://img.shields.io/badge/Target_Compute-500x_H100s-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **"I'm compressing 10,000 years of human history into a single weekend."**

## The Vision

Genesis-1M is a massively parallel **Computational Sociology** experiment. Unlike traditional RL environments (which simulate one agent in parallel 1M times), Genesis simulates **1 million unique agents in a single shared world**.

The goal is to observe **emergent history**:
* Can AI agents invent language from scratch?
* Will they discover agriculture, or starve?
* Will they form democracies, or tyrannies?

We provide the physics and the biological imperatives (hunger, reproduction, death). The agents provide the history.

## Architecture: The "Spotlight" Engine

Simulating 1M LLM-powered agents is impossible with brute force. We solve this using a **Level-of-Detail (LOD) Consciousness** system, similar to how video games render geometry.

| Tier | State | Tech Stack | Intelligence | % of Pop |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Active** | Llama-3-8B (vLLM) | Full reasoning, dialogue, planning | ~0.05% |
| **Tier 2** | **Reactive** | Weighted Heuristics | Personality-biased decisions | ~5.0% |
| **Tier 3** | **Background** | Numpy FSM | Move, Eat, Sleep, Wander | ~94.9% |

* **Dynamic Promotion:** When a Tier 3 agent enters a high-stakes situation (e.g., meeting a stranger, starvation), they are dynamically "promoted" to Tier 1 for that interaction.
* **Distributed Backend:** The world is sharded across a **Ray Cluster** using spatial hashing, targeting deployment on 500+ Nvidia H100 GPUs.

## Tech Stack

* **Core Logic:** Python 3.10+, Ray (Distributed Actors)
* **Inference:** vLLM (Batch inference for Tier 1)
* **Memory:** LanceDB (Vector storage for agent history)
* **Physics:** Custom Numpy-based 2D world with spatial hash partitioning

## Project Status

### Phase 1: MVP "Ant Farm" ✅
- [x] 100,000+ agents spawn, move, and interact
- [x] Spatial hash grid for O(1) neighbor queries
- [x] Agents eat, lose energy, and die
- [x] Biological reproduction with genetic inheritance
- [x] FSM states: WANDER, SEEK_FOOD, FLEE, SEEK_MATE, REST
- [x] Event logging (births, deaths, eating)
- [x] Checkpointing (save/restore simulation state)
- [x] HPC deployment scripts (PACE ICE)

### Phase 2: LLM Integration 🔄
- [ ] vLLM integration for Tier 1 agents
- [ ] Promotion scoring system
- [ ] Batch inference pipeline

### Phase 3: Memory Stream
- [ ] LanceDB integration
- [ ] Agent memory formation and recall
- [ ] Generational knowledge transfer

## Getting Started

### Local Development

```bash
# Clone the repo
git clone https://github.com/dhruv-2604/Genesis-1M.git
cd Genesis-1M

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run simulation (1000 agents, 100 ticks)
python run_simulation.py --agents 1000 --ticks 100

# Run benchmark
python scripts/benchmark.py
```

### HPC Deployment (PACE ICE)

```bash
# Sync code to cluster
bash scripts/hpc/sync_to_hpc.sh YOUR_GT_USERNAME

# On HPC: Setup environment (first time)
ssh YOUR_USERNAME@login-ice.pace.gatech.edu
cd ~/scratch/agent-sim-code
bash scripts/hpc/setup_env.sh

# Submit jobs
sbatch scripts/hpc/run_simulation.sbatch        # 1 H100, 100K agents
sbatch scripts/hpc/run_multinode.sbatch         # 32 H100s, 1M agents
```

## Project Structure

```
genesis-1m/
├── config/
│   ├── defaults.yaml       # Local dev config
│   └── hpc.yaml            # HPC optimized config
├── scripts/
│   ├── benchmark.py        # Performance testing
│   ├── analyze_events.py   # Event log analysis
│   └── hpc/                # SLURM job scripts
├── src/
│   ├── agents/tier3.py     # FSM logic, genetics
│   ├── core/
│   │   ├── simulation.py   # Main tick loop
│   │   ├── state.py        # AgentState, WorldState
│   │   ├── events.py       # Event logging
│   │   ├── checkpoint.py   # Save/restore
│   │   └── distributed.py  # Ray actors
│   ├── inference/          # LLM integration (Phase 2)
│   ├── memory/             # Vector DB (Phase 3)
│   ├── spatial/hash_grid.py # O(1) neighbor queries
│   └── world/
│       ├── terrain.py      # Perlin noise terrain
│       └── resources.py    # Food, tools, materials
├── tests/                  # 49 unit/integration tests
├── run_simulation.py       # Main entry point
└── requirements.txt
```

## Performance

Single-threaded Python (M1 Mac):
| Agents | TPS |
|--------|-----|
| 1,000  | 55  |
| 10,000 | 25  |
| 100,000| 3   |

Target with Ray cluster (32× H100): **200-500 TPS with 1M agents**

## License

MIT
