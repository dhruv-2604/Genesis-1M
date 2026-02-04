# Genesis-1M: The Million-Agent Civilization Engine

![Status](https://img.shields.io/badge/Status-Pre--Alpha-red) ![Agents](https://img.shields.io/badge/Scale-1_Million_Agents-blue) ![Compute](https://img.shields.io/badge/Target_Compute-500x_H100s-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **"I'm compressing 10,000 years of human history into a single weekend."**

## The Vision
Genesis-1M is a massively parallel **Computational Sociology** experiment. Unlike traditional RL environments (which simulate one agent in parallel 1M times), Genesis simulates **1 million agents in a single shared world**.

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
| **Tier 2** | **Reactive** | TinyLlama / SLM | Basic trade, flight, greetings | ~5.0% |
| **Tier 3** | **Background** | Numpy FSM | Move, Eat, Sleep, Wander | ~94.9% |

* **Dynamic Promotion:** When a Tier 3 agent enters a high-stakes situation (e.g., meeting a stranger, starvation), they are dynamically "promoted" to Tier 1 for that interaction.
* **Distributed Backend:** The world is sharded across a **Ray Cluster** using spatial hashing, targeting deployment on 500+ Nvidia H100 GPUs.

## Tech Stack
* **Core Logic:** Python 3.10+, Ray (Distributed Actors)
* **Inference:** vLLM (Batch inference for Tier 1)
* **Memory:** LanceDB (Vector storage for agent history)
* **Physics:** Custom Numpy-based 2D grid with spatial partitioning

## Getting Started (Dev Mode)

You can run a mini-version of Genesis (10k agents) on a single consumer laptop.

```bash
# Clone the repo
git clone [https://github.com/yourusername/genesis-1m.git](https://github.com/yourusername/genesis-1m.git)
cd genesis-1m

# Install dependencies
pip install -r requirements.txt

# Run the headless simulation (ASCII output)
python src/core/simulation.py --agents 10000 --dev-mode
