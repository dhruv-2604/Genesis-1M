#!/usr/bin/env python3
"""
Distributed simulation runner for Ray cluster on PACE ICE.
"""

import argparse
import time
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ray
from src.config import SimConfig


def main():
    parser = argparse.ArgumentParser(description='Distributed Agent Simulation')
    parser.add_argument('--agents', type=int, default=1000000)
    parser.add_argument('--ticks', type=int, default=100000)
    parser.add_argument('--ray-address', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Ray cluster at {args.ray_address}...")
    ray.init(address=args.ray_address)

    print(f"Ray cluster resources:")
    print(f"  Nodes: {len(ray.nodes())}")
    print(f"  CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"  GPUs: {ray.cluster_resources().get('GPU', 0)}")
    print()

    # Configure simulation
    config = SimConfig()
    config.INITIAL_AGENT_COUNT = args.agents
    config.MAX_AGENTS = int(args.agents * 1.2)
    config.WORLD_SIZE = 50000.0  # Large world for 1M agents
    config.CELL_SIZE = 100.0
    config.CHECKPOINT_DIR = str(output_dir / "checkpoints")
    config.EVENT_LOG_DIR = str(output_dir / "events")
    config.CHECKPOINT_INTERVAL = 10000
    config.LOG_INTERVAL = 1000

    print(f"Configuration:")
    print(f"  Agents: {config.INITIAL_AGENT_COUNT:,}")
    print(f"  World size: {config.WORLD_SIZE:,}")
    print(f"  Ticks: {args.ticks:,}")
    print()

    # Import distributed simulation
    from src.core.distributed import DistributedSimulation

    # Calculate actors based on available resources
    num_actors = int(ray.cluster_resources().get('CPU', 8) // 4)  # 4 CPUs per actor
    num_actors = max(8, min(num_actors, 64))  # 8-64 actors

    print(f"Starting distributed simulation with {num_actors} actors...")
    sim = DistributedSimulation(config=config, num_actors=num_actors)

    # Initialize with agents
    from src.core.state import AgentState, GeneIndex
    import numpy as np

    print("Spawning initial agents...")
    rng = np.random.default_rng(config.SEED)
    agents = []
    for i in range(config.INITIAL_AGENT_COUNT):
        agent = AgentState(
            id=i,
            x=rng.uniform(0, config.WORLD_SIZE),
            y=rng.uniform(0, config.WORLD_SIZE),
            energy=config.STARTING_ENERGY,
            genes=rng.uniform(0.5, 1.5, GeneIndex.NUM_GENES).astype(np.float32)
        )
        agents.append(agent)

        if (i + 1) % 100000 == 0:
            print(f"  Created {i + 1:,} agents...")

    print("Distributing agents to actors...")
    sim.initialize(agents)

    print()
    print("Starting simulation loop...")
    print("=" * 60)

    start_time = time.time()
    tick_times = []

    for tick in range(args.ticks):
        tick_start = time.perf_counter()
        stats = sim.step()
        tick_time = (time.perf_counter() - tick_start) * 1000
        tick_times.append(tick_time)

        if tick % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg_tick = sum(tick_times[-1000:]) / len(tick_times[-1000:])
            tps = 1000 / avg_tick if avg_tick > 0 else 0

            print(f"[Tick {tick:,}] Pop: {stats['population']:,} | "
                  f"B: {stats['births']} D: {stats['deaths']} | "
                  f"{avg_tick:.1f}ms ({tps:.1f} TPS) | "
                  f"Elapsed: {elapsed/60:.1f}min")

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total ticks: {args.ticks:,}")
    print(f"Final population: {sim.get_population():,}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average tick time: {sum(tick_times)/len(tick_times):.1f}ms")
    print(f"Average TPS: {len(tick_times) / (total_time) if total_time > 0 else 0:.1f}")

    # Shutdown
    sim.shutdown()
    ray.shutdown()


if __name__ == '__main__':
    main()
