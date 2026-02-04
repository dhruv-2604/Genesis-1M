#!/usr/bin/env python3
"""
Benchmark script for measuring simulation performance.

Tests different population sizes and measures tick rate.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulation import Simulation
from src.config import SimConfig


def benchmark(agent_count: int, num_ticks: int = 100) -> dict:
    """Run benchmark with given agent count"""
    print(f"\nBenchmarking with {agent_count:,} agents...")

    config = SimConfig()
    config.WORLD_SIZE = 10000.0
    config.CELL_SIZE = 50.0
    config.INITIAL_AGENT_COUNT = agent_count
    config.MAX_AGENTS = int(agent_count * 1.5)
    config.CHECKPOINT_DIR = f"/tmp/bench_{agent_count}/checkpoints"
    config.EVENT_LOG_DIR = f"/tmp/bench_{agent_count}/events"
    config.CHECKPOINT_INTERVAL = 0  # Disable
    config.LOG_INTERVAL = 10000  # Reduce logging

    # Initialize
    init_start = time.perf_counter()
    sim = Simulation(config=config)
    init_time = time.perf_counter() - init_start

    # Run ticks
    tick_times = []
    for i in range(num_ticks):
        start = time.perf_counter()
        sim.tick()
        tick_times.append(time.perf_counter() - start)

        if (i + 1) % 10 == 0:
            avg = sum(tick_times[-10:]) / 10 * 1000
            print(f"  Tick {i+1}: {avg:.1f}ms avg (last 10)")

    avg_tick = sum(tick_times) / len(tick_times) * 1000
    tps = 1000 / avg_tick if avg_tick > 0 else 0

    return {
        'agent_count': agent_count,
        'num_ticks': num_ticks,
        'init_time_s': init_time,
        'avg_tick_ms': avg_tick,
        'min_tick_ms': min(tick_times) * 1000,
        'max_tick_ms': max(tick_times) * 1000,
        'ticks_per_second': tps,
        'final_population': sim.world_state.population,
        'total_births': sim.world_state.total_births,
        'total_deaths': sim.world_state.total_deaths,
    }


def main():
    print("=" * 60)
    print("Agent Simulation Benchmark")
    print("=" * 60)

    # Test different scales
    results = []

    for count in [1000, 10000, 50000, 100000]:
        result = benchmark(count, num_ticks=50)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Agents':>10} | {'Init (s)':>10} | {'Avg Tick (ms)':>15} | {'TPS':>10}")
    print("-" * 60)

    for r in results:
        print(f"{r['agent_count']:>10,} | {r['init_time_s']:>10.2f} | {r['avg_tick_ms']:>15.1f} | {r['ticks_per_second']:>10.1f}")

    print("\n" + "=" * 60)

    # Check if we meet MVP target (100 TPS with Tier 3 only)
    for r in results:
        status = "✓" if r['ticks_per_second'] >= 100 else "✗"
        print(f"{r['agent_count']:,} agents: {r['ticks_per_second']:.1f} TPS {status}")


if __name__ == '__main__':
    main()
