#!/usr/bin/env python3
"""
Main entry point for running the agent simulation.

Usage:
    python run_simulation.py                    # Run with defaults
    python run_simulation.py --ticks 10000      # Run for 10000 ticks
    python run_simulation.py --config my.yaml   # Use custom config
    python run_simulation.py --resume           # Resume from last checkpoint
"""

import argparse
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulation import Simulation
from src.config import SimConfig, load_config


def main():
    parser = argparse.ArgumentParser(description='Agent Civilization Simulation')

    parser.add_argument('--config', type=str, default='config/defaults.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--ticks', type=int, default=1000,
                        help='Number of ticks to run (0 for infinite)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint ID to resume from')
    parser.add_argument('--agents', type=int, default=None,
                        help='Override initial agent count')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpointing')
    parser.add_argument('--tps', type=float, default=0,
                        help='Target ticks per second (0 for unlimited)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config from {config_path}")
    else:
        config = SimConfig()
        print("Using default config")

    # Apply overrides
    if args.agents:
        config.INITIAL_AGENT_COUNT = args.agents
    if args.seed is not None:
        config.SEED = args.seed
    if args.no_checkpoint:
        config.CHECKPOINT_INTERVAL = 0

    # Create simulation
    checkpoint_path = None
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint

    print(f"\nInitializing simulation...")
    print(f"  World size: {config.WORLD_SIZE}x{config.WORLD_SIZE}")
    print(f"  Initial agents: {config.INITIAL_AGENT_COUNT:,}")
    print(f"  Seed: {config.SEED}")

    sim = Simulation(config=config, checkpoint_path=checkpoint_path)

    # Set up graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        sim.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\nStarting simulation...")
    print(f"  Starting tick: {sim.world_state.tick}")
    print(f"  Population: {sim.world_state.population:,}")
    print(f"  Target ticks: {args.ticks if args.ticks > 0 else 'infinite'}")
    print(f"  Target TPS: {args.tps if args.tps > 0 else 'unlimited'}")
    print(f"\nPress Ctrl+C to stop\n")

    # Run simulation
    try:
        if args.ticks > 0:
            sim.run(num_ticks=args.ticks, target_tps=args.tps)
        else:
            # Infinite run
            while True:
                sim.tick()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Final stats
        stats = sim.get_stats()
        print(f"\n{'='*50}")
        print("Final Statistics:")
        print(f"  Ticks completed: {stats['tick']:,}")
        print(f"  Final population: {stats['population']:,}")
        print(f"  Total births: {stats['total_births']:,}")
        print(f"  Total deaths: {stats['total_deaths']:,}")
        print(f"  Events logged: {stats['events']['total_events']:,}")

        # Shutdown
        sim.shutdown()
        print("\nSimulation ended.")


if __name__ == '__main__':
    main()
