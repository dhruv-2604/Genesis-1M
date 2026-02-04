#!/usr/bin/env python3
"""
Analyze event logs from simulation runs.

Generates statistics and interesting narratives from logged events.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.events import EventReader, EventType


def analyze(log_dir: str):
    """Analyze events from log directory"""
    reader = EventReader(log_dir)

    print(f"Reading events from {log_dir}...")
    events = reader.read_all()
    print(f"Found {len(events):,} events\n")

    if not events:
        print("No events to analyze")
        return

    # Basic counts
    counts = defaultdict(int)
    for e in events:
        counts[e.type] += 1

    print("Event Counts:")
    print("-" * 40)
    for event_type in EventType:
        count = counts[event_type]
        if count > 0:
            print(f"  {event_type.name:15} : {count:,}")

    # Timeline analysis
    tick_range = (events[0].tick, events[-1].tick)
    print(f"\nTick range: {tick_range[0]:,} - {tick_range[1]:,}")

    # Birth/death analysis
    births = [e for e in events if e.type == EventType.BIRTH]
    deaths = [e for e in events if e.type == EventType.DEATH]

    if deaths:
        death_causes = defaultdict(int)
        ages_at_death = []
        for d in deaths:
            death_causes[d.data.get('cause', 'unknown')] += 1
            ages_at_death.append(d.data.get('age', 0))

        print("\nDeath Causes:")
        for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
            print(f"  {cause:15} : {count:,} ({100*count/len(deaths):.1f}%)")

        if ages_at_death:
            avg_age = sum(ages_at_death) / len(ages_at_death)
            print(f"\nAverage age at death: {avg_age:.0f} ticks")

    # Lineage analysis
    if births:
        print("\nLineage Analysis:")

        # Find most prolific parents
        parent_counts = defaultdict(int)
        for b in births:
            parent_counts[b.data['parent1_id']] += 1
            parent_counts[b.data['parent2_id']] += 1

        top_parents = sorted(parent_counts.items(), key=lambda x: -x[1])[:5]
        print("  Most prolific parents:")
        for parent_id, count in top_parents:
            print(f"    Agent {parent_id}: {count} offspring")

        # Find longest lineages
        print(f"\n  Total unique parents: {len(parent_counts):,}")

    # Geographic analysis
    if births or deaths:
        print("\nGeographic Hotspots:")

        # Grid the world into regions
        grid_size = 1000  # Assuming world_size around 10000
        birth_grid = defaultdict(int)
        death_grid = defaultdict(int)

        for b in births:
            cell = (int(b.data['x'] / grid_size), int(b.data['y'] / grid_size))
            birth_grid[cell] += 1

        for d in deaths:
            cell = (int(d.data['x'] / grid_size), int(d.data['y'] / grid_size))
            death_grid[cell] += 1

        top_birth = sorted(birth_grid.items(), key=lambda x: -x[1])[:3]
        top_death = sorted(death_grid.items(), key=lambda x: -x[1])[:3]

        print("  Top birth locations:")
        for (x, y), count in top_birth:
            print(f"    Region ({x},{y}): {count} births")

        print("  Top death locations:")
        for (x, y), count in top_death:
            print(f"    Region ({x},{y}): {count} deaths")


def find_stories(log_dir: str, num_stories: int = 5):
    """Find interesting agent stories"""
    reader = EventReader(log_dir)
    events = reader.read_all()

    if not events:
        return

    print("\n" + "=" * 50)
    print("INTERESTING STORIES")
    print("=" * 50)

    births = {e.data['child_id']: e for e in events if e.type == EventType.BIRTH}
    deaths = {e.data['agent_id']: e for e in events if e.type == EventType.DEATH}

    # Find agents with longest lifespans
    long_lived = []
    for agent_id, death in deaths.items():
        age = death.data.get('age', 0)
        if age > 0:
            long_lived.append((agent_id, age, death))

    long_lived.sort(key=lambda x: -x[1])

    print("\nLongest-lived agents:")
    for agent_id, age, death in long_lived[:num_stories]:
        birth = births.get(agent_id)
        birth_info = f"born tick {birth.tick}" if birth else "origin agent"
        print(f"  Agent {agent_id}: lived {age} ticks ({birth_info}, died {death.data['cause']})")

    # Find largest families
    parent_children = defaultdict(list)
    for child_id, birth in births.items():
        parent_children[birth.data['parent1_id']].append(child_id)
        parent_children[birth.data['parent2_id']].append(child_id)

    big_families = sorted(parent_children.items(), key=lambda x: -len(x[1]))[:num_stories]

    print("\nLargest families:")
    for parent_id, children in big_families:
        if parent_id < 0:  # Original agents
            continue
        print(f"  Agent {parent_id}: {len(children)} children")


def main():
    parser = argparse.ArgumentParser(description='Analyze simulation event logs')
    parser.add_argument('log_dir', type=str, nargs='?', default='logs/events',
                        help='Path to event log directory')
    parser.add_argument('--stories', action='store_true',
                        help='Find interesting agent stories')

    args = parser.parse_args()

    analyze(args.log_dir)

    if args.stories:
        find_stories(args.log_dir)


if __name__ == '__main__':
    main()
