"""Tier 3 Agent FSM - Vectorized operations for maximum throughput"""

import numpy as np
from typing import Tuple, Optional
from enum import IntEnum

from ..core.state import FSMState, GeneIndex, AgentArrays


class Tier3FSM:
    """
    Vectorized Finite State Machine for Tier 3 agents.

    All state transitions and actions are computed using NumPy operations
    for maximum throughput. No per-agent Python loops in hot path.
    """

    # Energy thresholds
    HUNGER_THRESHOLD = 30.0
    SATIATED_THRESHOLD = 70.0
    REST_THRESHOLD = 90.0
    CRITICAL_ENERGY = 10.0

    # Reproduction thresholds
    REPRODUCTION_ENERGY = 50.0
    REPRODUCTION_COOLDOWN = 100  # ticks
    MATURITY_AGE = 5000

    # Movement
    BASE_SPEED = 1.0
    FLEE_SPEED_MULTIPLIER = 1.5

    # Perception
    BASE_VISION = 50.0
    PREDATOR_DETECTION_RANGE = 30.0

    def __init__(self, world_size: float, cell_size: float):
        self.world_size = world_size
        self.cell_size = cell_size

    def compute_transitions(
        self,
        arrays: AgentArrays,
        neighbor_counts: np.ndarray,
        has_food_nearby: np.ndarray,
        has_predator_nearby: np.ndarray,
        has_mate_nearby: np.ndarray,
    ) -> np.ndarray:
        """
        Compute FSM state transitions for all agents.
        Returns new FSM states array.

        Transition priority:
        1. FLEE if predator nearby
        2. SEEK_FOOD if hungry
        3. SEEK_MATE if ready to reproduce
        4. REST if full
        5. WANDER otherwise
        """
        n = arrays.count
        mask = arrays.get_alive_mask()

        current_state = arrays.fsm_state[:n].copy()
        energy = arrays.energy[:n]
        age = arrays.age[:n]
        cooldown = arrays.reproductive_cooldown[:n]

        new_state = np.full(n, FSMState.WANDER, dtype=np.int8)

        # Priority 1: Flee from predators
        flee_mask = mask & has_predator_nearby[:n]
        new_state[flee_mask] = FSMState.FLEE

        # Priority 2: Seek food when hungry (but not fleeing)
        hungry_mask = mask & ~flee_mask & (energy < self.HUNGER_THRESHOLD)
        new_state[hungry_mask] = FSMState.SEEK_FOOD

        # Priority 3: Seek mate when conditions met
        can_mate = (
            mask &
            ~flee_mask &
            ~hungry_mask &
            (energy >= self.REPRODUCTION_ENERGY) &
            (age >= self.MATURITY_AGE) &
            (cooldown == 0) &
            has_mate_nearby[:n]
        )
        new_state[can_mate] = FSMState.SEEK_MATE

        # Priority 4: Rest when very full
        rest_mask = (
            mask &
            ~flee_mask &
            ~hungry_mask &
            ~can_mate &
            (energy >= self.REST_THRESHOLD)
        )
        new_state[rest_mask] = FSMState.REST

        # Everything else: wander (already set as default)

        return new_state

    def compute_movement(
        self,
        arrays: AgentArrays,
        target_x: np.ndarray,
        target_y: np.ndarray,
        has_target: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute movement vectors for all agents based on FSM state.
        Returns (dx, dy) arrays.
        """
        n = arrays.count
        mask = arrays.get_alive_mask()

        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)

        state = arrays.fsm_state[:n]
        speed = arrays.genes[:n, GeneIndex.SPEED] * self.BASE_SPEED
        x = arrays.x[:n]
        y = arrays.y[:n]

        # Agents with targets move toward them
        target_mask = mask & has_target[:n]
        if np.any(target_mask):
            tx = target_x[:n][target_mask]
            ty = target_y[:n][target_mask]
            ax = x[target_mask]
            ay = y[target_mask]

            # Direction to target (handle wrapping)
            ddx = tx - ax
            ddy = ty - ay

            # Wrap around world edges
            ddx = np.where(ddx > self.world_size / 2, ddx - self.world_size, ddx)
            ddx = np.where(ddx < -self.world_size / 2, ddx + self.world_size, ddx)
            ddy = np.where(ddy > self.world_size / 2, ddy - self.world_size, ddy)
            ddy = np.where(ddy < -self.world_size / 2, ddy + self.world_size, ddy)

            # Normalize
            dist = np.sqrt(ddx * ddx + ddy * ddy)
            dist = np.maximum(dist, 0.001)  # Avoid division by zero

            dx[target_mask] = (ddx / dist) * speed[target_mask]
            dy[target_mask] = (ddy / dist) * speed[target_mask]

        # Fleeing agents move faster and away from target
        flee_mask = mask & (state == FSMState.FLEE)
        dx[flee_mask] *= -self.FLEE_SPEED_MULTIPLIER
        dy[flee_mask] *= -self.FLEE_SPEED_MULTIPLIER

        # Wandering agents move randomly
        wander_mask = mask & ~has_target[:n] & (state == FSMState.WANDER)
        n_wander = np.sum(wander_mask)
        if n_wander > 0:
            angles = rng.uniform(0, 2 * np.pi, n_wander)
            dx[wander_mask] = np.cos(angles) * speed[wander_mask]
            dy[wander_mask] = np.sin(angles) * speed[wander_mask]

        # Resting agents don't move
        rest_mask = mask & (state == FSMState.REST)
        dx[rest_mask] = 0
        dy[rest_mask] = 0

        return dx, dy

    def apply_movement(
        self,
        arrays: AgentArrays,
        dx: np.ndarray,
        dy: np.ndarray
    ) -> None:
        """Apply movement vectors and wrap around world boundaries"""
        n = arrays.count

        arrays.x[:n] = (arrays.x[:n] + dx) % self.world_size
        arrays.y[:n] = (arrays.y[:n] + dy) % self.world_size

    def update_energy(
        self,
        arrays: AgentArrays,
        food_consumed: np.ndarray,
        base_drain: float = 0.1
    ) -> None:
        """
        Update energy levels based on metabolism and food consumption.
        """
        n = arrays.count
        mask = arrays.get_alive_mask()

        # Energy drain based on metabolism gene and movement state
        metabolism = arrays.genes[:n, GeneIndex.METABOLISM]
        state = arrays.fsm_state[:n]

        # Different states have different energy costs
        drain = np.full(n, base_drain, dtype=np.float32)
        drain *= metabolism  # Higher metabolism = faster energy loss

        # Fleeing costs extra
        drain[state == FSMState.FLEE] *= 2.0

        # Resting costs less
        drain[state == FSMState.REST] *= 0.5

        # Apply drain and food
        arrays.energy[:n] = np.clip(
            arrays.energy[:n] - drain + food_consumed[:n],
            0.0,
            100.0
        )

        # Mark dead agents
        arrays.alive[:n] &= (arrays.energy[:n] > 0)

    def update_age(self, arrays: AgentArrays, max_age: int = 30000) -> None:
        """Increment age and kill agents past max age"""
        n = arrays.count
        mask = arrays.get_alive_mask()

        arrays.age[:n] += mask.astype(np.int32)
        arrays.alive[:n] &= (arrays.age[:n] < max_age)

        # Decrement reproduction cooldown
        cooldown_mask = mask & (arrays.reproductive_cooldown[:n] > 0)
        arrays.reproductive_cooldown[:n] -= cooldown_mask.astype(np.int32)

    def find_reproduction_pairs(
        self,
        arrays: AgentArrays,
        neighbor_pairs: np.ndarray,  # (N, 2) array of potential mate pairs
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Determine which agent pairs will reproduce this tick.
        Returns array of (parent1_id, parent2_id) pairs.
        """
        if len(neighbor_pairs) == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)

        valid_pairs = []

        for i in range(len(neighbor_pairs)):
            id1, id2 = neighbor_pairs[i]

            if id1 not in arrays.id_to_index or id2 not in arrays.id_to_index:
                continue

            idx1 = arrays.id_to_index[id1]
            idx2 = arrays.id_to_index[id2]

            # Check if both can reproduce
            can1 = (
                arrays.alive[idx1] and
                arrays.energy[idx1] >= self.REPRODUCTION_ENERGY and
                arrays.age[idx1] >= self.MATURITY_AGE and
                arrays.reproductive_cooldown[idx1] == 0
            )
            can2 = (
                arrays.alive[idx2] and
                arrays.energy[idx2] >= self.REPRODUCTION_ENERGY and
                arrays.age[idx2] >= self.MATURITY_AGE and
                arrays.reproductive_cooldown[idx2] == 0
            )

            if can1 and can2:
                # Probability based on fertility genes
                fertility = (
                    arrays.genes[idx1, GeneIndex.FERTILITY] *
                    arrays.genes[idx2, GeneIndex.FERTILITY]
                )
                if rng.random() < fertility:
                    valid_pairs.append((id1, id2))

        if not valid_pairs:
            return np.array([], dtype=np.int32).reshape(0, 2)

        return np.array(valid_pairs, dtype=np.int32)


def crossover(genes1: np.ndarray, genes2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Genetic crossover between two parents"""
    # Random crossover point
    crossover_point = rng.integers(1, len(genes1))

    # Create child genes
    child_genes = np.concatenate([
        genes1[:crossover_point],
        genes2[crossover_point:]
    ])

    return child_genes


def mutate(genes: np.ndarray, rng: np.random.Generator, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> np.ndarray:
    """Apply random mutations to genes"""
    mutated = genes.copy()

    # Randomly select genes to mutate
    mutation_mask = rng.random(len(genes)) < mutation_rate

    # Apply gaussian noise to selected genes
    noise = rng.normal(0, mutation_strength, len(genes))
    mutated[mutation_mask] += noise[mutation_mask]

    # Clamp to valid range
    mutated = np.clip(mutated, 0.0, 2.0)

    return mutated


def create_child(
    parent1_genes: np.ndarray,
    parent2_genes: np.ndarray,
    parent1_id: int,
    parent2_id: int,
    x: float,
    y: float,
    new_id: int,
    rng: np.random.Generator
) -> dict:
    """Create child agent data from two parents"""
    child_genes = crossover(parent1_genes, parent2_genes, rng)
    child_genes = mutate(child_genes, rng)

    return {
        'id': new_id,
        'x': x,
        'y': y,
        'energy': 50.0,
        'age': 0,
        'genes': child_genes,
        'parent_ids': (parent1_id, parent2_id),
        'reproductive_cooldown': 0,
    }
