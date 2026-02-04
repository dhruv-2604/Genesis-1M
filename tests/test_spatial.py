"""Tests for spatial hash grid"""

import pytest
import numpy as np
from src.spatial.hash_grid import SpatialHashGrid


class TestSpatialHashGrid:
    def test_insert_and_query(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        # Insert agents in same cell
        grid.insert(0, 25, 25)
        grid.insert(1, 30, 30)
        grid.insert(2, 45, 45)

        neighbors = grid.get_neighbors(30, 30, exclude_id=1)
        assert 0 in neighbors
        assert 2 in neighbors
        assert 1 not in neighbors

    def test_neighbors_across_cells(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        # Agents in adjacent cells
        grid.insert(0, 25, 25)   # Cell (0, 0)
        grid.insert(1, 75, 25)   # Cell (1, 0)
        grid.insert(2, 25, 75)   # Cell (0, 1)

        # Query from center should see all
        neighbors = grid.get_neighbors(50, 50)
        assert 0 in neighbors
        assert 1 in neighbors
        assert 2 in neighbors

    def test_distant_agents_not_neighbors(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        grid.insert(0, 25, 25)    # Near origin
        grid.insert(1, 500, 500)  # Far away

        neighbors = grid.get_neighbors(25, 25, exclude_id=0)
        assert 1 not in neighbors

    def test_remove_agent(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        grid.insert(0, 25, 25)
        grid.insert(1, 30, 30)

        grid.remove(0)

        neighbors = grid.get_neighbors(30, 30, exclude_id=1)
        assert 0 not in neighbors

    def test_move_agent(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        grid.insert(0, 25, 25)
        grid.move(0, 200, 200)

        # Should not be neighbor of origin anymore
        neighbors = grid.get_neighbors(25, 25)
        assert 0 not in neighbors

        # Should be neighbor of new location
        neighbors = grid.get_neighbors(200, 200, exclude_id=0)
        # Only agent 0 is there
        assert len(neighbors) == 0  # No other agents

    def test_world_wrapping(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        # Agent at edge should wrap coordinates
        grid.insert(0, 990, 990)

        # Query should find it
        neighbors = grid.get_neighbors(990, 990, exclude_id=0)
        assert len(neighbors) == 0  # Only itself there

    def test_rebuild(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        positions = {
            0: (100, 100),
            1: (200, 200),
            2: (300, 300),
        }

        grid.rebuild(positions)

        assert len(grid) == 3
        assert 0 in grid
        assert 1 in grid
        assert 2 in grid

    def test_density_map(self):
        grid = SpatialHashGrid(world_size=500, cell_size=100)

        # Put multiple agents in same cell
        for i in range(5):
            grid.insert(i, 50 + i, 50 + i)

        density = grid.get_density_map()
        assert density[0, 0] == 5  # 5 agents in cell (0,0)

    def test_neighbors_in_radius(self):
        grid = SpatialHashGrid(world_size=1000, cell_size=50)

        grid.insert(0, 100, 100)
        grid.insert(1, 110, 100)  # 10 units away
        grid.insert(2, 150, 100)  # 50 units away

        positions = {
            0: (100, 100),
            1: (110, 100),
            2: (150, 100),
        }

        # Radius 20 should find agent 1 only
        neighbors = grid.get_neighbors_in_radius(100, 100, 20, positions, exclude_id=0)
        neighbor_ids = [n[0] for n in neighbors]
        assert 1 in neighbor_ids
        assert 2 not in neighbor_ids
