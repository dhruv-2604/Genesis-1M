"""Terrain generation using Perlin noise"""

from enum import IntEnum
from typing import Tuple
import numpy as np


class TerrainType(IntEnum):
    """Terrain types with different properties"""
    WATER = 0
    PLAINS = 1
    FOREST = 2
    MOUNTAIN = 3
    DESERT = 4


class TerrainGenerator:
    """
    Generate terrain using multi-octave Perlin noise.

    Terrain affects:
    - Agent movement speed
    - Resource spawning
    - Visibility
    """

    def __init__(
        self,
        world_size: float,
        resolution: int = 256,
        seed: int = 42
    ):
        self.world_size = world_size
        self.resolution = resolution
        self.cell_size = world_size / resolution

        self.rng = np.random.default_rng(seed)
        self.terrain_map = np.zeros((resolution, resolution), dtype=np.int8)
        self.elevation_map = np.zeros((resolution, resolution), dtype=np.float32)

        self._generate()

    def _generate(self) -> None:
        """Generate terrain using Perlin-like noise"""
        # Generate elevation using multiple octaves of noise
        self.elevation_map = self._octave_noise(
            octaves=6,
            persistence=0.5,
            scale=8.0
        )

        # Normalize to 0-1
        self.elevation_map = (self.elevation_map - self.elevation_map.min()) / \
                             (self.elevation_map.max() - self.elevation_map.min())

        # Generate moisture map for biome selection
        moisture_map = self._octave_noise(
            octaves=4,
            persistence=0.6,
            scale=4.0
        )
        moisture_map = (moisture_map - moisture_map.min()) / \
                       (moisture_map.max() - moisture_map.min())

        # Assign terrain types based on elevation and moisture
        for y in range(self.resolution):
            for x in range(self.resolution):
                elev = self.elevation_map[y, x]
                moist = moisture_map[y, x]

                if elev < 0.3:
                    self.terrain_map[y, x] = TerrainType.WATER
                elif elev > 0.8:
                    self.terrain_map[y, x] = TerrainType.MOUNTAIN
                elif moist < 0.3:
                    self.terrain_map[y, x] = TerrainType.DESERT
                elif moist > 0.6:
                    self.terrain_map[y, x] = TerrainType.FOREST
                else:
                    self.terrain_map[y, x] = TerrainType.PLAINS

    def _octave_noise(
        self,
        octaves: int = 4,
        persistence: float = 0.5,
        scale: float = 4.0
    ) -> np.ndarray:
        """Generate multi-octave noise"""
        result = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            noise = self._generate_noise_layer(frequency * scale)
            result += noise * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2

        return result / max_value

    def _generate_noise_layer(self, scale: float) -> np.ndarray:
        """Generate single noise layer using value noise interpolation"""
        # Create grid of random values at lower resolution
        grid_size = int(self.resolution / scale) + 2
        grid = self.rng.random((grid_size, grid_size)).astype(np.float32)

        # Interpolate to full resolution
        result = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        for y in range(self.resolution):
            for x in range(self.resolution):
                # Grid coordinates
                gx = x / scale
                gy = y / scale

                # Integer grid positions
                x0 = int(gx)
                y0 = int(gy)
                x1 = min(x0 + 1, grid_size - 1)
                y1 = min(y0 + 1, grid_size - 1)

                # Fractional parts
                fx = gx - x0
                fy = gy - y0

                # Smooth interpolation (smoothstep)
                fx = fx * fx * (3 - 2 * fx)
                fy = fy * fy * (3 - 2 * fy)

                # Bilinear interpolation
                v00 = grid[y0, x0]
                v10 = grid[y0, x1]
                v01 = grid[y1, x0]
                v11 = grid[y1, x1]

                v0 = v00 * (1 - fx) + v10 * fx
                v1 = v01 * (1 - fx) + v11 * fx

                result[y, x] = v0 * (1 - fy) + v1 * fy

        return result

    def get_terrain_at(self, x: float, y: float) -> TerrainType:
        """Get terrain type at world coordinates"""
        gx = int((x / self.world_size) * self.resolution) % self.resolution
        gy = int((y / self.world_size) * self.resolution) % self.resolution
        return TerrainType(self.terrain_map[gy, gx])

    def get_elevation_at(self, x: float, y: float) -> float:
        """Get elevation at world coordinates"""
        gx = int((x / self.world_size) * self.resolution) % self.resolution
        gy = int((y / self.world_size) * self.resolution) % self.resolution
        return float(self.elevation_map[gy, gx])

    def get_movement_multiplier(self, x: float, y: float) -> float:
        """Get movement speed multiplier based on terrain"""
        terrain = self.get_terrain_at(x, y)

        multipliers = {
            TerrainType.WATER: 0.3,    # Very slow in water
            TerrainType.PLAINS: 1.0,   # Normal speed
            TerrainType.FOREST: 0.7,   # Slower in forest
            TerrainType.MOUNTAIN: 0.4, # Very slow on mountains
            TerrainType.DESERT: 0.8,   # Slightly slow in desert
        }

        return multipliers.get(terrain, 1.0)

    def is_passable(self, x: float, y: float) -> bool:
        """Check if location is passable"""
        terrain = self.get_terrain_at(x, y)
        # Deep water and high mountains are impassable for now
        elev = self.get_elevation_at(x, y)
        if terrain == TerrainType.WATER and elev < 0.2:
            return False
        if terrain == TerrainType.MOUNTAIN and elev > 0.9:
            return False
        return True

    def get_resource_multiplier(self, x: float, y: float, resource_type: str) -> float:
        """Get resource spawn multiplier based on terrain"""
        terrain = self.get_terrain_at(x, y)

        # Different terrains favor different resources
        if resource_type == 'berries':
            return {
                TerrainType.FOREST: 2.0,
                TerrainType.PLAINS: 1.0,
                TerrainType.MOUNTAIN: 0.2,
                TerrainType.DESERT: 0.1,
                TerrainType.WATER: 0.0,
            }.get(terrain, 0.5)

        elif resource_type == 'meat':
            return {
                TerrainType.PLAINS: 1.5,
                TerrainType.FOREST: 1.0,
                TerrainType.MOUNTAIN: 0.5,
                TerrainType.DESERT: 0.3,
                TerrainType.WATER: 0.0,
            }.get(terrain, 0.5)

        elif resource_type == 'fish':
            return {
                TerrainType.WATER: 2.0,
                TerrainType.PLAINS: 0.0,
                TerrainType.FOREST: 0.0,
                TerrainType.MOUNTAIN: 0.0,
                TerrainType.DESERT: 0.0,
            }.get(terrain, 0.0)

        elif resource_type == 'flint':
            return {
                TerrainType.MOUNTAIN: 3.0,
                TerrainType.PLAINS: 0.3,
                TerrainType.FOREST: 0.1,
                TerrainType.DESERT: 0.5,
                TerrainType.WATER: 0.0,
            }.get(terrain, 0.2)

        return 1.0

    def to_dict(self) -> dict:
        """Serialize for checkpointing"""
        return {
            'world_size': self.world_size,
            'resolution': self.resolution,
            'terrain_map': self.terrain_map.tolist(),
            'elevation_map': self.elevation_map.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TerrainGenerator':
        """Deserialize from checkpoint"""
        gen = cls.__new__(cls)
        gen.world_size = data['world_size']
        gen.resolution = data['resolution']
        gen.cell_size = gen.world_size / gen.resolution
        gen.terrain_map = np.array(data['terrain_map'], dtype=np.int8)
        gen.elevation_map = np.array(data['elevation_map'], dtype=np.float32)
        return gen
