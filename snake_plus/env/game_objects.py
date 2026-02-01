"""
Module with game object classes.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple
import random


class ObjectType(Enum):
    """Types of objects on the field."""
    APPLE = auto()       # Regular apple: +1 length, +10 points
    GOLDEN = auto()      # Golden: +3 length, +30-70 points
    POISON = auto()      # Poison: death
    SOUR = auto()        # Sour: -1...-3 length, -5 points
    ROTTEN = auto()      # Rotten: detaches tail -> obstacle, -20 points
    OBSTACLE = auto()    # Obstacle (detached tail)


@dataclass
class GameObject:
    """Base game object class."""
    x: int
    y: int
    object_type: ObjectType
    lifetime: int = -1  # -1 = permanent, otherwise steps until disappearance

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


class ObjectFactory:
    """Factory for creating objects with random parameters."""

    def __init__(self, spawn_probs: dict, grid_size: Tuple[int, int]):
        """
        Args:
            spawn_probs: {"apple": 0.5, "golden": 0.1, ...}
            grid_size: (width, height)
        """
        self.spawn_probs = spawn_probs
        self.grid_size = grid_size

        # Normalize probabilities
        total = sum(spawn_probs.values())
        self.normalized_probs = {k: v / total for k, v in spawn_probs.items()}

    def create_random_object(self, occupied_positions: set) -> GameObject:
        """
        Creates a random object at a free position.

        Args:
            occupied_positions: set of occupied positions {(x, y), ...}

        Returns:
            GameObject or None if no space available
        """
        free_positions = [
            (x, y)
            for x in range(self.grid_size[0])
            for y in range(self.grid_size[1])
            if (x, y) not in occupied_positions
        ]

        if not free_positions:
            return None

        x, y = random.choice(free_positions)

        # Choose object type
        obj_type = self._random_type()

        return GameObject(x=x, y=y, object_type=obj_type)

    def _random_type(self) -> ObjectType:
        """Chooses a random type according to probabilities."""
        r = random.random()
        cumulative = 0

        type_mapping = {
            "apple": ObjectType.APPLE,
            "golden": ObjectType.GOLDEN,
            "poison": ObjectType.POISON,
            "sour": ObjectType.SOUR,
            "rotten": ObjectType.ROTTEN,
        }

        for name, prob in self.normalized_probs.items():
            cumulative += prob
            if r <= cumulative:
                return type_mapping[name]

        return ObjectType.APPLE  # fallback


class RewardCalculator:
    """Calculates rewards for different events."""

    # Base rewards
    REWARDS = {
        ObjectType.APPLE: 10,
        ObjectType.GOLDEN: (30, 70),  # random in range
        ObjectType.POISON: -1000,
        ObjectType.SOUR: -5,
        ObjectType.ROTTEN: -20,
    }

    # Additional rewards
    DEATH_PENALTY = -1000
    STEP_PENALTY = -0.1       # penalty per step (stimulates activity)
    SURVIVAL_BONUS = 0.5      # bonus for surviving

    @classmethod
    def get_reward(cls, obj_type: ObjectType) -> float:
        """Returns reward for eating an object."""
        reward = cls.REWARDS.get(obj_type, 0)

        if isinstance(reward, tuple):
            return random.uniform(reward[0], reward[1])

        return reward

    @classmethod
    def get_length_change(cls, obj_type: ObjectType) -> int:
        """Returns the change in snake length."""
        changes = {
            ObjectType.APPLE: 1,
            ObjectType.GOLDEN: 3,
            ObjectType.POISON: 0,  # death, doesn't matter
            ObjectType.SOUR: -random.randint(1, 3),  # random -1...-3
            ObjectType.ROTTEN: 0,  # handled separately (tail detach)
        }
        return changes.get(obj_type, 0)
