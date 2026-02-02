import gymnasium

from .game_objects import ObjectType, GameObject, ObjectFactory, RewardCalculator
from .snake import Snake, Direction, Action
from .snake_env import SnakePlusEnv

gymnasium.register(
    id="SnakePlus-v0",
    entry_point="env.snake_env:SnakePlusEnv",
    max_episode_steps=1000,
)
