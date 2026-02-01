"""
Gymnasium environment for Snake+.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import random

from .snake import Snake, Direction, Action
from .game_objects import (
    ObjectType, GameObject, ObjectFactory, RewardCalculator
)


class SnakePlusEnv(gym.Env):
    """
    Snake+ environment with various object types.

    Observations:
        Option 1 (for Q-table): feature vector
        Option 2 (for DQN): 2D state grid

    Actions:
        0: Move forward
        1: Turn left
        2: Turn right

    Rewards:
        - Apple: +10
        - Golden: +30...+70 (random)
        - Poison: -1000 (death)
        - Sour: -5
        - Rotten: -20
        - Each step: -0.1
        - Death (wall/body): -1000
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        spawn_probs: Optional[Dict[str, float]] = None,
        max_objects: int = 5,
        obstacle_decay: Optional[int] = 50,
        max_steps: int = 1000,
        observation_type: str = "features",  # "features" or "grid"
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            grid_size: field size (width, height)
            spawn_probs: object spawn probabilities
            max_objects: max objects on field (excluding obstacles)
            obstacle_decay: steps until obstacle disappears (None = never)
            max_steps: max steps per episode
            observation_type: observation type
            render_mode: rendering mode
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_objects = max_objects
        self.obstacle_decay = obstacle_decay
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.render_mode = render_mode

        # Default probabilities
        if spawn_probs is None:
            spawn_probs = {
                "apple": 0.50,
                "golden": 0.10,
                "poison": 0.15,
                "sour": 0.15,
                "rotten": 0.10,
            }

        self.spawn_probs = spawn_probs
        self.object_factory = ObjectFactory(spawn_probs, grid_size)

        # Action space: 3 actions
        self.action_space = spaces.Discrete(3)

        # Observation space
        if observation_type == "features":
            # Feature vector for Q-table
            # [danger_left, danger_straight, danger_right,  # 3
            #  dir_up, dir_down, dir_left, dir_right,       # 4
            #  food_up, food_down, food_left, food_right,   # 4
            #  nearest_obj_type (one-hot 5),                # 5
            #  distance_to_nearest_food,                    # 1
            #  snake_length_normalized]                     # 1
            # Total: 18 features
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(18,), dtype=np.float32
            )
        else:  # "grid"
            # 3D matrix for CNN
            # Channels: [head, body, apple, golden, poison, sour, rotten, obstacle]
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(8, grid_size[1], grid_size[0]),
                dtype=np.float32
            )

        # Game state (initialized in reset)
        self.snake: Optional[Snake] = None
        self.objects: List[GameObject] = []
        self.obstacles: List[GameObject] = []  # detached tails
        self.score: int = 0
        self.steps: int = 0

        # Renderer (initialized on first render)
        self.renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to initial state.

        Returns:
            observation: initial observation
            info: additional information
        """
        super().reset(seed=seed)

        # Initial snake position (center of field)
        start_x = self.grid_size[0] // 2
        start_y = self.grid_size[1] // 2

        self.snake = Snake(
            start_pos=(start_x, start_y),
            start_length=3,
            start_direction=Direction.RIGHT
        )

        self.objects = []
        self.obstacles = []
        self.score = 0
        self.steps = 0

        # Spawn initial objects
        self._spawn_objects()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step.

        Args:
            action: 0=forward, 1=left, 2=right

        Returns:
            observation: new observation
            reward: reward
            terminated: whether game ended (death)
            truncated: whether truncated (max_steps)
            info: additional information
        """
        self.steps += 1
        reward = RewardCalculator.STEP_PENALTY  # step penalty
        terminated = False
        truncated = False

        # Apply action
        self.snake.apply_action(Action(action))

        # Move snake
        new_head = self.snake.move()

        # Check wall collision
        if not self._is_valid_position(new_head):
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check body collision
        if self.snake.check_self_collision():
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check obstacle collision
        obstacle_positions = {obs.position for obs in self.obstacles}
        if new_head in obstacle_positions:
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check object collision
        eaten_object = None
        for obj in self.objects:
            if obj.position == new_head:
                eaten_object = obj
                break

        if eaten_object:
            reward += self._process_eaten_object(eaten_object)

            # Poison = death
            if eaten_object.object_type == ObjectType.POISON:
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self._get_info()

            # Check if snake became too short
            if self.snake.length < 1:
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Update obstacles (decay)
        self._update_obstacles()

        # Spawn new objects if needed
        self._spawn_objects()

        # Survival bonus
        reward += RewardCalculator.SURVIVAL_BONUS

        # Check max_steps
        if self.steps >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _process_eaten_object(self, obj: GameObject) -> float:
        """
        Processes eating an object.

        Returns:
            Reward
        """
        reward = RewardCalculator.get_reward(obj.object_type)
        self.score += max(0, int(reward))

        # Remove object
        self.objects.remove(obj)

        # Apply effect
        if obj.object_type == ObjectType.APPLE:
            self.snake.grow(1)

        elif obj.object_type == ObjectType.GOLDEN:
            self.snake.grow(3)

        elif obj.object_type == ObjectType.SOUR:
            shrink_amount = random.randint(1, 3)
            self.snake.shrink(shrink_amount)

        elif obj.object_type == ObjectType.ROTTEN:
            # Detach 3-5 tail segments
            detach_amount = random.randint(3, 5)
            detached_positions = self.snake.detach_tail(detach_amount)

            # Create obstacles
            for pos in detached_positions:
                obstacle = GameObject(
                    x=pos[0], y=pos[1],
                    object_type=ObjectType.OBSTACLE,
                    lifetime=self.obstacle_decay if self.obstacle_decay else -1
                )
                self.obstacles.append(obstacle)

        return reward

    def _update_obstacles(self) -> None:
        """Updates obstacle lifetimes, removes expired ones."""
        if self.obstacle_decay is None:
            return

        remaining = []
        for obs in self.obstacles:
            if obs.lifetime > 0:
                obs.lifetime -= 1
                if obs.lifetime > 0:
                    remaining.append(obs)
            elif obs.lifetime == -1:  # permanent
                remaining.append(obs)

        self.obstacles = remaining

    def _spawn_objects(self) -> None:
        """Spawns objects up to max_objects."""
        occupied = self._get_occupied_positions()

        while len(self.objects) < self.max_objects:
            obj = self.object_factory.create_random_object(occupied)
            if obj is None:
                break  # no space

            self.objects.append(obj)
            occupied.add(obj.position)

    def _get_occupied_positions(self) -> set:
        """Returns all occupied positions."""
        occupied = self.snake.get_body_set()
        occupied.update(obj.position for obj in self.objects)
        occupied.update(obs.position for obs in self.obstacles)
        return occupied

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Checks if position is within field bounds."""
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def _get_observation(self) -> np.ndarray:
        """Generates observation."""
        if self.observation_type == "features":
            return self._get_feature_observation()
        else:
            return self._get_grid_observation()

    def _get_feature_observation(self) -> np.ndarray:
        """
        Generates feature vector for Q-table agent.

        Features:
        [0-2]: danger_left, danger_straight, danger_right
        [3-6]: direction one-hot (up, down, left, right)
        [7-10]: food direction (up, down, left, right)
        [11-15]: nearest object type one-hot (apple, golden, poison, sour, rotten)
        [16]: distance to nearest food (normalized)
        [17]: snake length (normalized)
        """
        features = np.zeros(18, dtype=np.float32)

        head = self.snake.head
        direction = self.snake.direction

        # Danger in 3 directions (relative to current direction)
        features[0] = self._is_danger(self._get_left_direction(direction))
        features[1] = self._is_danger(direction)  # straight
        features[2] = self._is_danger(self._get_right_direction(direction))

        # Direction (one-hot)
        dir_idx = {Direction.UP: 3, Direction.DOWN: 4,
                   Direction.LEFT: 5, Direction.RIGHT: 6}
        features[dir_idx[direction]] = 1

        # Direction to nearest food
        food_objects = [obj for obj in self.objects
                       if obj.object_type in (ObjectType.APPLE, ObjectType.GOLDEN)]

        if food_objects:
            nearest_food = min(food_objects,
                             key=lambda o: abs(o.x - head[0]) + abs(o.y - head[1]))

            # food_up, food_down, food_left, food_right
            features[7] = 1 if nearest_food.y < head[1] else 0
            features[8] = 1 if nearest_food.y > head[1] else 0
            features[9] = 1 if nearest_food.x < head[0] else 0
            features[10] = 1 if nearest_food.x > head[0] else 0

            # Distance (normalized)
            dist = abs(nearest_food.x - head[0]) + abs(nearest_food.y - head[1])
            max_dist = self.grid_size[0] + self.grid_size[1]
            features[16] = dist / max_dist

        # Nearest object of any type
        if self.objects:
            nearest_obj = min(self.objects,
                            key=lambda o: abs(o.x - head[0]) + abs(o.y - head[1]))

            type_idx = {
                ObjectType.APPLE: 11,
                ObjectType.GOLDEN: 12,
                ObjectType.POISON: 13,
                ObjectType.SOUR: 14,
                ObjectType.ROTTEN: 15,
            }
            features[type_idx[nearest_obj.object_type]] = 1

        # Snake length (normalized)
        max_length = self.grid_size[0] * self.grid_size[1]
        features[17] = self.snake.length / max_length

        return features

    def _get_grid_observation(self) -> np.ndarray:
        """
        Generates 3D matrix for CNN.

        Channels:
        0: head
        1: body
        2: apple
        3: golden
        4: poison
        5: sour
        6: rotten
        7: obstacle
        """
        grid = np.zeros((8, self.grid_size[1], self.grid_size[0]), dtype=np.float32)

        # Head
        hx, hy = self.snake.head
        if 0 <= hx < self.grid_size[0] and 0 <= hy < self.grid_size[1]:
            grid[0, hy, hx] = 1

        # Body
        for i, (bx, by) in enumerate(self.snake.body):
            if i > 0:  # skip head
                if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]:
                    grid[1, by, bx] = 1

        # Objects
        channel_map = {
            ObjectType.APPLE: 2,
            ObjectType.GOLDEN: 3,
            ObjectType.POISON: 4,
            ObjectType.SOUR: 5,
            ObjectType.ROTTEN: 6,
        }

        for obj in self.objects:
            ch = channel_map[obj.object_type]
            grid[ch, obj.y, obj.x] = 1

        # Obstacles
        for obs in self.obstacles:
            grid[7, obs.y, obs.x] = 1

        return grid

    def _is_danger(self, direction: Direction) -> float:
        """Checks if there is danger in a direction."""
        hx, hy = self.snake.head
        dx, dy = direction.value
        next_pos = (hx + dx, hy + dy)

        # Wall
        if not self._is_valid_position(next_pos):
            return 1.0

        # Body
        if next_pos in self.snake.get_body_set():
            return 1.0

        # Obstacle
        if next_pos in {obs.position for obs in self.obstacles}:
            return 1.0

        return 0.0

    def _get_left_direction(self, d: Direction) -> Direction:
        """Returns direction to the left of current."""
        return Snake.TURN_LEFT_MAP[d]

    def _get_right_direction(self, d: Direction) -> Direction:
        """Returns direction to the right of current."""
        return Snake.TURN_RIGHT_MAP[d]

    def _get_info(self) -> Dict[str, Any]:
        """Returns additional information."""
        return {
            "score": self.score,
            "length": self.snake.length,
            "steps": self.steps,
            "obstacles_count": len(self.obstacles),
        }

    def render(self):
        """Renders current state."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            from .renderer import Renderer
            self.renderer = Renderer(
                grid_size=self.grid_size,
                render_mode=self.render_mode
            )

        return self.renderer.render(
            snake=self.snake,
            objects=self.objects,
            obstacles=self.obstacles,
            score=self.score,
            steps=self.steps
        )

    def close(self):
        """Closes environment."""
        if self.renderer:
            self.renderer.close()
