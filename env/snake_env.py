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
        - Apple / golden: positive base rewards, scaled up with snake length
        - Poison / sour / rotten: negative rewards, magnitude reduced as snake grows
        - Golden: +30...+70 (random) before length scaling
        - Poison: death; penalty magnitude scales with length like other deaths
        - Each step: -0.1; survival bonus; optional Manhattan proximity shaping
          toward apple/golden (reward) and away from poison/sour/rotten (penalty)
        - Death (wall/body/obstacle/starvation): base -1000 scaled down as snake grows
        - Starvation: die after ``starvation_max_steps`` consecutive steps without eating
          any object (-1 disables)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    _GOOD_TYPES = (ObjectType.APPLE, ObjectType.GOLDEN)
    _BAD_TYPES = (ObjectType.POISON, ObjectType.SOUR, ObjectType.ROTTEN)

    # Reward-proportional scalars for grid encoding (normalised to [0, 1])
    _MAX_POS_REWARD = 70.0   # golden upper bound
    _MAX_NEG_REWARD = 100.0  # poison
    _GOOD_SCALAR = {
        ObjectType.APPLE: 0.3,             # floor raised from 0.14 for detectability
        ObjectType.GOLDEN: 50.0 / 70.0,   # ≈ 0.71
    }
    _BAD_SCALAR = {
        ObjectType.SOUR: 5.0 / 100.0,     # 0.05
        ObjectType.ROTTEN: 20.0 / 100.0,  # 0.20
        ObjectType.POISON: 1.0,            # 1.00
    }

    _FIELD_ALPHA_OBJ = 0.15
    _FIELD_ALPHA_DANGER = 0.2


    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        spawn_probs: Optional[Dict[str, float]] = None,
        max_objects: int = 5,
        obstacle_decay: Optional[int] = 50,
        max_steps: int = 1000,
        observation_type: str = "features",  # "features" or "grid"
        render_mode: Optional[str] = None,
        starvation_max_steps: int = 400,
        proximity_good_scale: float = 0.01,
        proximity_bad_scale: float = 0.004,
        fruit_reward_length_coef: float = 0.5,
        fruit_penalty_length_coef: float = 0.5,
        fruit_penalty_min_factor: float = 0.25,
        death_penalty_length_coef: float = 0.5,
        death_penalty_min_scale: float = 0.35,
        object_lifetime: int = -1,
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
            starvation_max_steps: terminate if no object eaten for this many steps;
                -1 disables starvation
            proximity_good_scale: reward scale for moving closer to apple/golden (0 off)
            proximity_bad_scale: penalty scale for moving closer to poison/sour/rotten (0 off)
            fruit_reward_length_coef: positive eat rewards multiply by (1 + coef * L/L_max)
            fruit_penalty_length_coef: negative eat rewards scale by
                max(min_factor, 1 - coef * L/L_max)
            fruit_penalty_min_factor: floor for negative fruit penalty magnitude
            death_penalty_length_coef: death penalty scales by max(min_scale, 1 - coef * L/L_max)
            death_penalty_min_scale: minimum fraction of base death penalty
            object_lifetime: steps before an uneaten object despawns (-1 = permanent)
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_objects = max_objects
        self.obstacle_decay = obstacle_decay
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.render_mode = render_mode
        self.starvation_max_steps = starvation_max_steps
        self.proximity_good_scale = proximity_good_scale
        self.proximity_bad_scale = proximity_bad_scale
        self.fruit_reward_length_coef = fruit_reward_length_coef
        self.fruit_penalty_length_coef = fruit_penalty_length_coef
        self.fruit_penalty_min_factor = fruit_penalty_min_factor
        self.death_penalty_length_coef = death_penalty_length_coef
        self.death_penalty_min_scale = death_penalty_min_scale
        self.object_lifetime = object_lifetime

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
            # 29-dim feature vector (all relative to snake heading)
            # [0-2]   danger forward/left/right (same order as Action: 0=FWD,1=LEFT,2=RIGHT)
            # [3-5]   food forward/left/right (relative)
            # [6]     food distance (normalized)
            # [7-9]   bad forward/left/right (relative)
            # [10]    bad distance (normalized)
            # [11-14] clear steps forward/left/right/behind until wall, body, or obstacle
            # [15-18] direction one-hot (up/down/left/right)
            # [19-23] nearest object type one-hot (apple/golden/poison/sour/rotten)
            # [24]    snake length (normalized by grid area)
            # [25]    starvation countdown (1.0=just ate, 0.0=about to starve; 1.0 if disabled)
            # [26]    good objects on board (normalized by max_objects)
            # [27]    bad objects on board (normalized by max_objects)
            # [28]    obstacles on board (normalized by grid area)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(29,), dtype=np.float32
            )
        else:  # "grid"
            # 6-channel grid for CNN (dense distance-field encoding)
            # 0: snake (head=1.0, body decayed toward 0.005, step<=0.05)
            # 1: good-object proximity field (exp-Manhattan decay, weighted)
            # 2: bad-object proximity field  (exp-Manhattan decay, weighted)
            # 3: danger proximity field (walls + obstacles, exp-Manhattan)
            # 4-5: direction dx/dy broadcast to entire grid
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(6, *grid_size),
                dtype=np.float32
            )

        # Precomputed grids for dense distance-field observations
        W, H = grid_size
        ys, xs = np.mgrid[0:H, 0:W]
        self._grid_xs = xs.astype(np.float32)
        self._grid_ys = ys.astype(np.float32)
        wall_dist = np.minimum(
            np.minimum(xs, W - 1 - xs),
            np.minimum(ys, H - 1 - ys),
        ).astype(np.float32)
        self._wall_proximity = np.exp(-0.2 * wall_dist).astype(np.float32)

        # Game state (initialized in reset)
        self.snake: Optional[Snake] = None
        self.objects: List[GameObject] = []
        self.obstacles: List[GameObject] = []  # detached tails
        self._obstacle_pos_set: Optional[set] = None
        self.score: int = 0
        self.steps: int = 0
        self._steps_since_food: int = 0

        # Renderer (initialized on first render)
        self.renderer = None

    def _get_obstacle_set(self) -> set:
        """Returns cached set of obstacle positions."""
        if self._obstacle_pos_set is None:
            self._obstacle_pos_set = {obs.position for obs in self.obstacles}
        return self._obstacle_pos_set

    def _invalidate_obstacle_cache(self):
        self._obstacle_pos_set = None

    def _grid_cell_count(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def _norm_length(self, length: int) -> float:
        return length / self._grid_cell_count()

    def _death_length_scale(self) -> float:
        nl = self._norm_length(self.snake.length)
        if nl >= 1.0 / 3.0:
            return 0.0
        return max(
            self.death_penalty_min_scale,
            1.0 - self.death_penalty_length_coef * nl,
        )

    def _death_penalty(self) -> float:
        return RewardCalculator.DEATH_PENALTY * self._death_length_scale()

    def _scale_fruit_reward(self, base: float, length: int) -> float:
        """Scales eat reward by snake length (length before grow/shrink)."""
        nl = self._norm_length(length)
        if base > 0:
            return base * (1.0 + self.fruit_reward_length_coef * nl)
        if base < 0:
            factor = max(
                self.fruit_penalty_min_factor,
                1.0 - self.fruit_penalty_length_coef * nl,
            )
            return base * factor
        return base

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_in_types(
        self, pos: Tuple[int, int], types: Tuple[ObjectType, ...]
    ) -> Optional[GameObject]:
        candidates = [o for o in self.objects if o.object_type in types]
        if not candidates:
            return None
        return min(candidates, key=lambda o: self._manhattan(pos, o.position))

    def _proximity_shaping(self, old_head: Tuple[int, int], new_head: Tuple[int, int]) -> float:
        r = 0.0
        if self.proximity_good_scale != 0.0:
            good = self._nearest_in_types(old_head, self._GOOD_TYPES)
            if good is not None:
                d_old = self._manhattan(old_head, good.position)
                d_new = self._manhattan(new_head, good.position)
                delta_good = d_old - d_new
                r += self.proximity_good_scale * delta_good
        if self.proximity_bad_scale != 0.0:
            bad = self._nearest_in_types(old_head, self._BAD_TYPES)
            if bad is not None:
                d_old = self._manhattan(old_head, bad.position)
                d_new = self._manhattan(new_head, bad.position)
                delta_bad = d_old - d_new
                r -= self.proximity_bad_scale * delta_bad
        return r

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
        self._obstacle_pos_set = None
        self.score = 0
        self.steps = 0
        self._steps_since_food = 0

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

        old_head = self.snake.head

        # Apply action
        self.snake.apply_action(Action(action))

        # Move snake
        new_head = self.snake.move()

        # Check wall collision
        if not self._is_valid_position(new_head):
            reward += self._death_penalty()
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check body collision
        if self.snake.check_self_collision():
            reward += self._death_penalty()
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # Check obstacle collision
        if new_head in self._get_obstacle_set():
            reward += self._death_penalty()
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        reward += self._proximity_shaping(old_head, new_head)

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

        # Decay object lifetimes, remove expired
        self._update_objects()

        # Spawn new objects if needed
        self._spawn_objects()

        if eaten_object is not None:
            self._steps_since_food = 0
        elif self.starvation_max_steps >= 0:
            self._steps_since_food += 1
            if self._steps_since_food >= self.starvation_max_steps:
                reward += self._death_penalty()
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self._get_info()

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
        length_before = self.snake.length
        base = RewardCalculator.get_reward(obj.object_type)
        reward = self._scale_fruit_reward(base, length_before)
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
            self._invalidate_obstacle_cache()
            for pos in detached_positions:
                obstacle = GameObject(
                    x=pos[0], y=pos[1],
                    object_type=ObjectType.OBSTACLE,
                    lifetime=self.obstacle_decay + 1 if self.obstacle_decay else -1
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

        if len(remaining) != len(self.obstacles):
            self._invalidate_obstacle_cache()
        self.obstacles = remaining

    def _update_objects(self) -> None:
        """Decrements object lifetimes and removes expired ones."""
        if self.object_lifetime < 0:
            return

        remaining = []
        for obj in self.objects:
            if obj.lifetime > 0:
                obj.lifetime -= 1
                if obj.lifetime > 0:
                    remaining.append(obj)
            elif obj.lifetime == -1:
                remaining.append(obj)
        self.objects = remaining

    def _spawn_objects(self) -> None:
        """Spawns objects up to max_objects, guaranteeing at least one apple."""
        occupied = self._get_occupied_positions()
        lifetime = self.object_lifetime + 1 if self.object_lifetime >= 0 else -1

        has_apple = any(o.object_type == ObjectType.APPLE for o in self.objects)
        if not has_apple:
            apple = self.object_factory.create_typed_object(
                ObjectType.APPLE, occupied
            )
            if apple is not None:
                apple.lifetime = lifetime
                self.objects.append(apple)
                occupied.add(apple.position)

        while len(self.objects) < self.max_objects:
            obj = self.object_factory.create_random_object(occupied)
            if obj is None:
                break  # no space

            obj.lifetime = lifetime
            self.objects.append(obj)
            occupied.add(obj.position)

    def _get_occupied_positions(self) -> set:
        """Returns all occupied positions."""
        occupied = set(self.snake.get_body_set())
        occupied.update(obj.position for obj in self.objects)
        occupied.update(self._get_obstacle_set())
        return occupied

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Checks if position is within field bounds."""
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def _object_intensity(self, obj: GameObject) -> float:
        """Returns a value in [0.1, 1.0] based on remaining lifetime fraction."""
        if obj.lifetime < 0 or self.object_lifetime < 0:
            return 1.0
        return 0.1 + 0.9 * obj.lifetime / (self.object_lifetime + 1)

    def _get_observation(self) -> np.ndarray:
        """Generates observation."""
        if self.observation_type == "features":
            return self._get_feature_observation()
        else:
            return self._get_grid_observation()

    def _get_feature_observation(self) -> np.ndarray:
        """
        Generates 29-dim feature vector with all spatial info relative to heading.

        Layout:
        [0-2]   danger forward / left / right (indices match Action enum)
        [3-5]   food forward / left / right (relative to heading)
        [6]     food distance (normalized Manhattan)
        [7-9]   bad forward / left / right (relative to heading)
        [10]    bad distance (normalized Manhattan)
        [11-14] clear steps forward / left / right / behind until wall, body, or obstacle
            (normalized by max grid dimension)
        [15-18] direction one-hot (up / down / left / right)
        [19-23] nearest object type one-hot
        [24]    snake length (normalized by grid area)
        [25]    starvation countdown (1.0=just ate, 0.0=about to starve; 1.0 if disabled)
        [26]    good objects on board (normalized by max_objects)
        [27]    bad objects on board (normalized by max_objects)
        [28]    obstacles on board (normalized by grid area)
        """
        features = np.zeros(29, dtype=np.float32)

        head = self.snake.head
        direction = self.snake.direction
        left_dir = self._get_left_direction(direction)
        right_dir = self._get_right_direction(direction)

        fwd_vec = direction.value
        left_vec = left_dir.value
        right_vec = right_dir.value
        behind_vec = (-fwd_vec[0], -fwd_vec[1])

        # [0-2] Danger forward / left / right (aligned with Action.FORWARD/LEFT/RIGHT)
        features[0] = self._is_danger(direction)
        features[1] = self._is_danger(left_dir)
        features[2] = self._is_danger(right_dir)

        max_dist = self.grid_size[0] + self.grid_size[1]

        # [3-6] Relative food direction + distance (nearest by Manhattan)
        food_objects = [obj for obj in self.objects
                        if obj.object_type in (ObjectType.APPLE, ObjectType.GOLDEN)]
        if food_objects:
            nf = min(
                food_objects,
                key=lambda o: abs(o.x - head[0]) + abs(o.y - head[1]),
            )
            dx, dy = nf.x - head[0], nf.y - head[1]
            features[3] = 1.0 if (dx * fwd_vec[0] + dy * fwd_vec[1]) > 0 else 0.0
            features[4] = 1.0 if (dx * left_vec[0] + dy * left_vec[1]) > 0 else 0.0
            features[5] = 1.0 if (dx * right_vec[0] + dy * right_vec[1]) > 0 else 0.0
            features[6] = (abs(dx) + abs(dy)) / max_dist

        # [7-10] Relative bad-object direction + distance (nearest by Manhattan)
        bad_objects = [obj for obj in self.objects
                       if obj.object_type in self._BAD_TYPES]
        if bad_objects:
            nb = min(
                bad_objects,
                key=lambda o: abs(o.x - head[0]) + abs(o.y - head[1]),
            )
            dx, dy = nb.x - head[0], nb.y - head[1]
            features[7] = 1.0 if (dx * fwd_vec[0] + dy * fwd_vec[1]) > 0 else 0.0
            features[8] = 1.0 if (dx * left_vec[0] + dy * left_vec[1]) > 0 else 0.0
            features[9] = 1.0 if (dx * right_vec[0] + dy * right_vec[1]) > 0 else 0.0
            features[10] = (abs(dx) + abs(dy)) / max_dist

        # [11-14] Clear steps along each ray until wall, body, or obstacle (normalized)
        max_grid_dim = max(self.grid_size)
        for i, dvec in enumerate((fwd_vec, left_vec, right_vec, behind_vec)):
            clear = self._clear_steps_along_ray(dvec)
            features[11 + i] = clear / max_grid_dim

        # [15-18] Direction one-hot
        dir_idx = {Direction.UP: 15, Direction.DOWN: 16,
                   Direction.LEFT: 17, Direction.RIGHT: 18}
        features[dir_idx[direction]] = 1.0

        # [19-23] Nearest object type (scaled by remaining lifetime)
        if self.objects:
            nearest_obj = min(self.objects,
                              key=lambda o: abs(o.x - head[0]) + abs(o.y - head[1]))
            type_idx = {
                ObjectType.APPLE: 19, ObjectType.GOLDEN: 20,
                ObjectType.POISON: 21, ObjectType.SOUR: 22, ObjectType.ROTTEN: 23,
            }
            features[type_idx[nearest_obj.object_type]] = self._object_intensity(nearest_obj)

        # [24] Snake length normalized by grid area
        features[24] = self.snake.length / self._grid_cell_count()

        # [25] Starvation countdown: 1.0 = just ate / disabled, 0.0 = about to starve
        if self.starvation_max_steps > 0:
            features[25] = 1.0 - (self._steps_since_food / self.starvation_max_steps)
        else:
            features[25] = 1.0

        # [26-27] Object counts normalized by max_objects
        good_count = sum(1 for o in self.objects if o.object_type in self._GOOD_TYPES)
        bad_count = sum(1 for o in self.objects if o.object_type in self._BAD_TYPES)
        features[26] = good_count / max(1, self.max_objects)
        features[27] = bad_count / max(1, self.max_objects)

        # [28] Obstacle count normalized by grid area
        features[28] = len(self.obstacles) / self._grid_cell_count()

        return features

    def _distance_field(self, x: int, y: int, alpha: float) -> np.ndarray:
        """Exponential-decay Manhattan distance field from point (x, y)."""
        dist = np.abs(self._grid_xs - x) + np.abs(self._grid_ys - y)
        return np.exp(-alpha * dist)

    def _get_grid_observation(self) -> np.ndarray:
        """
        Generates 6-channel 3D tensor for CNN using dense distance fields.

        Channels:
        0: snake (head=1.0, body decayed toward 0.005, step<=0.05)
        1: good-object proximity field (weighted exp-Manhattan decay from each)
        2: bad-object proximity field  (weighted exp-Manhattan decay from each)
        3: danger proximity field (wall + obstacle distance fields)
        4: direction dx broadcast to entire grid
        5: direction dy broadcast to entire grid
        """
        grid = np.zeros((6, *self.grid_size), dtype=np.float32)

        # Ch 0: Snake (head=1.0, body decays from 0.995 toward 0.005)
        n_seg = len(self.snake.body)
        step = min(0.05, 0.995 / max(1, n_seg - 1)) if n_seg > 1 else 0.0
        for i, (bx, by) in enumerate(self.snake.body):
            if not (0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]):
                continue
            grid[0, by, bx] = max(0.005, 1.0 - i * step)

        # Ch 1: Good-object proximity field
        alpha_obj = self._FIELD_ALPHA_OBJ
        for obj in self.objects:
            if obj.object_type in self._GOOD_SCALAR:
                weight = self._GOOD_SCALAR[obj.object_type] * self._object_intensity(obj)
                field = weight * self._distance_field(obj.x, obj.y, alpha_obj)
                np.maximum(grid[1], field, out=grid[1])

        # Ch 2: Bad-object proximity field
        for obj in self.objects:
            if obj.object_type in self._BAD_SCALAR:
                weight = self._BAD_SCALAR[obj.object_type] * self._object_intensity(obj)
                field = weight * self._distance_field(obj.x, obj.y, alpha_obj)
                np.maximum(grid[2], field, out=grid[2])

        # Ch 3: Danger proximity (walls + obstacles)
        grid[3] = self._wall_proximity.copy()
        alpha_d = self._FIELD_ALPHA_DANGER
        for obs in self.obstacles:
            field = self._distance_field(obs.x, obs.y, alpha_d)
            np.maximum(grid[3], field, out=grid[3])

        # Ch 4-5: Direction broadcast to entire grid
        dx, dy = self.snake.direction.value
        grid[4, :, :] = (dx + 1) / 2.0
        grid[5, :, :] = (dy + 1) / 2.0

        return grid

    def _clear_steps_along_ray(self, dvec: Tuple[int, int]) -> int:
        """
        Counts how many consecutive steps from the head along ``dvec`` are free
        (inside the grid, not body, not obstacle). Stops at the first wall,
        body segment, or obstacle cell (that cell is not counted).
        """
        dx, dy = dvec
        x, y = self.snake.head
        body_set = self.snake.get_body_set()
        obs_set = self._get_obstacle_set()
        clear = 0
        while True:
            x += dx
            y += dy
            if not self._is_valid_position((x, y)):
                return clear
            if (x, y) in body_set or (x, y) in obs_set:
                return clear
            clear += 1

    def _is_danger(self, direction: Direction) -> float:
        """Checks if there is danger in a direction."""
        hx, hy = self.snake.head
        dx, dy = direction.value
        next_pos = (hx + dx, hy + dy)

        if not self._is_valid_position(next_pos):
            return 1.0

        if next_pos in self.snake.get_body_set():
            return 1.0

        if next_pos in self._get_obstacle_set():
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
            "steps_since_food": self._steps_since_food,
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


def make_snake_env(
    env_cfg: Dict[str, Any],
    render_mode: Optional[str] = None,
) -> SnakePlusEnv:
    """
    Builds ``SnakePlusEnv`` from a training-style ``env`` config dict (e.g. YAML ``env:`` block).

    Omitted keys use the same defaults as ``SnakePlusEnv.__init__``.
    """
    return SnakePlusEnv(
        grid_size=tuple(env_cfg["grid_size"]),
        spawn_probs=env_cfg.get("spawn_probs"),
        max_objects=env_cfg["max_objects"],
        obstacle_decay=env_cfg.get("obstacle_decay"),
        max_steps=env_cfg["max_steps"],
        observation_type=env_cfg["observation_type"],
        render_mode=render_mode,
        starvation_max_steps=env_cfg.get("starvation_max_steps", 400),
        proximity_good_scale=env_cfg.get("proximity_good_scale", 0.01),
        proximity_bad_scale=env_cfg.get("proximity_bad_scale", 0.004),
        fruit_reward_length_coef=env_cfg.get("fruit_reward_length_coef", 0.5),
        fruit_penalty_length_coef=env_cfg.get("fruit_penalty_length_coef", 0.5),
        fruit_penalty_min_factor=env_cfg.get("fruit_penalty_min_factor", 0.25),
        death_penalty_length_coef=env_cfg.get("death_penalty_length_coef", 0.5),
        death_penalty_min_scale=env_cfg.get("death_penalty_min_scale", 0.35),
        object_lifetime=env_cfg.get("object_lifetime", -1),
    )
