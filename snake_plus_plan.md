# 🐍 Snake+ : Детальний план реалізації

## Огляд проекту

**Мета:** Створити гру Snake з особливими об'єктами та навчити RL-агента грати в неї, досліджуючи вплив дисконтування на стратегії.

**Технології:** Python 3.10+, Gymnasium, PyTorch, Pygame, NumPy, Matplotlib

---

## 📁 Структура проекту

```
snake_plus/
├── README.md
├── requirements.txt
├── setup.py
│
├── env/
│   ├── __init__.py
│   ├── snake_env.py          # Gymnasium середовище
│   ├── game_objects.py       # Класи об'єктів (яблука, отрута тощо)
│   ├── snake.py              # Клас змійки
│   └── renderer.py           # Pygame візуалізація
│
├── agent/
│   ├── __init__.py
│   ├── q_table_agent.py      # Табличний Q-learning агент
│   ├── dqn_agent.py          # Deep Q-Network агент
│   ├── replay_buffer.py      # Experience replay буфер
│   └── networks.py           # Нейронні мережі для DQN
│
├── training/
│   ├── __init__.py
│   ├── train_q_table.py      # Скрипт навчання Q-table
│   ├── train_dqn.py          # Скрипт навчання DQN
│   └── callbacks.py          # Callbacks для логування
│
├── experiments/
│   ├── __init__.py
│   ├── discount_analysis.py  # Дослідження впливу γ
│   ├── compare_strategies.py # Порівняння стратегій
│   └── multi_agent.py        # Багатоагентні експерименти
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py              # Графіки навчання
│   ├── game_recorder.py      # Запис відео гри
│   └── dashboard.py          # Інтерактивний дашборд
│
├── configs/
│   ├── default_env.yaml      # Конфігурація середовища
│   ├── training.yaml         # Параметри навчання
│   └── experiments.yaml      # Параметри експериментів
│
├── tests/
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_game_logic.py
│
├── notebooks/
│   └── analysis.ipynb        # Jupyter для аналізу результатів
│
└── results/
    ├── models/               # Збережені моделі
    ├── logs/                 # Логи навчання
    └── plots/                # Збережені графіки
```

---

## 📦 Залежності (requirements.txt)

```
gymnasium>=0.29.0
pygame>=2.5.0
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tensorboard>=2.13.0
tqdm>=4.65.0
pytest>=7.3.0
imageio>=2.31.0
```

---

## 🎮 ЧАСТИНА 1: Середовище (env/)

### 1.1 game_objects.py

```python
"""
Модуль з класами ігрових об'єктів.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple
import random

class ObjectType(Enum):
    """Типи об'єктів на полі."""
    APPLE = auto()       # Звичайне яблуко: +1 довжина, +10 очок
    GOLDEN = auto()      # Золоте: +3 довжина, +30-70 очок
    POISON = auto()      # Отрута: смерть
    SOUR = auto()        # Кисле: -1...-3 довжина, -5 очок
    ROTTEN = auto()      # Гниле: відриває хвіст → перешкода, -20 очок
    OBSTACLE = auto()    # Перешкода (відірваний хвіст)

@dataclass
class GameObject:
    """Базовий клас ігрового об'єкта."""
    x: int
    y: int
    object_type: ObjectType
    lifetime: int = -1  # -1 = вічний, інакше кількість кроків до зникнення
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

class ObjectFactory:
    """Фабрика для створення об'єктів з випадковими параметрами."""
    
    def __init__(self, spawn_probs: dict, grid_size: Tuple[int, int]):
        """
        Args:
            spawn_probs: {"apple": 0.5, "golden": 0.1, ...}
            grid_size: (width, height)
        """
        self.spawn_probs = spawn_probs
        self.grid_size = grid_size
        
        # Нормалізуємо ймовірності
        total = sum(spawn_probs.values())
        self.normalized_probs = {k: v/total for k, v in spawn_probs.items()}
    
    def create_random_object(self, occupied_positions: set) -> GameObject:
        """
        Створює випадковий об'єкт у вільній позиції.
        
        Args:
            occupied_positions: множина зайнятих позицій {(x, y), ...}
        
        Returns:
            GameObject або None якщо немає місця
        """
        # Знаходимо вільну позицію
        free_positions = [
            (x, y) 
            for x in range(self.grid_size[0]) 
            for y in range(self.grid_size[1])
            if (x, y) not in occupied_positions
        ]
        
        if not free_positions:
            return None
        
        x, y = random.choice(free_positions)
        
        # Вибираємо тип об'єкта
        obj_type = self._random_type()
        
        return GameObject(x=x, y=y, object_type=obj_type)
    
    def _random_type(self) -> ObjectType:
        """Вибирає випадковий тип згідно з ймовірностями."""
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
    """Обчислює винагороди за різні події."""
    
    # Базові винагороди
    REWARDS = {
        ObjectType.APPLE: 10,
        ObjectType.GOLDEN: (30, 70),  # випадково в діапазоні
        ObjectType.POISON: -1000,
        ObjectType.SOUR: -5,
        ObjectType.ROTTEN: -20,
    }
    
    # Додаткові винагороди
    DEATH_PENALTY = -1000
    STEP_PENALTY = -0.1      # штраф за кожен крок (стимулює активність)
    SURVIVAL_BONUS = 0.5     # бонус за виживання
    
    @classmethod
    def get_reward(cls, obj_type: ObjectType) -> float:
        """Повертає винагороду за з'їдання об'єкта."""
        reward = cls.REWARDS.get(obj_type, 0)
        
        if isinstance(reward, tuple):
            return random.uniform(reward[0], reward[1])
        
        return reward
    
    @classmethod
    def get_length_change(cls, obj_type: ObjectType) -> int:
        """Повертає зміну довжини змійки."""
        changes = {
            ObjectType.APPLE: 1,
            ObjectType.GOLDEN: 3,
            ObjectType.POISON: 0,  # смерть, не важливо
            ObjectType.SOUR: -random.randint(1, 3),  # випадково -1...-3
            ObjectType.ROTTEN: 0,  # обробляється окремо (відрив хвоста)
        }
        return changes.get(obj_type, 0)
```

### 1.2 snake.py

```python
"""
Модуль з класом змійки.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional
from collections import deque

class Direction(Enum):
    """Напрямки руху."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Action(Enum):
    """Дії агента (відносні)."""
    FORWARD = 0     # Рухатись прямо
    TURN_LEFT = 1   # Повернути ліворуч
    TURN_RIGHT = 2  # Повернути праворуч

class Snake:
    """Клас змійки."""
    
    # Відображення повороту: поточний напрямок → новий напрямок при повороті
    TURN_LEFT_MAP = {
        Direction.UP: Direction.LEFT,
        Direction.LEFT: Direction.DOWN,
        Direction.DOWN: Direction.RIGHT,
        Direction.RIGHT: Direction.UP,
    }
    
    TURN_RIGHT_MAP = {
        Direction.UP: Direction.RIGHT,
        Direction.RIGHT: Direction.DOWN,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
    }
    
    def __init__(self, start_pos: Tuple[int, int], start_length: int = 3, 
                 start_direction: Direction = Direction.RIGHT):
        """
        Args:
            start_pos: початкова позиція голови (x, y)
            start_length: початкова довжина
            start_direction: початковий напрямок
        """
        self.direction = start_direction
        self.grow_pending = 0  # скільки сегментів додати
        
        # Тіло як deque: [голова, ..., хвіст]
        self.body: deque = deque()
        
        # Ініціалізуємо тіло
        x, y = start_pos
        dx, dy = start_direction.value
        
        for i in range(start_length):
            self.body.append((x - i * dx, y - i * dy))
    
    @property
    def head(self) -> Tuple[int, int]:
        """Позиція голови."""
        return self.body[0]
    
    @property
    def tail(self) -> Tuple[int, int]:
        """Позиція хвоста."""
        return self.body[-1]
    
    @property
    def length(self) -> int:
        """Довжина змійки."""
        return len(self.body)
    
    def get_body_set(self) -> set:
        """Повертає множину позицій тіла (для швидкої перевірки колізій)."""
        return set(self.body)
    
    def apply_action(self, action: Action) -> None:
        """Змінює напрямок згідно з дією."""
        if action == Action.TURN_LEFT:
            self.direction = self.TURN_LEFT_MAP[self.direction]
        elif action == Action.TURN_RIGHT:
            self.direction = self.TURN_RIGHT_MAP[self.direction]
        # FORWARD - напрямок не змінюється
    
    def move(self) -> Tuple[int, int]:
        """
        Рухає змійку на один крок.
        
        Returns:
            Нова позиція голови
        """
        # Обчислюємо нову позицію голови
        hx, hy = self.head
        dx, dy = self.direction.value
        new_head = (hx + dx, hy + dy)
        
        # Додаємо нову голову
        self.body.appendleft(new_head)
        
        # Видаляємо хвіст (якщо не потрібно рости)
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()
        
        return new_head
    
    def grow(self, amount: int = 1) -> None:
        """Збільшує довжину на amount сегментів."""
        self.grow_pending += amount
    
    def shrink(self, amount: int) -> None:
        """
        Зменшує довжину на amount сегментів.
        Мінімальна довжина = 1 (тільки голова).
        """
        for _ in range(min(amount, len(self.body) - 1)):
            self.body.pop()
    
    def detach_tail(self, amount: int) -> List[Tuple[int, int]]:
        """
        Відриває amount сегментів хвоста.
        
        Returns:
            Список позицій відірваних сегментів (стануть перешкодами)
        """
        detached = []
        for _ in range(min(amount, len(self.body) - 1)):
            pos = self.body.pop()
            detached.append(pos)
        return detached
    
    def check_self_collision(self) -> bool:
        """Перевіряє чи голова зіткнулась з тілом."""
        return self.head in list(self.body)[1:]
```

### 1.3 snake_env.py

```python
"""
Gymnasium середовище для Snake+.
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
    Snake+ середовище з різними типами об'єктів.
    
    Спостереження (observation):
        Варіант 1 (для Q-table): вектор ознак
        Варіант 2 (для DQN): 2D матриця стану поля
    
    Дії (actions):
        0: Рухатись прямо
        1: Повернути ліворуч
        2: Повернути праворуч
    
    Винагороди:
        - Яблуко: +10
        - Золоте: +30...+70 (випадково)
        - Отрута: -1000 (смерть)
        - Кисле: -5
        - Гниле: -20
        - Кожен крок: -0.1
        - Смерть (стіна/тіло): -1000
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        spawn_probs: Optional[Dict[str, float]] = None,
        max_objects: int = 5,
        obstacle_decay: Optional[int] = 50,
        max_steps: int = 1000,
        observation_type: str = "features",  # "features" або "grid"
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            grid_size: розмір поля (width, height)
            spawn_probs: ймовірності появи об'єктів
            max_objects: максимум об'єктів на полі (без перешкод)
            obstacle_decay: через скільки кроків зникає перешкода (None = ніколи)
            max_steps: максимум кроків за епізод
            observation_type: тип спостереження
            render_mode: режим рендерингу
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_objects = max_objects
        self.obstacle_decay = obstacle_decay
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.render_mode = render_mode
        
        # Ймовірності за замовчуванням
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
        
        # Простір дій: 3 дії
        self.action_space = spaces.Discrete(3)
        
        # Простір спостережень
        if observation_type == "features":
            # Вектор ознак для Q-table
            # [danger_left, danger_straight, danger_right,  # 3
            #  dir_up, dir_down, dir_left, dir_right,       # 4
            #  food_up, food_down, food_left, food_right,   # 4
            #  nearest_obj_type (one-hot 5),                # 5
            #  distance_to_nearest_food,                    # 1
            #  snake_length_normalized]                     # 1
            # Всього: 18 ознак
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(18,), dtype=np.float32
            )
        else:  # "grid"
            # 3D матриця для CNN
            # Канали: [голова, тіло, яблуко, золоте, отрута, кисле, гниле, перешкода]
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(8, grid_size[1], grid_size[0]),
                dtype=np.float32
            )
        
        # Стан гри (ініціалізується в reset)
        self.snake: Optional[Snake] = None
        self.objects: List[GameObject] = []
        self.obstacles: List[GameObject] = []  # відірвані хвости
        self.score: int = 0
        self.steps: int = 0
        
        # Рендерер (ініціалізується при першому render)
        self.renderer = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Скидає середовище до початкового стану.
        
        Returns:
            observation: початкове спостереження
            info: додаткова інформація
        """
        super().reset(seed=seed)
        
        # Початкова позиція змійки (центр поля)
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
        
        # Спавнимо початкові об'єкти
        self._spawn_objects()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Виконує один крок.
        
        Args:
            action: 0=прямо, 1=ліворуч, 2=праворуч
        
        Returns:
            observation: нове спостереження
            reward: винагорода
            terminated: чи закінчилась гра (смерть)
            truncated: чи обрізано (max_steps)
            info: додаткова інформація
        """
        self.steps += 1
        reward = RewardCalculator.STEP_PENALTY  # штраф за крок
        terminated = False
        truncated = False
        
        # Застосовуємо дію
        self.snake.apply_action(Action(action))
        
        # Рухаємо змійку
        new_head = self.snake.move()
        
        # Перевіряємо колізії зі стінами
        if not self._is_valid_position(new_head):
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()
        
        # Перевіряємо колізію з тілом
        if self.snake.check_self_collision():
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()
        
        # Перевіряємо колізію з перешкодами
        obstacle_positions = {obs.position for obs in self.obstacles}
        if new_head in obstacle_positions:
            reward += RewardCalculator.DEATH_PENALTY
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self._get_info()
        
        # Перевіряємо колізію з об'єктами
        eaten_object = None
        for obj in self.objects:
            if obj.position == new_head:
                eaten_object = obj
                break
        
        if eaten_object:
            reward += self._process_eaten_object(eaten_object)
            
            # Отрута = смерть
            if eaten_object.object_type == ObjectType.POISON:
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self._get_info()
            
            # Перевіряємо чи змійка не стала занадто короткою
            if self.snake.length < 1:
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self._get_info()
        
        # Оновлюємо перешкоди (decay)
        self._update_obstacles()
        
        # Спавнимо нові об'єкти якщо потрібно
        self._spawn_objects()
        
        # Бонус за виживання
        reward += RewardCalculator.SURVIVAL_BONUS
        
        # Перевіряємо max_steps
        if self.steps >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_eaten_object(self, obj: GameObject) -> float:
        """
        Обробляє з'їдання об'єкта.
        
        Returns:
            Винагорода
        """
        reward = RewardCalculator.get_reward(obj.object_type)
        self.score += max(0, int(reward))
        
        # Видаляємо об'єкт
        self.objects.remove(obj)
        
        # Застосовуємо ефект
        if obj.object_type == ObjectType.APPLE:
            self.snake.grow(1)
        
        elif obj.object_type == ObjectType.GOLDEN:
            self.snake.grow(3)
        
        elif obj.object_type == ObjectType.SOUR:
            shrink_amount = random.randint(1, 3)
            self.snake.shrink(shrink_amount)
        
        elif obj.object_type == ObjectType.ROTTEN:
            # Відриваємо 3-5 сегментів хвоста
            detach_amount = random.randint(3, 5)
            detached_positions = self.snake.detach_tail(detach_amount)
            
            # Створюємо перешкоди
            for pos in detached_positions:
                obstacle = GameObject(
                    x=pos[0], y=pos[1],
                    object_type=ObjectType.OBSTACLE,
                    lifetime=self.obstacle_decay if self.obstacle_decay else -1
                )
                self.obstacles.append(obstacle)
        
        return reward
    
    def _update_obstacles(self) -> None:
        """Оновлює час життя перешкод, видаляє старі."""
        if self.obstacle_decay is None:
            return
        
        remaining = []
        for obs in self.obstacles:
            if obs.lifetime > 0:
                obs.lifetime -= 1
                if obs.lifetime > 0:
                    remaining.append(obs)
            elif obs.lifetime == -1:  # вічний
                remaining.append(obs)
        
        self.obstacles = remaining
    
    def _spawn_objects(self) -> None:
        """Спавнить об'єкти до max_objects."""
        occupied = self._get_occupied_positions()
        
        while len(self.objects) < self.max_objects:
            obj = self.object_factory.create_random_object(occupied)
            if obj is None:
                break  # немає місця
            
            self.objects.append(obj)
            occupied.add(obj.position)
    
    def _get_occupied_positions(self) -> set:
        """Повертає всі зайняті позиції."""
        occupied = self.snake.get_body_set()
        occupied.update(obj.position for obj in self.objects)
        occupied.update(obs.position for obs in self.obstacles)
        return occupied
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Перевіряє чи позиція в межах поля."""
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
    
    def _get_observation(self) -> np.ndarray:
        """Генерує спостереження."""
        if self.observation_type == "features":
            return self._get_feature_observation()
        else:
            return self._get_grid_observation()
    
    def _get_feature_observation(self) -> np.ndarray:
        """
        Генерує вектор ознак для Q-table агента.
        
        Ознаки:
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
        
        # Небезпека в 3 напрямках (відносно поточного напрямку)
        features[0] = self._is_danger(self._get_left_direction(direction))
        features[1] = self._is_danger(direction)  # прямо
        features[2] = self._is_danger(self._get_right_direction(direction))
        
        # Напрямок (one-hot)
        dir_idx = {Direction.UP: 3, Direction.DOWN: 4, 
                   Direction.LEFT: 5, Direction.RIGHT: 6}
        features[dir_idx[direction]] = 1
        
        # Напрямок до найближчої їжі
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
            
            # Відстань (normalized)
            dist = abs(nearest_food.x - head[0]) + abs(nearest_food.y - head[1])
            max_dist = self.grid_size[0] + self.grid_size[1]
            features[16] = dist / max_dist
        
        # Найближчий об'єкт будь-якого типу
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
        
        # Довжина змійки (normalized)
        max_length = self.grid_size[0] * self.grid_size[1]
        features[17] = self.snake.length / max_length
        
        return features
    
    def _get_grid_observation(self) -> np.ndarray:
        """
        Генерує 3D матрицю для CNN.
        
        Канали:
        0: голова
        1: тіло
        2: яблуко
        3: золоте
        4: отрута
        5: кисле
        6: гниле
        7: перешкода
        """
        grid = np.zeros((8, self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        
        # Голова
        hx, hy = self.snake.head
        grid[0, hy, hx] = 1
        
        # Тіло
        for i, (bx, by) in enumerate(self.snake.body):
            if i > 0:  # пропускаємо голову
                grid[1, by, bx] = 1
        
        # Об'єкти
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
        
        # Перешкоди
        for obs in self.obstacles:
            grid[7, obs.y, obs.x] = 1
        
        return grid
    
    def _is_danger(self, direction: Direction) -> float:
        """Перевіряє чи є небезпека в напрямку."""
        hx, hy = self.snake.head
        dx, dy = direction.value
        next_pos = (hx + dx, hy + dy)
        
        # Стіна
        if not self._is_valid_position(next_pos):
            return 1.0
        
        # Тіло
        if next_pos in self.snake.get_body_set():
            return 1.0
        
        # Перешкода
        if next_pos in {obs.position for obs in self.obstacles}:
            return 1.0
        
        return 0.0
    
    def _get_left_direction(self, d: Direction) -> Direction:
        """Повертає напрямок ліворуч від поточного."""
        return Snake.TURN_LEFT_MAP[d]
    
    def _get_right_direction(self, d: Direction) -> Direction:
        """Повертає напрямок праворуч від поточного."""
        return Snake.TURN_RIGHT_MAP[d]
    
    def _get_info(self) -> Dict[str, Any]:
        """Повертає додаткову інформацію."""
        return {
            "score": self.score,
            "length": self.snake.length,
            "steps": self.steps,
            "obstacles_count": len(self.obstacles),
        }
    
    def render(self):
        """Рендерить поточний стан."""
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
        """Закриває середовище."""
        if self.renderer:
            self.renderer.close()
```

### 1.4 renderer.py

```python
"""
Pygame рендерер для візуалізації гри.
"""

import pygame
import numpy as np
from typing import Tuple, List, Optional

from .snake import Snake
from .game_objects import GameObject, ObjectType

class Renderer:
    """Візуалізація гри за допомогою Pygame."""
    
    # Кольори
    COLORS = {
        "background": (30, 30, 40),
        "grid": (50, 50, 60),
        "snake_head": (0, 200, 100),
        "snake_body": (0, 150, 80),
        "apple": (220, 50, 50),
        "golden": (255, 215, 0),
        "poison": (20, 20, 20),
        "poison_skull": (200, 200, 200),
        "sour": (255, 255, 100),
        "rotten": (139, 90, 43),
        "obstacle": (100, 100, 120),
        "text": (255, 255, 255),
    }
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        cell_size: int = 30,
        render_mode: str = "human"
    ):
        """
        Args:
            grid_size: розмір поля (width, height)
            cell_size: розмір клітинки в пікселях
            render_mode: "human" або "rgb_array"
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        
        # Розміри вікна
        self.window_width = grid_size[0] * cell_size
        self.window_height = grid_size[1] * cell_size + 50  # +50 для інфо-панелі
        
        pygame.init()
        pygame.display.set_caption("Snake+ RL")
        
        if render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        else:
            self.screen = pygame.Surface(
                (self.window_width, self.window_height)
            )
        
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
    
    def render(
        self,
        snake: Snake,
        objects: List[GameObject],
        obstacles: List[GameObject],
        score: int,
        steps: int
    ) -> Optional[np.ndarray]:
        """
        Рендерить поточний стан гри.
        
        Returns:
            RGB array якщо render_mode == "rgb_array", інакше None
        """
        # Очищаємо екран
        self.screen.fill(self.COLORS["background"])
        
        # Малюємо сітку
        self._draw_grid()
        
        # Малюємо перешкоди
        for obs in obstacles:
            self._draw_cell(obs.x, obs.y, self.COLORS["obstacle"])
        
        # Малюємо об'єкти
        for obj in objects:
            self._draw_object(obj)
        
        # Малюємо змійку
        self._draw_snake(snake)
        
        # Малюємо інфо-панель
        self._draw_info(score, steps, snake.length)
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(10)  # 10 FPS
            return None
        else:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                (1, 0, 2)
            )
    
    def _draw_grid(self):
        """Малює сітку."""
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size[1] * self.cell_size)
            )
        
        for y in range(self.grid_size[1] + 1):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (0, y * self.cell_size),
                (self.window_width, y * self.cell_size)
            )
    
    def _draw_cell(self, x: int, y: int, color: Tuple[int, int, int], 
                   margin: int = 2):
        """Малює заповнену клітинку."""
        rect = pygame.Rect(
            x * self.cell_size + margin,
            y * self.cell_size + margin,
            self.cell_size - 2 * margin,
            self.cell_size - 2 * margin
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
    
    def _draw_snake(self, snake: Snake):
        """Малює змійку."""
        # Тіло
        for i, (x, y) in enumerate(snake.body):
            if i == 0:
                # Голова
                self._draw_cell(x, y, self.COLORS["snake_head"])
                # Очі
                self._draw_eyes(x, y, snake.direction)
            else:
                # Тіло з градієнтом
                ratio = i / len(snake.body)
                color = self._interpolate_color(
                    self.COLORS["snake_body"],
                    (0, 100, 50),
                    ratio
                )
                self._draw_cell(x, y, color)
    
    def _draw_eyes(self, x: int, y: int, direction):
        """Малює очі змійки."""
        from .snake import Direction
        
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + self.cell_size // 2
        
        eye_offsets = {
            Direction.UP: [(-5, -3), (5, -3)],
            Direction.DOWN: [(-5, 3), (5, 3)],
            Direction.LEFT: [(-3, -5), (-3, 5)],
            Direction.RIGHT: [(3, -5), (3, 5)],
        }
        
        for ox, oy in eye_offsets[direction]:
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (cx + ox, cy + oy),
                3
            )
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (cx + ox, cy + oy),
                1
            )
    
    def _draw_object(self, obj: GameObject):
        """Малює ігровий об'єкт."""
        x, y = obj.x, obj.y
        
        color_map = {
            ObjectType.APPLE: self.COLORS["apple"],
            ObjectType.GOLDEN: self.COLORS["golden"],
            ObjectType.POISON: self.COLORS["poison"],
            ObjectType.SOUR: self.COLORS["sour"],
            ObjectType.ROTTEN: self.COLORS["rotten"],
        }
        
        color = color_map.get(obj.object_type, (255, 255, 255))
        
        # Малюємо круг для яблук
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4
        
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        
        # Додаткові деталі для отрути (череп)
        if obj.object_type == ObjectType.POISON:
            # Простий череп
            pygame.draw.circle(
                self.screen,
                self.COLORS["poison_skull"],
                (cx, cy - 2),
                radius // 2
            )
            # Очі
            pygame.draw.circle(self.screen, (0, 0, 0), (cx - 3, cy - 3), 2)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx + 3, cy - 3), 2)
        
        # Блиск для золотого
        if obj.object_type == ObjectType.GOLDEN:
            pygame.draw.circle(
                self.screen,
                (255, 255, 200),
                (cx - 3, cy - 3),
                3
            )
    
    def _draw_info(self, score: int, steps: int, length: int):
        """Малює інформаційну панель."""
        y = self.grid_size[1] * self.cell_size + 10
        
        texts = [
            f"Score: {score}",
            f"Steps: {steps}",
            f"Length: {length}",
        ]
        
        x_positions = [10, 150, 290]
        
        for text, x in zip(texts, x_positions):
            surface = self.font.render(text, True, self.COLORS["text"])
            self.screen.blit(surface, (x, y))
    
    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        ratio: float
    ) -> Tuple[int, int, int]:
        """Інтерполює між двома кольорами."""
        return tuple(
            int(c1 + (c2 - c1) * ratio)
            for c1, c2 in zip(color1, color2)
        )
    
    def close(self):
        """Закриває pygame."""
        pygame.quit()
```

---

## 🤖 ЧАСТИНА 2: Агенти (agent/)

### 2.1 q_table_agent.py

```python
"""
Табличний Q-learning агент.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from collections import defaultdict

class QTableAgent:
    """
    Q-learning агент з таблицею Q-значень.
    
    Підходить для дискретизованого простору станів (features).
    """
    
    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        """
        Args:
            n_actions: кількість дій
            learning_rate: швидкість навчання (α)
            discount_factor: коефіцієнт дисконтування (γ)
            epsilon_start: початкове значення ε
            epsilon_end: мінімальне значення ε
            epsilon_decay: швидкість зменшення ε
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-таблиця як defaultdict
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )
        
        # Статистика
        self.training_steps = 0
    
    def discretize_state(self, observation: np.ndarray) -> Tuple:
        """
        Перетворює неперервне спостереження в дискретний ключ.
        
        Для feature observation (18 значень 0-1):
        - Бінаризуємо значення > 0.5
        """
        # Бінаризація
        discrete = tuple((observation > 0.5).astype(int))
        return discrete
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Вибирає дію за ε-greedy стратегією.
        
        Args:
            observation: спостереження
            training: чи в режимі навчання
        
        Returns:
            Індекс дії
        """
        state = self.discretize_state(observation)
        
        # ε-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Greedy
        q_values = self.q_table[state]
        
        # Якщо всі Q-значення однакові, вибираємо випадково
        if np.allclose(q_values, q_values[0]):
            return np.random.randint(self.n_actions)
        
        return np.argmax(q_values)
    
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> float:
        """
        Оновлює Q-таблицю.
        
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        
        Returns:
            TD error
        """
        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)
        
        # Поточне Q-значення
        current_q = self.q_table[state][action]
        
        # Цільове Q-значення
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD error
        td_error = target_q - current_q
        
        # Оновлення
        self.q_table[state][action] += self.lr * td_error
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        self.training_steps += 1
        
        return td_error
    
    def save(self, path: str):
        """Зберігає агента."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "params": {
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Завантажує агента."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["q_table"]
        )
        self.epsilon = data["epsilon"]
        self.training_steps = data["training_steps"]
    
    def get_stats(self) -> Dict:
        """Повертає статистику."""
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
```

### 2.2 replay_buffer.py

```python
"""
Experience Replay буфер для DQN.
"""

import numpy as np
from collections import deque
from typing import Tuple, List
import random

class ReplayBuffer:
    """Circular buffer для збереження досвіду."""
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: максимальний розмір буфера
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Додає досвід до буфера."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Випадково вибирає batch_size елементів.
        
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.
    Досвід з більшим TD error семплюється частіше.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Args:
            capacity: розмір буфера
            alpha: ступінь пріоритезації (0 = uniform, 1 = full priority)
            beta_start: початкове значення β для importance sampling
            beta_frames: скільки кроків до β = 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.frame = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Додає досвід з максимальним пріоритетом."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Семплює з урахуванням пріоритетів."""
        self.frame += 1
        
        # Обчислюємо β
        beta = min(1.0, self.beta_start + 
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)
        
        # Ймовірності
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Семплюємо індекси
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Збираємо batch
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Оновлює пріоритети."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self) -> int:
        return len(self.buffer)
```

### 2.3 networks.py

```python
"""
Нейронні мережі для DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DQN_MLP(nn.Module):
    """
    Багатошаровий персептрон для feature-based спостережень.
    """
    
    def __init__(
        self,
        input_size: int = 18,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        n_actions: int = 3
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQN_CNN(nn.Module):
    """
    CNN для grid-based спостережень.
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        grid_size: Tuple[int, int] = (15, 15),
        n_actions: int = 3
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Обчислюємо розмір після conv
        conv_out_size = self._get_conv_output_size(input_channels, grid_size)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_output_size(self, channels: int, grid_size: Tuple[int, int]) -> int:
        dummy = torch.zeros(1, channels, grid_size[1], grid_size[0])
        out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN архітектура.
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
    """
    
    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 128,
        n_actions: int = 3
    ):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
```

### 2.4 dqn_agent.py

```python
"""
Deep Q-Network агент.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from .networks import DQN_MLP, DQN_CNN, DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    """
    DQN агент з target network та experience replay.
    """
    
    def __init__(
        self,
        observation_type: str = "features",
        n_actions: int = 3,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        device: str = "auto"
    ):
        """
        Args:
            observation_type: "features" або "grid"
            n_actions: кількість дій
            learning_rate: швидкість навчання
            discount_factor: γ
            epsilon_start/end: параметри ε-greedy
            epsilon_decay_steps: за скільки кроків ε падає до мінімуму
            buffer_size: розмір replay buffer
            batch_size: розмір batch
            target_update_freq: частота оновлення target network
            use_double_dqn: чи використовувати Double DQN
            use_dueling: чи використовувати Dueling архітектуру
            use_prioritized_replay: чи використовувати PER
            device: "cpu", "cuda", або "auto"
        """
        # Визначаємо пристрій
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.n_actions = n_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        # Створюємо мережі
        if observation_type == "features":
            if use_dueling:
                self.q_network = DuelingDQN(n_actions=n_actions).to(self.device)
                self.target_network = DuelingDQN(n_actions=n_actions).to(self.device)
            else:
                self.q_network = DQN_MLP(n_actions=n_actions).to(self.device)
                self.target_network = DQN_MLP(n_actions=n_actions).to(self.device)
        else:
            self.q_network = DQN_CNN(n_actions=n_actions).to(self.device)
            self.target_network = DQN_CNN(n_actions=n_actions).to(self.device)
        
        # Копіюємо ваги
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Оптимізатор
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        self.use_prioritized_replay = use_prioritized_replay
        
        # Лічильники
        self.training_steps = 0
        self.updates = 0
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Вибирає дію за ε-greedy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Зберігає перехід в буфер."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[Dict]:
        """
        Виконує один крок навчання.
        
        Returns:
            Словник з метриками або None якщо буфер занадто малий
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Семплюємо batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Конвертуємо в тензори
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Поточні Q-значення
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Цільові Q-значення
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: вибираємо дію з q_network, оцінюємо з target
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # TD error
        td_errors = target_q - current_q
        
        # Loss (зважений для PER)
        loss = (weights * td_errors.pow(2)).mean()
        
        # Оновлюємо мережу
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        
        self.optimizer.step()
        
        # Оновлюємо пріоритети в PER
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(
                indices,
                td_errors.detach().cpu().numpy()
            )
        
        # Оновлюємо epsilon
        self._update_epsilon()
        
        # Оновлюємо target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_steps += 1
        
        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "epsilon": self.epsilon,
        }
    
    def _update_epsilon(self):
        """Оновлює epsilon за лінійним розкладом."""
        progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def save(self, path: str):
        """Зберігає модель."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "updates": self.updates,
        }, path)
    
    def load(self, path: str):
        """Завантажує модель."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.updates = checkpoint["updates"]
```

---

## 🏋️ ЧАСТИНА 3: Навчання (training/)

### 3.1 train_dqn.py

```python
"""
Скрипт навчання DQN агента.
"""

import argparse
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Імпорти проекту
import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent

def train(config: dict):
    """Основний цикл навчання."""
    
    # Створюємо директорію для результатів
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Зберігаємо конфіг
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Створюємо середовище
    env = SnakePlusEnv(
        grid_size=tuple(config["env"]["grid_size"]),
        spawn_probs=config["env"]["spawn_probs"],
        max_objects=config["env"]["max_objects"],
        obstacle_decay=config["env"].get("obstacle_decay"),
        max_steps=config["env"]["max_steps"],
        observation_type=config["env"]["observation_type"],
    )
    
    # Створюємо агента
    agent = DQNAgent(
        observation_type=config["env"]["observation_type"],
        n_actions=3,
        learning_rate=config["agent"]["learning_rate"],
        discount_factor=config["agent"]["discount_factor"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_end=config["agent"]["epsilon_end"],
        epsilon_decay_steps=config["agent"]["epsilon_decay_steps"],
        buffer_size=config["agent"]["buffer_size"],
        batch_size=config["agent"]["batch_size"],
        target_update_freq=config["agent"]["target_update_freq"],
        use_double_dqn=config["agent"]["use_double_dqn"],
        use_dueling=config["agent"]["use_dueling"],
        use_prioritized_replay=config["agent"]["use_prioritized_replay"],
    )
    
    # Метрики
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    losses = []
    
    # Параметри навчання
    n_episodes = config["training"]["n_episodes"]
    eval_freq = config["training"]["eval_freq"]
    save_freq = config["training"]["save_freq"]
    
    # Основний цикл
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Вибираємо дію
            action = agent.select_action(state, training=True)
            
            # Виконуємо крок
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Зберігаємо перехід
            agent.store_transition(state, action, reward, next_state, done)
            
            # Навчаємо агента
            metrics = agent.train_step()
            if metrics:
                losses.append(metrics["loss"])
            
            # Оновлюємо статистику
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Зберігаємо метрики епізоду
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info["score"])
        
        # Логування
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            print(f"\nEpisode {episode + 1}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
        
        # Зберігаємо модель
        if (episode + 1) % save_freq == 0:
            agent.save(run_dir / f"model_ep{episode + 1}.pt")
    
    # Зберігаємо фінальну модель
    agent.save(run_dir / "model_final.pt")
    
    # Зберігаємо метрики
    np.savez(
        run_dir / "metrics.npz",
        rewards=episode_rewards,
        lengths=episode_lengths,
        scores=episode_scores,
        losses=losses,
    )
    
    # Будуємо графіки
    plot_training_curves(episode_rewards, episode_scores, losses, run_dir)
    
    env.close()
    print(f"\nTraining complete! Results saved to {run_dir}")


def plot_training_curves(rewards, scores, losses, save_dir):
    """Будує графіки навчання."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3)
    axes[0, 0].plot(moving_average(rewards, 100))
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    
    # Scores
    axes[0, 1].plot(scores, alpha=0.3)
    axes[0, 1].plot(moving_average(scores, 100))
    axes[0, 1].set_title("Episode Scores")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Score")
    
    # Losses
    axes[1, 0].plot(losses, alpha=0.3)
    axes[1, 0].plot(moving_average(losses, 1000))
    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    
    # Histogram of final rewards
    axes[1, 1].hist(rewards[-1000:], bins=50)
    axes[1, 1].set_title("Reward Distribution (last 1000)")
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()


def moving_average(data, window):
    """Обчислює ковзне середнє."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train(config)
```

---

## 🔬 ЧАСТИНА 4: Експерименти (experiments/)

### 4.1 discount_analysis.py

```python
"""
Дослідження впливу коефіцієнта дисконтування на стратегію.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import List, Dict

# Імпорти проекту
import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent
from training.train_dqn import train

def run_discount_experiment(
    gamma_values: List[float] = [0.1, 0.5, 0.9, 0.99, 0.999],
    n_episodes: int = 5000,
    n_eval_episodes: int = 100,
    base_config_path: str = "configs/training.yaml"
):
    """
    Запускає експеримент з різними значеннями γ.
    
    Гіпотеза:
    - Низький γ (0.1-0.5): агент уникає довгострокового планування,
      частіше їсть гнилі яблука, швидко гине від перешкод
    - Високий γ (0.9-0.999): агент уникає гнилих яблук,
      довше виживає, накопичує більше очок
    """
    results = {}
    
    # Завантажуємо базовий конфіг
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    for gamma in gamma_values:
        print(f"\n{'='*50}")
        print(f"Training with γ = {gamma}")
        print(f"{'='*50}")
        
        # Модифікуємо конфіг
        config = base_config.copy()
        config["agent"]["discount_factor"] = gamma
        config["training"]["n_episodes"] = n_episodes
        
        # Навчаємо агента
        agent, env = train_and_return(config)
        
        # Оцінюємо
        eval_results = evaluate_agent(agent, env, n_eval_episodes)
        
        results[gamma] = {
            "mean_score": np.mean(eval_results["scores"]),
            "mean_length": np.mean(eval_results["lengths"]),
            "mean_steps": np.mean(eval_results["steps"]),
            "rotten_eaten": np.mean(eval_results["rotten_eaten"]),
            "death_by_obstacle": np.mean(eval_results["death_by_obstacle"]),
            "survival_rate": np.mean(eval_results["survived"]),
        }
        
        print(f"Results for γ = {gamma}:")
        for k, v in results[gamma].items():
            print(f"  {k}: {v:.3f}")
    
    # Візуалізація
    plot_discount_analysis(results)
    
    return results


def evaluate_agent(
    agent: DQNAgent,
    env: SnakePlusEnv,
    n_episodes: int
) -> Dict[str, List]:
    """Оцінює агента."""
    results = {
        "scores": [],
        "lengths": [],
        "steps": [],
        "rotten_eaten": [],
        "death_by_obstacle": [],
        "survived": [],
    }
    
    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        rotten_count = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Рахуємо гнилі яблука (за винагородою)
            if reward == -20:  # ROTTEN reward
                rotten_count += 1
        
        results["scores"].append(info["score"])
        results["lengths"].append(info["length"])
        results["steps"].append(info["steps"])
        results["rotten_eaten"].append(rotten_count)
        results["death_by_obstacle"].append(
            1 if info["obstacles_count"] > 0 and terminated else 0
        )
        results["survived"].append(1 if truncated else 0)
    
    return results


def plot_discount_analysis(results: Dict[float, Dict]):
    """Будує графіки аналізу."""
    gammas = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ("mean_score", "Average Score"),
        ("mean_steps", "Average Survival Steps"),
        ("rotten_eaten", "Avg Rotten Apples Eaten"),
        ("death_by_obstacle", "Death by Obstacle Rate"),
        ("survival_rate", "Survival Rate (reached max steps)"),
        ("mean_length", "Average Final Length"),
    ]
    
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        values = [results[g][metric] for g in gammas]
        ax.bar(range(len(gammas)), values, tick_label=[str(g) for g in gammas])
        ax.set_title(title)
        ax.set_xlabel("Discount Factor (γ)")
        ax.set_ylabel(metric)
    
    plt.suptitle("Impact of Discount Factor on Agent Behavior", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/discount_analysis.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_discount_experiment()
```

---

## 📊 ЧАСТИНА 5: Конфігурації (configs/)

### 5.1 default_env.yaml

```yaml
# Конфігурація середовища Snake+

grid_size: [15, 15]

spawn_probs:
  apple: 0.50
  golden: 0.10
  poison: 0.15
  sour: 0.15
  rotten: 0.10

max_objects: 5
obstacle_decay: 50  # null для вічних перешкод
max_steps: 1000
observation_type: "features"  # або "grid"
```

### 5.2 training.yaml

```yaml
# Конфігурація навчання

env:
  grid_size: [15, 15]
  spawn_probs:
    apple: 0.50
    golden: 0.10
    poison: 0.15
    sour: 0.15
    rotten: 0.10
  max_objects: 5
  obstacle_decay: 50
  max_steps: 1000
  observation_type: "features"

agent:
  learning_rate: 0.0001
  discount_factor: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 50000
  buffer_size: 100000
  batch_size: 64
  target_update_freq: 1000
  use_double_dqn: true
  use_dueling: false
  use_prioritized_replay: false

training:
  n_episodes: 10000
  eval_freq: 500
  save_freq: 1000
```

---

## 🎯 ЧАСТИНА 6: Візуалізація (visualization/)

### 6.1 dashboard.py

```python
"""
Інтерактивний дашборд для демонстрації.
"""

import pygame
import numpy as np
from pathlib import Path

# Імпорти проекту
import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent

class Dashboard:
    """
    Інтерактивний дашборд для:
    - Гри вручну
    - Спостереження за агентом
    - Зміни параметрів в реальному часі
    """
    
    def __init__(self, config_path: str = "configs/training.yaml"):
        pygame.init()
        
        # Розміри вікна
        self.game_width = 450  # 15 * 30
        self.panel_width = 300
        self.window_width = self.game_width + self.panel_width
        self.window_height = 500
        
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption("Snake+ RL Dashboard")
        
        # Шрифти
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Стан
        self.mode = "agent"  # "agent" або "human"
        self.paused = False
        self.speed = 10  # FPS
        
        # Завантажуємо конфіг
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Створюємо середовище
        self.env = SnakePlusEnv(
            grid_size=tuple(self.config["env"]["grid_size"]),
            render_mode="rgb_array"
        )
        
        # Агент (буде завантажено)
        self.agent = None
        
        # Статистика
        self.stats = {
            "episodes": 0,
            "total_score": 0,
            "best_score": 0,
            "avg_score": 0,
        }
        
        self.clock = pygame.time.Clock()
    
    def load_agent(self, model_path: str):
        """Завантажує навченого агента."""
        self.agent = DQNAgent(
            observation_type=self.config["env"]["observation_type"],
            discount_factor=self.config["agent"]["discount_factor"],
        )
        self.agent.load(model_path)
        self.agent.epsilon = 0  # Вимикаємо exploration
    
    def run(self):
        """Запускає дашборд."""
        state, _ = self.env.reset()
        running = True
        
        while running:
            # Обробка подій
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_m:
                        self.mode = "human" if self.mode == "agent" else "agent"
                    elif event.key == pygame.K_r:
                        state, _ = self.env.reset()
                    elif event.key == pygame.K_UP:
                        self.speed = min(60, self.speed + 5)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(1, self.speed - 5)
            
            if not self.paused:
                # Вибираємо дію
                if self.mode == "agent" and self.agent:
                    action = self.agent.select_action(state, training=False)
                else:
                    # Керування людиною
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT]:
                        action = 1  # Turn left
                    elif keys[pygame.K_RIGHT]:
                        action = 2  # Turn right
                    else:
                        action = 0  # Forward
                
                # Крок
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated or truncated:
                    self.stats["episodes"] += 1
                    self.stats["total_score"] += info["score"]
                    self.stats["best_score"] = max(
                        self.stats["best_score"], 
                        info["score"]
                    )
                    self.stats["avg_score"] = (
                        self.stats["total_score"] / self.stats["episodes"]
                    )
                    state, _ = self.env.reset()
                else:
                    state = next_state
            
            # Рендеринг
            self._render(info if 'info' in dir() else {"score": 0, "length": 3})
            
            self.clock.tick(self.speed)
        
        pygame.quit()
    
    def _render(self, info: dict):
        """Рендерить весь інтерфейс."""
        # Очищаємо
        self.screen.fill((30, 30, 40))
        
        # Гра
        game_surface = self.env.render()
        if game_surface is not None:
            game_surface = pygame.surfarray.make_surface(
                game_surface.swapaxes(0, 1)
            )
            self.screen.blit(game_surface, (0, 0))
        
        # Панель управління
        self._draw_panel(info)
        
        pygame.display.flip()
    
    def _draw_panel(self, info: dict):
        """Малює панель з інформацією."""
        x = self.game_width + 10
        y = 10
        
        # Заголовок
        title = self.font_large.render("Snake+ Dashboard", True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        y += 40
        
        # Режим
        mode_text = f"Mode: {self.mode.upper()}"
        mode_color = (100, 200, 100) if self.mode == "agent" else (200, 200, 100)
        self.screen.blit(
            self.font.render(mode_text, True, mode_color), (x, y)
        )
        y += 30
        
        # Статус
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (200, 100, 100) if self.paused else (100, 200, 100)
        self.screen.blit(
            self.font.render(f"Status: {status}", True, status_color), (x, y)
        )
        y += 30
        
        # Швидкість
        self.screen.blit(
            self.font.render(f"Speed: {self.speed} FPS", True, (200, 200, 200)), 
            (x, y)
        )
        y += 40
        
        # Поточна гра
        self.screen.blit(
            self.font_large.render("Current Game", True, (255, 255, 255)), (x, y)
        )
        y += 30
        self.screen.blit(
            self.font.render(f"Score: {info.get('score', 0)}", True, (200, 200, 200)), 
            (x, y)
        )
        y += 25
        self.screen.blit(
            self.font.render(f"Length: {info.get('length', 3)}", True, (200, 200, 200)), 
            (x, y)
        )
        y += 40
        
        # Статистика
        self.screen.blit(
            self.font_large.render("Statistics", True, (255, 255, 255)), (x, y)
        )
        y += 30
        self.screen.blit(
            self.font.render(f"Episodes: {self.stats['episodes']}", True, (200, 200, 200)), 
            (x, y)
        )
        y += 25
        self.screen.blit(
            self.font.render(f"Best Score: {self.stats['best_score']}", True, (200, 200, 200)), 
            (x, y)
        )
        y += 25
        self.screen.blit(
            self.font.render(f"Avg Score: {self.stats['avg_score']:.1f}", True, (200, 200, 200)), 
            (x, y)
        )
        y += 40
        
        # Управління
        self.screen.blit(
            self.font_large.render("Controls", True, (255, 255, 255)), (x, y)
        )
        y += 30
        controls = [
            "SPACE - Pause/Resume",
            "M - Toggle Mode",
            "R - Reset Game",
            "↑/↓ - Speed",
            "←/→ - Turn (Human)",
        ]
        for ctrl in controls:
            self.screen.blit(
                self.font.render(ctrl, True, (150, 150, 150)), (x, y)
            )
            y += 22


if __name__ == "__main__":
    dashboard = Dashboard()
    
    # Опціонально завантажуємо модель
    import sys
    if len(sys.argv) > 1:
        dashboard.load_agent(sys.argv[1])
    
    dashboard.run()
```

---

## ✅ Чекліст виконання

### Фаза 1: Базова структура ✅
- [x] Створити структуру папок
- [x] Написати `requirements.txt`
- [x] Реалізувати `game_objects.py` — ObjectType (6 типів), GameObject, ObjectFactory, RewardCalculator
- [x] Реалізувати `snake.py` — Direction, Action, Snake з deque-based body, grow/shrink/detach_tail
- [x] Написати unit-тести для game logic — test_game_logic.py (24+ тестів)

### Фаза 2: Середовище ✅
- [x] Реалізувати `snake_env.py` — Gymnasium env, Discrete(3) actions, feature (18-dim) та grid (8×15×15) observations
- [x] Реалізувати `renderer.py` — Pygame візуалізація з кольоровими об'єктами, info panel, human/rgb_array modes
- [x] Протестувати середовище з випадковим агентом — test_env.py (13+ тестів)
- [ ] Зареєструвати в Gymnasium

### Фаза 3: Агенти ✅
- [x] Реалізувати `q_table_agent.py` — табличний Q-learning, ε-greedy, discretization, save/load
- [x] Реалізувати `replay_buffer.py` — ReplayBuffer + PrioritizedReplayBuffer з importance sampling
- [x] Реалізувати `networks.py` — DQN_MLP, DQN_CNN, DuelingDQN архітектури
- [x] Реалізувати `dqn_agent.py` — Double DQN, target network, PER, gradient clipping, CUDA support
- [x] Написати unit-тести для агентів — test_agent.py (10+ тестів)
- [x] Виправлено dropout під час inference в select_action()
- [x] Виправлено dropout в train_step для target network
- [x] Виправлено off-by-one помилку в obstacle lifetime decay

### Фаза 4: Навчання
- [ ] Написати `train_dqn.py`
- [ ] Створити конфіги
- [ ] Запустити перше навчання
- [ ] Налаштувати гіперпараметри

### Фаза 5: Експерименти
- [ ] Реалізувати `discount_analysis.py`
- [ ] Провести експерименти з різними γ
- [ ] Задокументувати результати

### Фаза 6: Візуалізація та документація
- [ ] Реалізувати `dashboard.py`
- [ ] Записати відео демонстрації
- [x] Написати README.md
- [ ] Підготувати звіт

---

## 📝 Примітки для реалізації

1. **Починай з тестів** - напиши тести для game logic перед реалізацією

2. **Ітеративна розробка** - спочатку проста версія, потім ускладнюй

3. **Версіонування** - використовуй git, коміть часто

4. **Логування** - використовуй TensorBoard для візуалізації навчання

5. **Reproducibility** - фіксуй random seeds

6. **Документація** - docstrings для всіх функцій

7. **Профілювання** - якщо повільно, профілюй і оптимізуй
