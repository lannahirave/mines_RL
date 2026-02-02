# Snake+ : Детальний план реалізації

## Огляд проекту

**Мета:** Створити гру Snake з особливими об'єктами та навчити RL-агента грати в неї, досліджуючи вплив дисконтування на стратегії.

**Технології:** Python 3.10+, Gymnasium, PyTorch, Pygame, NumPy, Matplotlib

---

## Структура проекту

```
├── README.md
├── CLAUDE.md
├── requirements.txt
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
│   └── train_dqn.py          # Скрипт навчання DQN
│
├── experiments/
│   └── __init__.py
│
├── visualization/
│   └── __init__.py
│
├── configs/
│   ├── default_env.yaml      # Конфігурація середовища
│   └── training.yaml         # Параметри навчання
│
├── tests/
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_game_logic.py
│
└── results/
    └── runs/                 # Результати навчання (per-run directories)
```

---

## ЧАСТИНА 1: Середовище (env/) — Реалізовано

### 1.1 game_objects.py

Реалізовано модуль ігрових об'єктів:

- **ObjectType** (Enum) — 6 типів: APPLE, GOLDEN, POISON, SOUR, ROTTEN, OBSTACLE
- **GameObject** (dataclass) — базовий об'єкт з позицією, типом і lifetime
- **ObjectFactory** — фабрика для створення випадкових об'єктів з нормалізованими ймовірностями, вибір вільної позиції на полі
- **RewardCalculator** — обчислення винагород та зміни довжини за типом об'єкта

Винагороди: Apple +10, Golden +30..+70, Poison -1000 (смерть), Sour -5, Rotten -20. Штраф за крок: -0.1, бонус за виживання: +0.01, смерть (стіна/тіло/перешкода): -1000.

### 1.2 snake.py

Реалізовано клас змійки:

- **Direction** (Enum) — UP, DOWN, LEFT, RIGHT з координатними значеннями
- **Action** (Enum) — FORWARD (0), TURN_LEFT (1), TURN_RIGHT (2)
- **Snake** — deque-based тіло, apply_action (відносні повороти через lookup maps), move, grow, shrink, detach_tail (для гнилих яблук → перешкоди), check_self_collision

### 1.3 snake_env.py

Реалізовано Gymnasium середовище **SnakePlusEnv**:

- Discrete(3) action space (forward, turn left, turn right)
- Два типи спостережень:
  - `"features"` — 18-dim вектор: danger (3), direction one-hot (4), food direction (4), nearest object type one-hot (5), distance to food (1), snake length (1)
  - `"grid"` — 8-channel tensor (8 × height × width): head, body, apple, golden, poison, sour, rotten, obstacle
- Повний step() з обробкою колізій (стіни, тіло, перешкоди, об'єкти), obstacle decay, object spawning
- reset() з ініціалізацією змійки в центрі поля

### 1.4 renderer.py

Реалізовано Pygame рендерер:

- Підтримка `"human"` та `"rgb_array"` render modes
- Кольорова візуалізація: gradient для тіла змійки, очі на голові, спеціальні деталі для об'єктів (череп для отрути, блиск для золотого)
- Інформаційна панель (Score, Steps, Length)

---

## ЧАСТИНА 2: Агенти (agent/) — Реалізовано

### 2.1 q_table_agent.py

Реалізовано табличний Q-learning агент **QTableAgent**:

- defaultdict-based Q-таблиця з бінарною дискретизацією спостережень (threshold 0.5)
- epsilon-greedy вибір дій з exponential decay
- Стандартне Q-learning оновлення: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
- save/load через pickle, статистика (q_table_size, epsilon, training_steps)

### 2.2 replay_buffer.py

Реалізовано два replay буфери:

- **ReplayBuffer** — circular buffer на deque, uniform sampling
- **PrioritizedReplayBuffer** — priority-based sampling з alpha (prioritization degree), beta (importance sampling correction з анілінгом), update_priorities по TD error

### 2.3 networks.py

Реалізовано три нейронні мережі:

- **DQN_MLP** — Multi-layer perceptron для feature-based спостережень (18 → hidden layers → 3 actions), з Dropout(0.1)
- **DQN_CNN** — CNN для grid-based спостережень (Conv2d layers → FC → 3 actions)
- **DuelingDQN** — Dueling архітектура: shared features → Value stream + Advantage stream → Q = V + A - mean(A)

### 2.4 dqn_agent.py

Реалізовано DQN агент **DQNAgent**:

- Auto device selection (CUDA/CPU)
- Автоматичний вибір мережі за observation_type та use_dueling
- Target network з periodic hard updates
- Double DQN (вибір дії з q_network, оцінка з target_network)
- Prioritized Experience Replay (опціонально)
- Лінійний epsilon decay, gradient clipping (max_norm=10)
- save/load checkpoints (q_network, target_network, optimizer, epsilon, steps)

Виправлені помилки:
- Dropout під час inference в select_action() (eval mode)
- Dropout в train_step для target network
- Off-by-one в obstacle lifetime decay

---

## ЧАСТИНА 3: Навчання (training/) — Реалізовано

### 3.1 train_dqn.py

Реалізовано скрипт навчання:

- CLI з argparse (--config для YAML файлу)
- Основний цикл: episode loop → step loop → store_transition → train_step
- Periodic logging (кожні 100 епізодів): avg reward, score, length, epsilon
- Model checkpointing (кожні save_freq епізодів)
- Збереження метрик в .npz (rewards, lengths, scores, losses)
- Побудова графіків навчання (rewards, scores, losses, reward distribution)
- Результати зберігаються в `results/runs/<timestamp>/`

---

## ЧАСТИНА 4: Конфігурації (configs/) — Реалізовано

### training.yaml

Три секції:
- **env**: grid_size [15,15], spawn_probs, max_objects 5, obstacle_decay 50, max_steps 1000, observation_type "features"
- **agent**: learning_rate 0.0001, discount_factor 0.99, epsilon schedule (1.0→0.01 за 50000 кроків), buffer_size 100000, batch_size 64, target_update_freq 1000, double_dqn on, dueling off, PER off
- **training**: n_episodes 10000, eval_freq 500, save_freq 1000

### default_env.yaml

Окрема конфігурація середовища з тими ж параметрами env.

---

## ЧАСТИНА 5: Тестування (tests/) — Реалізовано

- **test_game_logic.py** — 24+ тестів для game objects, snake mechanics, factory, reward calculator
- **test_env.py** — 13+ тестів для Gymnasium середовища (reset, step, observations, collisions, object interactions)
- **test_agent.py** — 10+ тестів для Q-table та DQN агентів

---

## Виконаний чекліст

### Фаза 1: Базова структура
- [x] Створити структуру папок
- [x] Написати `requirements.txt`
- [x] Реалізувати `game_objects.py` — ObjectType (6 типів), GameObject, ObjectFactory, RewardCalculator
- [x] Реалізувати `snake.py` — Direction, Action, Snake з deque-based body, grow/shrink/detach_tail
- [x] Написати unit-тести для game logic — test_game_logic.py (24+ тестів)

### Фаза 2: Середовище
- [x] Реалізувати `snake_env.py` — Gymnasium env, Discrete(3) actions, feature (18-dim) та grid (8×15×15) observations
- [x] Реалізувати `renderer.py` — Pygame візуалізація з кольоровими об'єктами, info panel, human/rgb_array modes
- [x] Протестувати середовище з випадковим агентом — test_env.py (13+ тестів)

### Фаза 3: Агенти
- [x] Реалізувати `q_table_agent.py` — табличний Q-learning, ε-greedy, discretization, save/load
- [x] Реалізувати `replay_buffer.py` — ReplayBuffer + PrioritizedReplayBuffer з importance sampling
- [x] Реалізувати `networks.py` — DQN_MLP, DQN_CNN, DuelingDQN архітектури
- [x] Реалізувати `dqn_agent.py` — Double DQN, target network, PER, gradient clipping, CUDA support
- [x] Написати unit-тести для агентів — test_agent.py (10+ тестів)
- [x] Виправлено dropout під час inference в select_action()
- [x] Виправлено dropout в train_step для target network
- [x] Виправлено off-by-one помилку в obstacle lifetime decay

### Фаза 4: Навчання
- [x] Написати `train_dqn.py` — CLI з --config, metric logging, checkpointing
- [x] Створити конфіги — training.yaml, default_env.yaml
- [x] Написати README.md

---

## TODO

### Реєстрація середовища
- [ ] Зареєструвати SnakePlusEnv в Gymnasium (через `gymnasium.register()`)

### Навчання та тюнінг
- [ ] Запустити перше навчання DQN агента
- [ ] Налаштувати гіперпараметри (learning rate, epsilon schedule, buffer size, тощо)

### Експерименти (experiments/)
- [ ] Реалізувати `discount_analysis.py` — дослідження впливу коефіцієнта дисконтування (γ) на стратегію агента
  - Запуск навчання з різними γ (0.1, 0.5, 0.9, 0.99, 0.999)
  - Збір метрик: mean score, survival steps, rotten apples eaten, death by obstacle rate, survival rate, final length
  - Візуалізація порівняння стратегій
- [ ] Провести експерименти з різними γ
- [ ] Задокументувати результати

### Візуалізація (visualization/)
- [ ] Реалізувати `dashboard.py` — інтерактивний Pygame дашборд
  - Два режими: спостереження за агентом / гра вручну
  - Панель статистики (episodes, best/avg score)
  - Управління: пауза, зміна швидкості, переключення режиму, reset
  - Завантаження навченої моделі
- [ ] Записати відео демонстрації

### Документація та звіт
- [ ] Підготувати фінальний звіт з результатами експериментів
