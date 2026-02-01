# Snake+ RL

A Snake game with special objects and Reinforcement Learning agents that learn to play it. The project investigates the impact of the discount factor (gamma) on learning strategies.

## Project Structure

```
snake_plus/
├── env/                      # Game environment
│   ├── game_objects.py       # Object classes (apples, poison, etc.)
│   ├── snake.py              # Snake class
│   ├── snake_env.py          # Gymnasium environment
│   └── renderer.py           # Pygame visualization
│
├── agent/                    # RL agents
│   ├── q_table_agent.py      # Tabular Q-learning agent
│   ├── dqn_agent.py          # Deep Q-Network agent
│   ├── replay_buffer.py      # Experience replay buffer
│   └── networks.py           # Neural networks (MLP, CNN, Dueling DQN)
│
├── training/                 # Training scripts
│   └── train_dqn.py         # DQN training loop with evaluation & plotting
│
├── configs/                  # YAML configuration files
│   ├── default_env.yaml     # Environment defaults
│   └── training.yaml        # Training hyperparameters
│
├── experiments/              # Experimental analysis (Phase 5)
├── visualization/            # Visualization tools (Phase 6)
├── tests/                    # Unit tests
└── results/                  # Output storage
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Gymnasium 0.29+
- Pygame 2.5+
- NumPy, Matplotlib, PyYAML, tqdm

## Installation

```bash
cd snake_plus
pip install -r requirements.txt
```

## Quick Start

### Run Tests

```bash
cd snake_plus
python -m pytest tests/ -v
```

### Use the Environment Programmatically

```python
from env.snake_env import SnakePlusEnv

# Create environment
env = SnakePlusEnv(
    grid_size=(15, 15),
    observation_type="features",  # or "grid" for CNN
    max_steps=1000,
)

# Run a random agent
obs, info = env.reset(seed=42)
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Score: {info['score']}, Steps: {info['steps']}, Length: {info['length']}")
env.close()
```

### Use the Q-Table Agent

```python
from env.snake_env import SnakePlusEnv
from agent.q_table_agent import QTableAgent

env = SnakePlusEnv(observation_type="features")
agent = QTableAgent(
    n_actions=3,
    learning_rate=0.1,
    discount_factor=0.99,
)

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs

    if (episode + 1) % 100 == 0:
        stats = agent.get_stats()
        print(f"Episode {episode+1} | Q-table size: {stats['q_table_size']} | Epsilon: {stats['epsilon']:.3f}")

# Save the trained agent
agent.save("results/models/q_table_agent.pkl")
```

### Use the DQN Agent

```python
from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent

env = SnakePlusEnv(observation_type="features")
agent = DQNAgent(
    observation_type="features",
    n_actions=3,
    learning_rate=1e-4,
    discount_factor=0.99,
    use_double_dqn=True,
)

# Training loop
for episode in range(5000):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.train_step()
        obs = next_obs
        episode_reward += reward

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1} | Reward: {episode_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

# Save the trained model
agent.save("results/models/dqn_model.pt")
```

### Train with the Training Script

```bash
cd snake_plus
python -m training.train_dqn --config configs/training.yaml
```

Results (model checkpoints, metrics, training curves) are saved to `results/runs/<timestamp>/`.

To use a custom config, copy `configs/training.yaml`, edit the values, and pass the path:

```bash
python -m training.train_dqn --config configs/my_config.yaml
```

### Render the Game (requires display)

```python
from env.snake_env import SnakePlusEnv

env = SnakePlusEnv(render_mode="human")
obs, info = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Game Objects

| Object  | Effect                          | Reward   |
|---------|---------------------------------|----------|
| Apple   | +1 length                       | +10      |
| Golden  | +3 length                       | +30..+70 |
| Poison  | Instant death                   | -1000    |
| Sour    | -1 to -3 length                 | -5       |
| Rotten  | Detaches 3-5 tail segments (become obstacles) | -20 |

## Observation Types

- **features** (18-dim vector): danger signals, direction, food direction, nearest object type, distance, snake length. Suitable for Q-table and MLP-based DQN.
- **grid** (8x15x15 tensor): multi-channel grid with separate channels for head, body, and each object type. Suitable for CNN-based DQN.

## Implemented Phases

- **Phase 1**: Basic structure, game objects, snake logic, unit tests
- **Phase 2**: Gymnasium environment, Pygame renderer
- **Phase 3**: Q-table agent, DQN agent (with Double DQN, Dueling, PER support), replay buffers, neural networks
- **Phase 4**: Training script with CLI, YAML configs, evaluation, metric logging, training curve plots
