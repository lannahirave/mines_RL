# Training Guide

Step-by-step instructions for training Snake+ RL agents and running experiments.

## Prerequisites

Python 3.10+ and uv:

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify the setup:

```
uv run python -m pytest tests/ -v
```

All 80 tests should pass.

## 1. Basic DQN Training

Train a DQN agent with the default config:

```
uv run python -m training.train_dqn --config configs/training.yaml
```

Default settings: 10,000 episodes, 15x15 grid, feature observations, Double DQN enabled, epsilon decaying from 1.0 to 0.01 over 50k steps.

Output goes to `results/runs/<timestamp>/` containing:
- `model_final.pt` — final trained model
- `model_ep*.pt` — intermediate checkpoints (every 1,000 episodes)
- `metrics.npz` — raw training metrics (rewards, scores, losses)
- `training_curves.png` — reward, score, loss, and distribution plots
- `config.yaml` — copy of the config used

Progress is logged every 100 episodes. Greedy evaluation runs every 500 episodes.

## 2. Custom Configurations

Copy and edit the config to experiment with different settings:

```
cp configs/training.yaml configs/my_config.yaml
# edit configs/my_config.yaml
uv run python -m training.train_dqn --config configs/my_config.yaml
```

### Key parameters to tune

**Environment** (`env` section):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | [15, 15] | Width and height of the game grid |
| `observation_type` | "features" | "features" (18-dim vector, MLP) or "grid" (8-channel tensor, CNN) |
| `max_objects` | 5 | Maximum objects on the field at once |
| `obstacle_decay` | 50 | Steps until rotten-fruit obstacles disappear |
| `max_steps` | 1000 | Episode truncation limit |

**Agent** (`agent` section):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0001 | Adam optimizer learning rate |
| `discount_factor` | 0.99 | Gamma — how much the agent values future rewards |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay_steps` | 50000 | Steps over which epsilon decays linearly |
| `buffer_size` | 100000 | Replay buffer capacity |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 1000 | Steps between target network updates |
| `use_double_dqn` | true | Reduces Q-value overestimation |
| `use_dueling` | false | Separate value/advantage streams |
| `use_prioritized_replay` | false | Focus training on high-error transitions |

**Training** (`training` section):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_episodes` | 10000 | Total training episodes |
| `eval_freq` | 500 | Episodes between greedy evaluations |
| `save_freq` | 1000 | Episodes between model checkpoints |
| `seed` | 42 | Random seed for reproducibility |

### Example: CNN agent with Dueling DQN and PER

```yaml
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
  observation_type: "grid"

agent:
  learning_rate: 0.0001
  discount_factor: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 100000
  buffer_size: 200000
  batch_size: 128
  target_update_freq: 2000
  use_double_dqn: true
  use_dueling: true
  use_prioritized_replay: true

training:
  n_episodes: 20000
  eval_freq: 500
  save_freq: 2000
  seed: 42
```

## 3. Discount Factor Experiment

The discount analysis experiment trains agents with different gamma values and compares their strategies:

```
uv run python -m experiments.discount_analysis --config configs/training.yaml
```

Defaults to gamma values: 0.1, 0.5, 0.9, 0.99, 0.999.

### Options

```
# Custom gamma values
uv run python -m experiments.discount_analysis --config configs/training.yaml \
    --gammas 0.5 0.9 0.99

# Override episode count (faster experiment)
uv run python -m experiments.discount_analysis --config configs/training.yaml \
    --n-episodes 3000

# Custom output directory
uv run python -m experiments.discount_analysis --config configs/training.yaml \
    --output-dir results/experiments/my_gamma_test

# All options combined
uv run python -m experiments.discount_analysis \
    --config configs/training.yaml \
    --gammas 0.1 0.5 0.99 \
    --n-episodes 5000 \
    --output-dir results/experiments/quick_test
```

Output goes to `results/experiments/<timestamp>/` containing:
- `gamma_<value>/model_gamma_<value>.pt` — trained model for each gamma
- `training_curves_comparison.png` — side-by-side training curves (rewards, scores, lengths, survival steps)
- `evaluation_comparison.png` — bar charts comparing mean score, survival steps, final length, survival rate, death causes
- `summary.yaml` — numerical summary of all evaluation metrics
- `experiment_config.yaml` — full experiment configuration

Each gamma value is evaluated over 100 greedy episodes after training.

### What the experiment measures

For each discount factor, the experiment collects:
- **Mean score** — total points from eating apples and golden fruits
- **Mean survival steps** — how long the agent stays alive
- **Mean final length** — snake length at episode end
- **Survival rate** — fraction of episodes where the agent survives to max steps
- **Death cause breakdown** — wall collision, body collision, obstacle/poison death, or shrink death

Low gamma (0.1, 0.5) agents tend to be short-sighted, taking immediate rewards without considering danger. High gamma (0.99, 0.999) agents develop more cautious strategies, avoiding obstacles and planning ahead.

## 4. Interactive Dashboard

Watch a trained agent play or play the game manually:

```
# Watch a trained agent
uv run python -m visualization.dashboard \
    --model results/runs/<timestamp>/model_final.pt \
    --config configs/training.yaml

# Play manually
uv run python -m visualization.dashboard --mode play

# Play manually with custom config
uv run python -m visualization.dashboard --mode play --config configs/training.yaml
```

Requires a display (not headless).

### Controls

| Key | Action |
|-----|--------|
| SPACE | Pause / unpause |
| +/- | Increase / decrease game speed |
| R | Reset the current episode |
| M | Toggle between observe and play modes |
| ESC | Quit |
| Arrow keys / WASD | Move snake (play mode only) |

The side panel shows current episode stats (score, steps, length, obstacles) and session stats (episodes played, best score, average score).

## 5. Suggested Training Plans

### Quick validation (~5 min)

Verify that training works and the agent improves:

```yaml
training:
  n_episodes: 1000
  eval_freq: 200
  save_freq: 500
  seed: 42
```

```
uv run python -m training.train_dqn --config configs/quick_test.yaml
```

Look for the average reward to trend upward over episodes.

### Standard training (~1-2 hours on CPU)

The default `configs/training.yaml` with 10,000 episodes. Should produce an agent that consistently avoids walls and collects apples.

### Extended training with all features

```yaml
agent:
  learning_rate: 0.0001
  discount_factor: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 100000
  buffer_size: 200000
  batch_size: 128
  target_update_freq: 2000
  use_double_dqn: true
  use_dueling: true
  use_prioritized_replay: true

training:
  n_episodes: 50000
  eval_freq: 1000
  save_freq: 5000
  seed: 42
```

Longer training with Dueling DQN and PER enabled. Best results but requires more compute.

### Quick discount experiment (~15 min)

Test just three gamma values with fewer episodes:

```
uv run python -m experiments.discount_analysis \
    --config configs/training.yaml \
    --gammas 0.5 0.9 0.99 \
    --n-episodes 2000
```

### Full discount experiment (~several hours)

All five gamma values with the default episode count from the config:

```
uv run python -m experiments.discount_analysis --config configs/training.yaml
```

## 6. GPU Acceleration

The DQN agent auto-detects CUDA. If a GPU is available, training runs on it automatically. No config changes needed.

To force CPU (e.g., for debugging):

```python
agent = DQNAgent(..., device="cpu")
```

There is no CLI flag for this — modify the config or code directly if needed.

## 7. Analyzing Results

### Training metrics

Load saved metrics for custom analysis:

```python
import numpy as np

data = np.load("results/runs/<timestamp>/metrics.npz")
rewards = data["rewards"]
scores = data["scores"]
losses = data["losses"]
```

### Experiment summary

The discount analysis experiment saves a YAML summary:

```python
import yaml

with open("results/experiments/<timestamp>/summary.yaml") as f:
    summary = yaml.safe_load(f)

for gamma, metrics in summary.items():
    print(f"gamma={gamma}: score={metrics['mean_score']:.1f}, "
          f"survival={metrics['survival_rate']:.1%}")
```
