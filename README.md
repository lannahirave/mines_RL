# Snake+ RL

A reinforcement learning project built around a custom Snake game with diverse game objects. The project explores how different RL algorithms and hyperparameters — particularly the discount factor (gamma) — affect learning strategies in a dynamic environment.

## About the Game

Snake+ extends the classic Snake game with multiple object types that create a richer decision-making landscape for RL agents. The snake moves on a 15x15 grid and can perform three actions: move forward, turn left, or turn right.

### Game Objects

| Object | Effect | Reward |
|--------|--------|--------|
| Apple | Grows the snake by 1 segment | +10 |
| Golden Fruit | Grows the snake by 3 segments | +30 to +70 (randomized) |
| Poison | Kills the snake instantly | -1000 |
| Sour Fruit | Shrinks the snake by 1-3 segments | -5 |
| Rotten Fruit | Detaches 3-5 tail segments, which become obstacles | -20 |

Obstacles created by rotten fruit decay after a configurable number of steps. Colliding with walls, the snake's own body, or obstacles results in death (-1000 penalty). A small per-step penalty of -0.1 encourages efficient play.

## RL Agents

The project implements two agent types:

**Q-Table Agent** — A tabular Q-learning agent that discretizes the 18-dimensional feature observation space. Best suited for quick experimentation and baseline comparisons.

**DQN Agent** — A Deep Q-Network agent with support for several advanced techniques:
- Double DQN to reduce Q-value overestimation
- Dueling network architecture to separately estimate state value and action advantages
- Prioritized Experience Replay (PER) to focus training on high-error transitions
- Target network with configurable update frequency
- Linear epsilon-greedy exploration schedule with configurable decay

Three neural network architectures are available: a fully-connected MLP (for feature observations), a CNN (for grid observations), and a Dueling variant of either.

## Observation Types

The environment provides two observation formats:

**Features** — An 18-dimensional vector containing danger signals in three directions, the snake's current direction (one-hot), food direction indicators, nearest object type (one-hot), distance to the nearest food, and normalized snake length. Used with Q-Table and MLP-based DQN agents.

**Grid** — An 8-channel 15x15 tensor with separate binary channels for the snake head, body, and each object type (apple, golden, poison, sour, rotten, obstacle). Used with CNN-based DQN agents.

## Project Structure

- `env/` — Gymnasium-compliant game environment, snake logic, game object definitions and reward calculator, Pygame renderer
- `agent/` — Q-Table and DQN agent implementations, neural network architectures, standard and prioritized replay buffers
- `training/` — Main training script with CLI interface, evaluation loop, metric logging, and training curve generation
- `configs/` — YAML configuration files for environment and training hyperparameters
- `tests/` — Unit tests for the environment, agents, and game logic
- `experiments/` — Experimental analysis (planned)
- `visualization/` — Visualization dashboard (planned)

## Configuration

All training and environment parameters are controlled via YAML config files in `configs/`. The main config file is `configs/training.yaml`, which defines:

- **Environment settings**: grid size, object spawn probabilities, max objects on the field, obstacle decay rate, max steps per episode, and observation type
- **Agent hyperparameters**: learning rate, discount factor, epsilon schedule, replay buffer size, batch size, target network update frequency, and toggles for Double DQN / Dueling / PER
- **Training settings**: number of episodes, evaluation frequency, checkpoint save frequency, and random seed

To customize training, copy `configs/training.yaml`, edit the values, and pass your config path to the training command.

## Setup

Requires Python 3.10 or higher.

Install dependencies with uv:

    uv pip install -r requirements.txt

Or set up a virtual environment first:

    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt

## Training

Run the training script with the default configuration:

    uv run python -m training.train_dqn --config configs/training.yaml

To use a custom config:

    uv run python -m training.train_dqn --config configs/my_custom_config.yaml

Training outputs are saved to `results/runs/<timestamp>/` and include:
- Model checkpoints saved at regular intervals
- A final model file (`model_final.pt`)
- Training metrics (`metrics.npz`)
- A copy of the config used (`config.yaml`)
- Training curve plots (`training_curves.png`) showing rewards, scores, losses, and reward distributions

The training script logs progress every 100 episodes and runs greedy evaluation episodes every 500 episodes (configurable). Checkpoints are saved every 1000 episodes by default.

## Running a Trained Model

After training, load a saved model and run it in the environment. Pass `render_mode="human"` to the environment constructor to visualize the game with Pygame (requires a display).

To run evaluation without rendering, instantiate the environment without a render mode, load the agent from a checkpoint using its `load` method, and run episodes with `training=False` in `select_action` to use greedy action selection (no exploration).

Models are saved as `.pt` files (DQN) or `.pkl` files (Q-Table) and can be loaded back with each agent's `load` class method.

## Running Tests

    uv run python -m pytest tests/ -v

## Implementation Status

- **Phase 1**: Game foundation — game objects, snake logic, unit tests
- **Phase 2**: Gymnasium environment integration, Pygame renderer
- **Phase 3**: Q-Table agent, DQN agent with Double DQN / Dueling / PER support, replay buffers, neural network architectures
- **Phase 4**: Training CLI script, YAML-based configuration, evaluation loop, metric logging, training curve plots
- **Phase 5**: Experimental analysis of discount factor impact (planned)
- **Phase 6**: Visualization tools and dashboard (planned)
