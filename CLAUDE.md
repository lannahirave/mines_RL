# CLAUDE.md

## Project Overview

Snake+ RL — a reinforcement learning project where agents learn to play a custom Snake game with diverse game objects (apples, poison, golden fruit, sour fruit, rotten fruit). Built on Gymnasium and PyTorch.

## Common Commands

Install dependencies:
    uv pip install -r requirements.txt

Train a DQN agent:
    uv run python -m training.train_dqn --config configs/training.yaml

Run all tests:
    uv run python -m pytest tests/ -v

Run a single test file:
    uv run python -m pytest tests/test_env.py -v

## Architecture

- `env/` — Gymnasium environment (`SnakePlusEnv`), snake physics, game objects with factory pattern, Pygame renderer
- `agent/` — `QTableAgent` (tabular Q-learning) and `DQNAgent` (Deep Q-Network with Double DQN, Dueling, PER options), neural networks (`DQN_MLP`, `DQN_CNN`, `DuelingDQN`), replay buffers
- `training/` — Single entry point `train_dqn.py` with argparse CLI, loads YAML config, runs training loop with periodic evaluation and checkpointing
- `configs/` — YAML files with three sections: `env`, `agent`, `training`
- `tests/` — pytest-based, one test class per component, uses `setup_method()` for initialization

## Code Conventions

- Python 3.10+, 4-space indentation, no formatter/linter configured
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for enum constants
- Type hints on all function signatures and return types
- Docstrings with Args/Returns sections on public classes and methods
- Relative imports within packages (`from .networks import DQN_MLP`), absolute imports across packages (`from env.snake_env import SnakePlusEnv`)
- Enums for Direction and Action, dataclasses for GameObject
- Factory pattern for object spawning (`ObjectFactory`)

## Key Patterns

- Gymnasium 5-tuple API: `step()` returns `(obs, reward, terminated, truncated, info)`
- `terminated` = game-ending event (death), `truncated` = max steps reached
- Agent interface: `select_action(obs, training=bool)`, `store_transition(...)`, `train_step()`
- `training=True` uses epsilon-greedy exploration, `training=False` uses greedy action selection
- DQN agent auto-selects CUDA if available, falls back to CPU
- Two observation types: `"features"` (18-dim vector for MLP/Q-table) and `"grid"` (8-channel tensor for CNN)
- Training outputs go to `results/runs/<timestamp>/` with model checkpoints, metrics, config copy, and plots

## Configuration

All hyperparameters live in `configs/training.yaml`. Three sections:
- `env`: grid_size, spawn_probs, max_objects, obstacle_decay, max_steps, observation_type
- `agent`: learning_rate, discount_factor, epsilon schedule, buffer_size, batch_size, target_update_freq, toggles for double_dqn/dueling/prioritized_replay
- `training`: n_episodes, eval_freq, save_freq, seed

## Rewards

- Apple: +10, Golden: +30-70, Sour: -5, Rotten: -20, Poison: -1000
- Wall/body/obstacle collision: -1000
- Step penalty: -0.1, Survival bonus: +0.01 per step

## Testing

Tests use pytest with class-based organization. Each test class has `setup_method()` for per-test initialization. Test files mirror source structure: `test_env.py`, `test_agent.py`, `test_game_logic.py`.

## Tool Rules

- Always use `uv` for all Python operations: `uv run`, `uv pip install`, etc. Never use bare `pip` or `python`.
- Never install dependencies without asking the user first. Always confirm before running any `uv pip install` command.
