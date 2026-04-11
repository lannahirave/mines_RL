# CLAUDE.md

## Project Overview

Snake+ RL — a reinforcement learning project where agents learn to play a custom Snake game with diverse game objects (apples, poison, golden fruit, sour fruit, rotten fruit). Built on Gymnasium and PyTorch.

## Common Commands

Install dependencies:
    uv pip install -r requirements.txt

Train a DQN agent:
    uv run python -m training.train_dqn --config configs/training.yaml

Evaluate a checkpoint (greedy):
    uv run python -m training.eval --config configs/training.yaml

Evaluate with lookahead (depth 15, 5 samples):
    uv run python -m training.eval --config configs/training.yaml --lookahead 15 --episodes 20

Watch agent play (dashboard):
    uv run python -m visualization.dashboard --model results/runs/<timestamp>/model_ep18000.pt --config results/runs/<timestamp>/config.yaml

Watch with lookahead:
    uv run python -m visualization.dashboard --model results/runs/<timestamp>/model_ep18000.pt --config results/runs/<timestamp>/config.yaml --lookahead 15 --samples 5

Run all tests:
    uv run python -m pytest tests/ -v

Run a single test file:
    uv run python -m pytest tests/test_env.py -v

## Architecture

- `env/` — Gymnasium environment (`SnakePlusEnv`), snake physics, game objects with factory pattern, Pygame renderer
- `agent/` — `QTableAgent` (tabular Q-learning) and `DQNAgent` (Deep Q-Network with Double DQN, Dueling, PER options), neural networks (`DQN_MLP`, `DQN_CNN`), replay buffers, `lookahead.py` (inference-time rollout)
- `training/` — `train_dqn.py` (training loop), `eval.py` (standalone checkpoint evaluation)
- `visualization/` — `dashboard.py` (interactive Pygame dashboard: observe agent or play manually)
- `configs/` — YAML files with three sections: `env`, `agent`, `training`
- `tests/` — pytest-based, one test class per component, uses `setup_method()` for initialization
- `algo.md` — algorithm context document for the lookahead path-selection logic

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
- Two observation types: `"features"` (29-dim vector for MLP/Q-table) and `"grid"` (6-channel tensor for CNN: snake body decay, good-object proximity field, bad-object proximity field, danger proximity field, direction dx/dy)
- Training outputs go to `results/runs/<timestamp>/` with model checkpoints, metrics, config copy, and plots

## Configuration

All hyperparameters live in `configs/training.yaml`. Three sections:

- `env`: grid_size, spawn_probs, max_objects, obstacle_decay, max_steps, observation_type, starvation_max_steps, proximity_good_scale, proximity_bad_scale, fruit/death length-scaling coefs (`make_snake_env` builds `SnakePlusEnv` from YAML)
- `agent`: learning_rate, discount_factor, epsilon schedule, buffer_size, batch_size, target_update_freq, toggles for double_dqn/dueling/prioritized_replay
- `training`: n_episodes, eval_freq, save_freq, seed, `weights` (checkpoint path for eval), `lookahead_depth` (0 = disabled; >0 enables rollout lookahead during periodic eval)

## Rewards

- Apple: +10, Golden: +30-70, Sour: -5, Rotten: -20, Poison: -500
- Wall/body/obstacle collision: -500 (scaled down by length; minimum 35% of base)
- Step penalty: -0.1, Survival bonus: +0.0 per step

## Lookahead (inference-time rollout)

`agent/lookahead.py` provides path-selection on top of the trained policy at eval/dashboard time — never during training.

- For each of 3 actions, simulates `min(snake_length, max_depth)` steps on a `deepcopy` of the env using the greedy policy
- Runs `n_samples` independent rollouts per action in parallel (`ThreadPoolExecutor`); each sample gets different stochastic spawn outcomes because Python's global `random` state advances between deepcopies
- Scores are aggregated as `mean − RISK_LAMBDA × death_rate × 50` (risk-adjusted mean: penalises paths that sometimes kill the snake)
- `lookahead_action_with_scores()` returns `(action, scores_list)` so callers can compare against the agent's own prediction and pick the higher-scored one
- GPU (CUDA/MPS) inference releases the GIL so threads run concurrently; falls back to CPU automatically
- See `algo.md` for full algorithm context and improvement directions

## Dashboard (`visualization/dashboard.py`)

Interactive Pygame window. Key bindings: SPACE pause, +/− speed, R reset, M toggle observe/play, 1–6 channel overlay, ESC quit.

- **Danger preds** panel: counts per-step predictions where the agent's raw greedy choice scored below the death threshold (−50), broken down by action type. Requires `--lookahead` to be active.
- `--lookahead N` enables rollout lookahead of depth N; `--samples K` sets rollout samples per action (default 5)

## Testing

Tests use pytest with class-based organization. Each test class has `setup_method()` for per-test initialization. Test files mirror source structure: `test_env.py`, `test_agent.py`, `test_game_logic.py`.

## Tool Rules

- Always use `uv` for all Python operations: `uv run`, `uv pip install`, etc. Never use bare `pip` or `python`.
- Never install dependencies without asking the user first. Always confirm before running any `uv pip install` command.
