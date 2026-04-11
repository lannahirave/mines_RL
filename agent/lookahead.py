"""
Greedy rollout lookahead for inference-time action selection.

Instead of one-step greedy, each candidate action is tried on a copy of the
environment and followed by ``depth-1`` more greedy steps.  The action whose
rollout yields the highest risk-adjusted return is selected.

Depth is capped at ``min(snake_length, max_depth)`` so shorter snakes don't
waste time on long rollouts, while longer snakes with dangerous tails benefit
from deeper lookahead.

Multi-sample mode (n_samples > 1): each candidate action's rollout is repeated
up to ``n_samples`` times.  Because the environment uses Python's global random
state (object spawning, placement), successive deepcopies from the same base
state produce different stochastic outcomes.

Scoring: all sample scores are collected per action and aggregated as a
risk-adjusted mean:

    score(action) = mean(samples) - RISK_LAMBDA * death_rate * |DEATH_SCORE_THRESHOLD|

where death_rate = fraction of samples below DEATH_SCORE_THRESHOLD.  This
penalises actions that sometimes kill the snake even when their average return
is high, preferring consistent safe paths over high-variance gambles.

Parallel execution: all n_samples x n_actions rollouts are dispatched at once
to a ThreadPoolExecutor.  PyTorch GPU operations (CUDA / MPS) release the GIL,
so inference calls run truly concurrently across threads.  On CPU, env steps
still benefit from overlapping with inference work in other threads.  Device
priority follows the agent's own device (CUDA -> MPS -> CPU).
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple

import numpy as np


# Scores below this threshold indicate the rollout encountered a death.
# Death penalties range from -175 (long snake) to -500 (short snake).
# The worst non-death single step is rotten fruit at -20, so -50 is a
# clean separator between "death happened" and "normal bad step".
DEATH_SCORE_THRESHOLD = -50.0

# How hard to penalise death risk in the aggregation.
# With the default 2.0: an action that dies in every sample gets an extra
# -100 on top of its already-negative mean; an action with a 20% death rate
# gets -20 extra — enough to rank it below a safe path averaging +5.
RISK_LAMBDA = 2.0


def _rollout(env_copy: Any, agent: Any, obs: np.ndarray, depth: int, discount: float) -> float:
    """Rolls out ``depth`` greedy steps and returns the discounted return."""
    total = 0.0
    gamma = 1.0
    for _ in range(depth):
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, _ = env_copy.step(action)
        total += gamma * reward
        gamma *= discount
        if terminated or truncated:
            break
    return total


def _single_action_score(
    env: Any,
    agent: Any,
    obs: np.ndarray,
    action: int,
    depth: int,
    discount: float,
) -> float:
    """Returns the discounted return for one candidate action from the current state."""
    env_copy = copy.deepcopy(env)
    next_obs, reward, terminated, truncated, _ = env_copy.step(action)
    if terminated or truncated:
        return reward
    return reward + discount * _rollout(env_copy, agent, next_obs, depth - 1, discount)


def _risk_adjusted_score(sample_scores: List[float], risk_lambda: float = RISK_LAMBDA) -> float:
    """Aggregates multiple rollout scores into a single risk-adjusted value.

    Formula: mean(samples) - risk_lambda * death_rate * |DEATH_SCORE_THRESHOLD|

    A path that sometimes kills the snake is penalised proportionally to how
    often it does so, preferring consistent paths over lucky outliers.
    """
    if not sample_scores:
        return -float("inf")
    mean = sum(sample_scores) / len(sample_scores)
    death_rate = sum(1.0 for s in sample_scores if s < DEATH_SCORE_THRESHOLD) / len(sample_scores)
    return mean - risk_lambda * death_rate * abs(DEATH_SCORE_THRESHOLD)


def _compute_scores(
    env: Any,
    agent: Any,
    obs: np.ndarray,
    n_actions: int,
    depth: int,
    discount: float,
    n_samples: int,
    risk_lambda: float = RISK_LAMBDA,
) -> List[float]:
    """Runs all n_samples x n_actions rollouts in parallel.

    Returns per-action risk-adjusted scores.
    """
    n_tasks = n_samples * n_actions
    # Collect all sample scores per action before aggregating.
    all_samples: List[List[float]] = [[] for _ in range(n_actions)]
    lock = threading.Lock()

    def run_task(action: int) -> None:
        score = _single_action_score(env, agent, obs, action, depth, discount)
        with lock:
            all_samples[action].append(score)

    with ThreadPoolExecutor(max_workers=n_tasks) as executor:
        futures = [
            executor.submit(run_task, action)
            for _ in range(n_samples)
            for action in range(n_actions)
        ]
        for f in as_completed(futures):
            f.result()

    return [_risk_adjusted_score(all_samples[a], risk_lambda) for a in range(n_actions)]


def lookahead_action_with_scores(
    env: Any,
    agent: Any,
    obs: np.ndarray,
    snake_length: int,
    n_actions: int = 3,
    max_depth: int = 15,
    discount: float = 0.99,
    n_samples: int = 1,
    risk_lambda: float = RISK_LAMBDA,
) -> Tuple[int, List[float]]:
    """Like ``lookahead_action`` but also returns the per-action risk-adjusted scores.

    Returns:
        (best_action, scores) where scores[i] is the risk-adjusted aggregate
        return for action i across all samples.
    """
    depth = max(1, min(snake_length, max_depth))
    scores = _compute_scores(env, agent, obs, n_actions, depth, discount, n_samples, risk_lambda)
    return int(np.argmax(scores)), scores


def lookahead_action(
    env: Any,
    agent: Any,
    obs: np.ndarray,
    snake_length: int,
    n_actions: int = 3,
    max_depth: int = 15,
    discount: float = 0.99,
    n_samples: int = 1,
    risk_lambda: float = RISK_LAMBDA,
) -> int:
    """Selects the action with the highest risk-adjusted lookahead return.

    All n_samples x n_actions rollouts run in parallel (ThreadPoolExecutor).
    GPU inference (CUDA / MPS) releases the GIL so threads run concurrently.

    Returns:
        Best action index.
    """
    action, _ = lookahead_action_with_scores(
        env, agent, obs, snake_length, n_actions, max_depth, discount, n_samples, risk_lambda
    )
    return action
