"""
Greedy rollout lookahead for inference-time action selection.

Instead of one-step greedy, each candidate action is tried on a copy of the
environment and followed by ``depth-1`` more greedy steps.  The action whose
rollout yields the highest discounted return is selected.

Depth is capped at ``min(snake_length, max_depth)`` so shorter snakes don't
waste time on long rollouts, while longer snakes with dangerous tails benefit
from deeper lookahead.

Multi-sample mode (n_samples > 1): each candidate action's rollout is repeated
up to ``n_samples`` times.  Because the environment uses Python's global random
state (object spawning, placement), successive deepcopies from the same base
state produce different stochastic outcomes.  The action with the highest peak
return across all samples wins.

Parallel execution: all n_samples × n_actions rollouts are dispatched at once
to a ThreadPoolExecutor.  PyTorch GPU operations (CUDA / MPS) release the GIL,
so inference calls run truly concurrently across threads.  On CPU, env steps
still benefit from overlapping with inference work in other threads.  Device
priority follows the agent's own device (CUDA → MPS → CPU).
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np


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


def lookahead_action(
    env: Any,
    agent: Any,
    obs: np.ndarray,
    snake_length: int,
    n_actions: int = 3,
    max_depth: int = 15,
    discount: float = 0.99,
    n_samples: int = 1,
) -> int:
    """Selects the action with the highest lookahead return.

    All n_samples × n_actions rollouts are run in parallel via
    ThreadPoolExecutor.  GPU inference (CUDA / MPS) releases Python's GIL so
    threads execute concurrently on the device; CPU falls back to
    GIL-interleaved execution which still overlaps env steps with inference.

    Args:
        env: current environment (may be wrapped; will be deepcopied per task).
        agent: agent with ``select_action(obs, training=False) -> int``.
        obs: current observation.
        snake_length: current snake length (used to compute rollout depth).
        n_actions: number of candidate actions.
        max_depth: hard cap on rollout depth.
        discount: per-step discount applied inside rollouts.
        n_samples: independent rollout samples per action (stochastic outcomes
            differ because Python's global RNG advances between deepcopies).
            The best peak return across all samples for each action is used.

    Returns:
        Best action index.
    """
    depth = max(1, min(snake_length, max_depth))
    n_tasks = n_samples * n_actions

    # Per-action best scores, protected by a lock for concurrent writes.
    best_scores = [-float("inf")] * n_actions
    lock = threading.Lock()

    def run_task(action: int) -> None:
        score = _single_action_score(env, agent, obs, action, depth, discount)
        with lock:
            if score > best_scores[action]:
                best_scores[action] = score

    # All tasks are independent — dispatch everything at once.
    with ThreadPoolExecutor(max_workers=n_tasks) as executor:
        futures = [
            executor.submit(run_task, action)
            for _ in range(n_samples)
            for action in range(n_actions)
        ]
        for f in as_completed(futures):
            f.result()  # re-raise any exception from a worker thread

    return int(np.argmax(best_scores))
