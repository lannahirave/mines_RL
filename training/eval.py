"""
Standalone evaluation script.

Loads a trained DQN checkpoint and runs greedy evaluation episodes,
optionally with lookahead rollouts and/or Pygame rendering.

Usage:
    uv run python -m training.eval --config configs/training.yaml
    uv run python -m training.eval --config configs/training.yaml --weights results/runs/.../model_final.pt
    uv run python -m training.eval --config configs/training.yaml --lookahead 15
    uv run python -m training.eval --config configs/training.yaml --render --episodes 5
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import make_snake_env
from env.wrappers import FrameStack
from agent.dqn_agent import DQNAgent
from agent.lookahead import lookahead_action


def run_eval(config: dict, weights: str, n_episodes: int, lookahead_depth: int, render: bool):
    env_cfg = dict(config["env"])
    if render:
        env_cfg["render_mode"] = "human"

    env = make_snake_env(env_cfg)
    n_frames = config["agent"].get("n_frames", 1)
    if n_frames > 1:
        env = FrameStack(env, n_frames=n_frames)

    agent = DQNAgent(
        observation_type=config["env"]["observation_type"],
        n_actions=3,
        learning_rate=config["agent"]["learning_rate"],
        discount_factor=config["agent"]["discount_factor"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_end=config["agent"]["epsilon_end"],
        epsilon_decay_steps=config["agent"]["epsilon_decay_steps"],
        buffer_size=1000,  # minimal; not used during eval
        batch_size=config["agent"]["batch_size"],
        target_update_freq=config["agent"]["target_update_freq"],
        use_double_dqn=config["agent"]["use_double_dqn"],
        use_dueling=config["agent"]["use_dueling"],
        use_prioritized_replay=config["agent"]["use_prioritized_replay"],
        tau=config["agent"].get("tau", 1.0),
        sign_log_reward=config["agent"].get("sign_log_reward", False),
        n_step=config["agent"].get("n_step", 1),
        n_frames=n_frames,
        grid_size=tuple(config["env"]["grid_size"]),
        network_type=config["agent"].get("network_type", "grid"),
        per_beta_frames=config["agent"].get("per_beta_frames", 100000),
    )

    agent.load(weights)
    agent.epsilon = 0.0  # force greedy
    print(f"Loaded weights: {weights}")
    if lookahead_depth > 0:
        print(f"Lookahead depth: {lookahead_depth}")
    print(f"Episodes: {n_episodes}\n")

    scores, lengths, steps_list = [], [], []
    seed = config["training"].get("seed", 0)

    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            if lookahead_depth > 0:
                action = lookahead_action(
                    env, agent, state,
                    snake_length=info["length"],
                    max_depth=lookahead_depth,
                    discount=config["agent"]["discount_factor"],
                )
            else:
                action = agent.select_action(state, training=False)

            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if render:
                env.render()
                time.sleep(0.05)

        scores.append(info["score"])
        lengths.append(info["length"])
        steps_list.append(info["steps"])

        print(f"  ep {ep + 1:>3d}  score={info['score']:>6.1f}  length={info['length']:>3d}  steps={info['steps']:>5d}")

    print(f"\n{'-' * 45}")
    print(f"  Mean score:  {np.mean(scores):.2f}  ± {np.std(scores):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f}  ± {np.std(lengths):.2f}")
    print(f"  Mean steps:  {np.mean(steps_list):.2f}  ± {np.std(steps_list):.2f}")
    print(f"  Max score:   {np.max(scores):.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to .pt checkpoint (overrides config training.weights)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=None,
                        help="Lookahead rollout depth (overrides config training.lookahead_depth)")
    parser.add_argument("--render", action="store_true", help="Render with Pygame")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    weights = args.weights or config["training"].get("weights")
    if not weights:
        parser.error("No weights specified. Use --weights or set training.weights in config.")

    lookahead_depth = args.lookahead if args.lookahead is not None else config["training"].get("lookahead_depth", 0)

    run_eval(config, weights, n_episodes=args.episodes, lookahead_depth=lookahead_depth, render=args.render)
