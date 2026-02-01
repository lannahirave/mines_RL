"""
DQN agent training script.

Usage:
    python -m training.train_dqn --config configs/training.yaml
"""

import argparse
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent


def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train(config: dict):
    """Main training loop."""

    # Set seed if provided
    seed = config["training"].get("seed")
    if seed is not None:
        set_seed(seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create environment
    env = SnakePlusEnv(
        grid_size=tuple(config["env"]["grid_size"]),
        spawn_probs=config["env"]["spawn_probs"],
        max_objects=config["env"]["max_objects"],
        obstacle_decay=config["env"].get("obstacle_decay"),
        max_steps=config["env"]["max_steps"],
        observation_type=config["env"]["observation_type"],
    )

    # Create agent
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

    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    losses = []

    # Training parameters
    n_episodes = config["training"]["n_episodes"]
    eval_freq = config["training"]["eval_freq"]
    save_freq = config["training"]["save_freq"]

    # Main loop
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, info = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0
        episode_length = 0

        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            metrics = agent.train_step()
            if metrics:
                losses.append(metrics["loss"])

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info["score"])

        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])

            print(f"\nEpisode {episode + 1}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            if losses:
                print(f"  Avg Loss (last 1000): {np.mean(losses[-1000:]):.4f}")

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_results = evaluate(agent, config, n_episodes=20)
            print(f"\n  [Eval] Avg Score: {eval_results['mean_score']:.2f}, "
                  f"Avg Length: {eval_results['mean_length']:.2f}, "
                  f"Avg Steps: {eval_results['mean_steps']:.2f}")

        # Save model checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(str(run_dir / f"model_ep{episode + 1}.pt"))

    # Save final model
    agent.save(str(run_dir / "model_final.pt"))

    # Save metrics
    np.savez(
        run_dir / "metrics.npz",
        rewards=episode_rewards,
        lengths=episode_lengths,
        scores=episode_scores,
        losses=losses,
    )

    # Plot training curves
    plot_training_curves(episode_rewards, episode_scores, losses, run_dir)

    env.close()
    print(f"\nTraining complete! Results saved to {run_dir}")

    return agent, env


def evaluate(agent: DQNAgent, config: dict, n_episodes: int = 100) -> dict:
    """Evaluates agent without exploration."""
    env = SnakePlusEnv(
        grid_size=tuple(config["env"]["grid_size"]),
        spawn_probs=config["env"]["spawn_probs"],
        max_objects=config["env"]["max_objects"],
        obstacle_decay=config["env"].get("obstacle_decay"),
        max_steps=config["env"]["max_steps"],
        observation_type=config["env"]["observation_type"],
    )

    scores = []
    lengths = []
    steps = []

    for _ in range(n_episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info["score"])
        lengths.append(info["length"])
        steps.append(info["steps"])

    env.close()

    return {
        "mean_score": np.mean(scores),
        "mean_length": np.mean(lengths),
        "mean_steps": np.mean(steps),
    }


def plot_training_curves(rewards, scores, losses, save_dir):
    """Plots training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3)
    if len(rewards) >= 100:
        axes[0, 0].plot(moving_average(rewards, 100), color="red")
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")

    # Scores
    axes[0, 1].plot(scores, alpha=0.3)
    if len(scores) >= 100:
        axes[0, 1].plot(moving_average(scores, 100), color="red")
    axes[0, 1].set_title("Episode Scores")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Score")

    # Losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.3)
        if len(losses) >= 1000:
            axes[1, 0].plot(moving_average(losses, 1000), color="red")
        axes[1, 0].set_title("Training Loss")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
    else:
        axes[1, 0].set_title("Training Loss (no data)")

    # Histogram of final rewards
    n_hist = min(1000, len(rewards))
    if n_hist > 0:
        axes[1, 1].hist(rewards[-n_hist:], bins=50)
        axes[1, 1].set_title(f"Reward Distribution (last {n_hist})")
        axes[1, 1].set_xlabel("Reward")
        axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()


def moving_average(data, window):
    """Computes moving average."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Snake+")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to training config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
