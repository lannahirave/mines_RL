"""
Discount factor (gamma) analysis experiment.

Trains DQN agents with different discount factors and compares their strategies.

Usage:
    python -m experiments.discount_analysis --config configs/training.yaml
    python -m experiments.discount_analysis --config configs/training.yaml --gammas 0.1 0.5 0.99
    python -m experiments.discount_analysis --config configs/training.yaml --n-episodes 5000
"""

import argparse
import copy
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from env.game_objects import ObjectType
from agent.dqn_agent import DQNAgent


DEFAULT_GAMMAS = [0.1, 0.5, 0.9, 0.99, 0.999]


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_env(config: dict) -> SnakePlusEnv:
    """Creates environment from config."""
    return SnakePlusEnv(
        grid_size=tuple(config["env"]["grid_size"]),
        spawn_probs=config["env"]["spawn_probs"],
        max_objects=config["env"]["max_objects"],
        obstacle_decay=config["env"].get("obstacle_decay"),
        max_steps=config["env"]["max_steps"],
        observation_type=config["env"]["observation_type"],
    )


def train_with_gamma(
    config: dict,
    gamma: float,
    run_dir: Path,
    n_episodes: int,
) -> Dict[str, Any]:
    """
    Trains a DQN agent with a specific discount factor.

    Args:
        config: base training config
        gamma: discount factor to use
        run_dir: directory to save results
        n_episodes: number of training episodes

    Returns:
        Dictionary with training metrics and evaluation results
    """
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    env = create_env(config)

    agent = DQNAgent(
        observation_type=config["env"]["observation_type"],
        n_actions=3,
        learning_rate=config["agent"]["learning_rate"],
        discount_factor=gamma,
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

    # Training metrics
    episode_rewards: List[float] = []
    episode_scores: List[int] = []
    episode_lengths: List[int] = []
    episode_steps: List[int] = []
    losses: List[float] = []

    for episode in tqdm(range(n_episodes), desc=f"gamma={gamma}"):
        state, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            metrics = agent.train_step()
            if metrics:
                losses.append(metrics["loss"])

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_scores.append(info["score"])
        episode_lengths.append(info["length"])
        episode_steps.append(info["steps"])

    # Save model
    agent.save(str(run_dir / f"model_gamma_{gamma}.pt"))

    env.close()

    # Detailed evaluation
    eval_results = evaluate_agent(agent, config, n_episodes=100)

    return {
        "gamma": gamma,
        "episode_rewards": episode_rewards,
        "episode_scores": episode_scores,
        "episode_lengths": episode_lengths,
        "episode_steps": episode_steps,
        "losses": losses,
        "eval": eval_results,
    }


def evaluate_agent(
    agent: DQNAgent,
    config: dict,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates agent and collects detailed metrics.

    Args:
        agent: trained DQN agent
        config: environment config
        n_episodes: number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics
    """
    env = create_env(config)

    scores: List[int] = []
    survival_steps: List[int] = []
    final_lengths: List[int] = []
    death_causes: List[str] = []

    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        cause = "truncated"

        while not done:
            action = agent.select_action(state, training=False)
            prev_obstacles = len(env.obstacles)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                # Determine death cause
                if reward <= -999:
                    head = env.snake.head
                    if not env._is_valid_position(head):
                        cause = "wall"
                    elif env.snake.check_self_collision():
                        cause = "body"
                    else:
                        cause = "obstacle_or_poison"
                else:
                    cause = "shrink"

        scores.append(info["score"])
        survival_steps.append(info["steps"])
        final_lengths.append(info["length"])
        death_causes.append(cause)

    env.close()

    survival_rate = sum(1 for c in death_causes if c == "truncated") / n_episodes
    obstacle_death_rate = sum(
        1 for c in death_causes if c == "obstacle_or_poison"
    ) / n_episodes

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_survival_steps": float(np.mean(survival_steps)),
        "std_survival_steps": float(np.std(survival_steps)),
        "mean_final_length": float(np.mean(final_lengths)),
        "survival_rate": survival_rate,
        "obstacle_death_rate": obstacle_death_rate,
        "death_causes": death_causes,
    }


def plot_comparison(
    results: List[Dict[str, Any]],
    save_dir: Path,
) -> None:
    """
    Generates comparison plots for all gamma values.

    Args:
        results: list of result dicts from train_with_gamma
        save_dir: directory to save plots
    """
    sns.set_theme(style="whitegrid")
    gammas = [r["gamma"] for r in results]

    # --- Figure 1: Training curves ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves by Discount Factor", fontsize=14)

    for r in results:
        label = f"γ={r['gamma']}"
        window = 100

        # Rewards
        rewards = r["episode_rewards"]
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0, 0].plot(smoothed, label=label)
        axes[0, 0].set_title("Episode Rewards (smoothed)")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        # Scores
        scores = r["episode_scores"]
        if len(scores) >= window:
            smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
            axes[0, 1].plot(smoothed, label=label)
        axes[0, 1].set_title("Episode Scores (smoothed)")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Score")

        # Lengths
        lengths = r["episode_lengths"]
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
            axes[1, 0].plot(smoothed, label=label)
        axes[1, 0].set_title("Snake Length (smoothed)")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Length")

        # Steps survived
        steps = r["episode_steps"]
        if len(steps) >= window:
            smoothed = np.convolve(steps, np.ones(window) / window, mode="valid")
            axes[1, 1].plot(smoothed, label=label)
        axes[1, 1].set_title("Survival Steps (smoothed)")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps")

    for ax in axes.flat:
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves_comparison.png", dpi=150)
    plt.close()

    # --- Figure 2: Evaluation bar charts ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Evaluation Metrics by Discount Factor", fontsize=14)

    gamma_labels = [str(g) for g in gammas]
    eval_data = [r["eval"] for r in results]

    # Mean score
    vals = [e["mean_score"] for e in eval_data]
    errs = [e["std_score"] for e in eval_data]
    axes[0, 0].bar(gamma_labels, vals, yerr=errs, capsize=5, color=sns.color_palette())
    axes[0, 0].set_title("Mean Score")
    axes[0, 0].set_xlabel("γ")

    # Mean survival steps
    vals = [e["mean_survival_steps"] for e in eval_data]
    errs = [e["std_survival_steps"] for e in eval_data]
    axes[0, 1].bar(gamma_labels, vals, yerr=errs, capsize=5, color=sns.color_palette())
    axes[0, 1].set_title("Mean Survival Steps")
    axes[0, 1].set_xlabel("γ")

    # Mean final length
    vals = [e["mean_final_length"] for e in eval_data]
    axes[0, 2].bar(gamma_labels, vals, color=sns.color_palette())
    axes[0, 2].set_title("Mean Final Length")
    axes[0, 2].set_xlabel("γ")

    # Survival rate
    vals = [e["survival_rate"] for e in eval_data]
    axes[1, 0].bar(gamma_labels, vals, color=sns.color_palette())
    axes[1, 0].set_title("Survival Rate")
    axes[1, 0].set_xlabel("γ")
    axes[1, 0].set_ylim(0, 1)

    # Obstacle/poison death rate
    vals = [e["obstacle_death_rate"] for e in eval_data]
    axes[1, 1].bar(gamma_labels, vals, color=sns.color_palette())
    axes[1, 1].set_title("Obstacle/Poison Death Rate")
    axes[1, 1].set_xlabel("γ")
    axes[1, 1].set_ylim(0, 1)

    # Death cause breakdown (stacked bar)
    cause_types = ["wall", "body", "obstacle_or_poison", "shrink", "truncated"]
    cause_labels = ["Wall", "Body", "Obstacle/Poison", "Shrink", "Survived"]
    bottom = np.zeros(len(gammas))
    for cause, label in zip(cause_types, cause_labels):
        vals = [
            sum(1 for c in e["death_causes"] if c == cause) / len(e["death_causes"])
            for e in eval_data
        ]
        axes[1, 2].bar(gamma_labels, vals, bottom=bottom, label=label)
        bottom += np.array(vals)
    axes[1, 2].set_title("Death Cause Breakdown")
    axes[1, 2].set_xlabel("γ")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / "evaluation_comparison.png", dpi=150)
    plt.close()


def run_experiment(
    config: dict,
    gammas: List[float],
    n_episodes: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Runs the full discount factor experiment.

    Args:
        config: base training config
        gammas: list of discount factors to test
        n_episodes: training episodes per gamma
        output_dir: directory to save all results

    Returns:
        List of result dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    exp_config = {
        "base_config": config,
        "gammas": gammas,
        "n_episodes": n_episodes,
    }
    with open(output_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(exp_config, f)

    results = []
    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"Training with gamma = {gamma}")
        print(f"{'='*60}")

        gamma_dir = output_dir / f"gamma_{gamma}"
        gamma_dir.mkdir(parents=True, exist_ok=True)

        result = train_with_gamma(config, gamma, gamma_dir, n_episodes)
        results.append(result)

        # Print evaluation summary
        ev = result["eval"]
        print(f"\n  Eval Results (gamma={gamma}):")
        print(f"    Mean Score:          {ev['mean_score']:.2f} +/- {ev['std_score']:.2f}")
        print(f"    Mean Survival Steps: {ev['mean_survival_steps']:.1f}")
        print(f"    Mean Final Length:   {ev['mean_final_length']:.1f}")
        print(f"    Survival Rate:       {ev['survival_rate']:.2%}")
        print(f"    Obstacle Death Rate: {ev['obstacle_death_rate']:.2%}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, output_dir)

    # Save summary
    summary = {
        str(r["gamma"]): {
            "mean_score": r["eval"]["mean_score"],
            "std_score": r["eval"]["std_score"],
            "mean_survival_steps": r["eval"]["mean_survival_steps"],
            "mean_final_length": r["eval"]["mean_final_length"],
            "survival_rate": r["eval"]["survival_rate"],
            "obstacle_death_rate": r["eval"]["obstacle_death_rate"],
        }
        for r in results
    }
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nExperiment complete! Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Discount factor analysis for Snake+ DQN agent"
    )
    parser.add_argument(
        "--config", type=str, default="configs/training.yaml",
        help="Path to base training config YAML"
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS,
        help="Discount factors to test (default: 0.1 0.5 0.9 0.99 0.999)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Training episodes per gamma (default: from config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/experiments/<timestamp>)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_episodes = args.n_episodes or config["training"]["n_episodes"]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/experiments/{timestamp}")

    run_experiment(config, args.gammas, n_episodes, output_dir)


if __name__ == "__main__":
    main()
