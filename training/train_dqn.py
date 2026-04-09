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

from env.snake_env import make_snake_env
from env.wrappers import FrameStack
from agent.dqn_agent import DQNAgent


def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device():
    """Configures GPU backend for maximum performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  cuDNN benchmark: enabled")
        print(f"  TF32: enabled")
        print(f"  CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        print(f"Device: MPS (Apple Silicon)")
    else:
        print("No GPU available, running on CPU")


def train(config: dict):
    """Main training loop."""

    # Setup GPU optimizations
    setup_device()

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

    # Create environment (with optional frame stacking)
    env = make_snake_env(config["env"])
    n_frames = config["agent"].get("n_frames", 1)
    if n_frames > 1:
        env = FrameStack(env, n_frames=n_frames)

    # GPU config with defaults
    gpu_config = config.get("gpu", {})

    # Create agent with GPU optimizations
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
        device=gpu_config.get("device", "auto"),
        use_amp=gpu_config.get("use_amp", True),
        use_compile=gpu_config.get("use_compile", False),
        pin_memory=gpu_config.get("pin_memory", True),
        train_steps_per_update=gpu_config.get("train_steps_per_update", 1),
        lr_cosine_steps=config["agent"].get("lr_cosine_steps", 500000),
        lr_min=config["agent"].get("lr_min", 1e-6),
        warmup_steps=config["agent"].get("warmup_steps", 0),
        tau=config["agent"].get("tau", 1.0),
        sign_log_reward=config["agent"].get("sign_log_reward", False),
        n_step=config["agent"].get("n_step", 1),
        n_frames=n_frames,
        grid_size=tuple(config["env"]["grid_size"]),
        network_type=config["agent"].get("network_type", "grid"),
        per_beta_frames=config["agent"].get("per_beta_frames", 100000),
    )

    print(f"Device: {agent.device}")
    print(f"AMP: {'enabled' if agent.use_amp else 'disabled'}")
    print(f"Batch size: {config['agent']['batch_size']}")
    print(f"Buffer size: {config['agent']['buffer_size']}")
    print(f"Train steps per update: {gpu_config.get('train_steps_per_update', 1)}")
    print(f"LR schedule: cosine annealing over {config['agent'].get('lr_cosine_steps', 500000)} steps "
          f"to {config['agent'].get('lr_min', 1e-6)}")
    print(f"Warmup steps: {agent.warmup_steps}")
    print(f"Tau: {agent.tau}")
    print(f"Sign-log reward: {agent.sign_log_reward}")
    print(f"N-step: {agent.n_step}")
    print(f"Frame stack: {n_frames}")

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

            # Store transition (use terminated, not done, so truncated episodes bootstrap)
            agent.store_transition(state, action, reward, next_state, terminated)

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
            print(f"  LR: {agent.optimizer.param_groups[0]['lr']:.2e}")
            if losses:
                print(f"  Avg Loss (last 1000): {np.mean(losses[-1000:]):.4f}")
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / (1024 ** 2)
                gpu_mem_cached = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"  GPU Memory: {gpu_mem_used:.0f} MB allocated, {gpu_mem_cached:.0f} MB reserved")
            elif torch.backends.mps.is_available():
                gpu_mem_used = torch.mps.current_allocated_memory() / (1024 ** 2)
                gpu_mem_driver = torch.mps.driver_allocated_memory() / (1024 ** 2)
                print(f"  MPS Memory: {gpu_mem_used:.0f} MB allocated, {gpu_mem_driver:.0f} MB driver")

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_results = evaluate(agent, config, n_episodes=20)
            print(f"\n  [Eval] Avg Score: {eval_results['mean_score']:.2f}, "
                  f"Avg Length: {eval_results['mean_length']:.2f}, "
                  f"Avg Steps: {eval_results['mean_steps']:.2f}")

        # Save model checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(str(run_dir / f"model_ep{episode + 1}.pt"))
            plot_training_curves(episode_rewards, episode_scores, losses, run_dir)

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
    env = make_snake_env(config["env"])
    n_frames = config["agent"].get("n_frames", 1)
    if n_frames > 1:
        env = FrameStack(env, n_frames=n_frames)

    scores = []
    lengths = []
    steps = []

    eval_seed = config["training"].get("seed", 0)
    for ep in range(n_episodes):
        state, info = env.reset(seed=eval_seed + ep)
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
