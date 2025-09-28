"""Statistics display utilities for training visualization."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from collections import deque


class StatsTracker:
    """Track and display training statistics."""

    def __init__(self, window_size: int = 100):
        """Initialize statistics tracker.

        Args:
            window_size: Size of rolling window for averages
        """
        self.window_size = window_size
        self.episode_scores = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

        # Rolling windows for smooth plotting
        self.score_window = deque(maxlen=window_size)
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)

    def add_episode(self, score: int, total_reward: float, episode_length: int) -> None:
        """Add episode statistics.

        Args:
            score: Final score (food eaten)
            total_reward: Total reward accumulated
            episode_length: Number of steps in episode
        """
        self.episode_scores.append(score)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)

        self.score_window.append(score)
        self.reward_window.append(total_reward)
        self.length_window.append(episode_length)

    def add_loss(self, loss: float) -> None:
        """Add training loss.

        Args:
            loss: Training loss value
        """
        self.losses.append(loss)

    def get_current_stats(self) -> dict:
        """Get current statistics summary.

        Returns:
            Dictionary with current statistics
        """
        if not self.episode_scores:
            return {
                "episodes": 0,
                "avg_score": 0.0,
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "best_score": 0,
                "recent_avg_score": 0.0
            }

        return {
            "episodes": len(self.episode_scores),
            "avg_score": np.mean(self.episode_scores),
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "best_score": max(self.episode_scores),
            "recent_avg_score": np.mean(self.score_window) if self.score_window else 0.0
        }

    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress graphs.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.episode_scores:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Snake RL Training Progress")

        # Episode scores
        axes[0, 0].plot(self.episode_scores, alpha=0.7, label="Score")
        if len(self.episode_scores) >= self.window_size:
            # Plot rolling average
            rolling_avg = []
            for i in range(self.window_size - 1, len(self.episode_scores)):
                avg = np.mean(self.episode_scores[i - self.window_size + 1:i + 1])
                rolling_avg.append(avg)

            x_avg = range(self.window_size - 1, len(self.episode_scores))
            axes[0, 0].plot(x_avg, rolling_avg, color='red', label=f"Avg ({self.window_size} episodes)")

        axes[0, 0].set_title("Episode Scores")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode rewards
        axes[0, 1].plot(self.episode_rewards, alpha=0.7, color="green")
        axes[0, 1].set_title("Episode Rewards")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Total Reward")
        axes[0, 1].grid(True, alpha=0.3)

        # Episode lengths
        axes[1, 0].plot(self.episode_lengths, alpha=0.7, color="orange")
        axes[1, 0].set_title("Episode Lengths")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].grid(True, alpha=0.3)

        # Training loss
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.7, color="red")
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No loss data", ha="center", va="center",
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Training Loss")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Training progress plot saved: {save_path}")

        plt.show()

    def save_stats_log(self, filepath: str) -> None:
        """Save statistics to a log file.

        Args:
            filepath: Path to save the log file
        """
        import json
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        stats_data = {
            "episode_scores": self.episode_scores,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "summary": self.get_current_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)

        print(f"Statistics saved: {filepath}")

    def print_progress(self, episode: int, score: int, total_reward: float,
                      epsilon: float, steps: int) -> None:
        """Print current training progress.

        Args:
            episode: Current episode number
            score: Episode score
            total_reward: Episode total reward
            epsilon: Current epsilon value
            steps: Episode steps
        """
        stats = self.get_current_stats()

        print(f"Episode {episode:4d} | "
              f"Score: {score:2d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Steps: {steps:3d} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Avg Score: {stats['recent_avg_score']:.1f}")

        # Print milestone information
        if episode % 100 == 0 and episode > 0:
            print(f"\n--- Episode {episode} Summary ---")
            print(f"Best Score: {stats['best_score']}")
            print(f"Average Score (last {self.window_size}): {stats['recent_avg_score']:.2f}")
            print(f"Average Reward: {stats['avg_reward']:.2f}")
            print(f"Average Length: {stats['avg_length']:.2f}")
            print("-" * 40)