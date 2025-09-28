import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from src.agent.base_agent import BaseAgent
from src.environment.snake_env import SnakeEnv
from src.visualization.pygame_renderer import PyGameRenderer
from src.visualization.stats_display import StatsTracker
from src.utils.config import get_training_config, get_visualization_config


class UniversalTrainer:
    """
    Universal trainer that can train any agent type with the Snake environment.
    Provides real-time visualization, statistics tracking, and model checkpointing.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: SnakeEnv,
        agent_name: str,
        config: Optional[Dict[str, Any]] = None,
        enable_rendering: bool = True,
        save_dir: str = "models"
    ):
        """
        Initialize the universal trainer.

        Args:
            agent: The RL agent to train
            env: Snake environment
            agent_name: Name of the agent (for saving models)
            config: Training configuration
            enable_rendering: Whether to enable PyGame rendering
            save_dir: Directory to save models
        """
        self.agent = agent
        self.env = env
        self.agent_name = agent_name
        self.enable_rendering = enable_rendering

        # Load configurations
        self.training_config = config or get_training_config()
        self.viz_config = get_visualization_config()

        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize renderer and stats tracker
        if self.enable_rendering:
            self.renderer = PyGameRenderer(
                grid_size=env.grid_size,
                window_size=(800, 600)
            )
        else:
            self.renderer = None

        self.stats_tracker = StatsTracker()

        # Training state
        self.episode = 0
        self.best_score = -float('inf')
        self.episodes_without_improvement = 0
        self.training_start_time = None

        # Performance tracking
        self.episode_scores = []
        self.episode_lengths = []
        self.training_metrics = []

    def train(self, num_episodes: int, evaluate_freq: int = 50) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
            evaluate_freq: Frequency of evaluation (episodes)

        Returns:
            Training summary statistics
        """
        print(f"Starting training for {self.agent_name} agent...")
        print(f"Target episodes: {num_episodes}")
        print(f"Evaluation frequency: {evaluate_freq}")

        self.training_start_time = time.time()
        self.agent.set_training_mode(True)

        try:
            for episode in range(num_episodes):
                self.episode = episode

                # Run training episode
                episode_stats = self._run_episode(training=True)

                # Check if user requested quit
                if episode_stats.get('quit_requested', False):
                    print("Training stopped by user request")
                    break

                # Update statistics
                self._update_statistics(episode_stats)

                # Log progress
                if episode % self.training_config['log_frequency'] == 0:
                    self._log_progress(episode, episode_stats)

                # Evaluate agent periodically
                if episode % evaluate_freq == 0 and episode > 0:
                    eval_stats = self.evaluate(num_episodes=5)
                    print(f"Episode {episode} - Evaluation: {eval_stats}")

                # Save model checkpoints
                if episode % self.training_config['save_frequency'] == 0 and episode > 0:
                    self._save_checkpoint(episode)

                # Check for early stopping
                if self._check_early_stopping():
                    print(f"Early stopping at episode {episode}")
                    break

                # Update visualization
                if self.renderer:
                    self.renderer.handle_events()

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at episode {self.episode}")

        finally:
            # Save final model
            self._save_final_model()

            # Close renderer
            if self.renderer:
                self.renderer.close()

        # Return training summary
        return self._get_training_summary()

    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent's performance.

        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation

        Returns:
            Evaluation statistics
        """
        self.agent.set_training_mode(False)

        scores = []
        lengths = []

        for episode in range(num_episodes):
            print(f"Testing episode {episode + 1}/{num_episodes}")
            episode_stats = self._run_episode(training=False, render=render)

            # Check if user requested quit
            if episode_stats.get('quit_requested', False):
                print("Evaluation stopped by user request")
                break

            scores.append(episode_stats['score'])
            lengths.append(episode_stats['length'])

        self.agent.set_training_mode(True)

        # Handle case where no episodes completed
        if not scores:
            return {
                'mean_score': 0.0,
                'std_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'mean_length': 0.0,
                'success_rate': 0.0
            }

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'mean_length': np.mean(lengths),
            'success_rate': np.mean([s > 0 for s in scores])
        }

    def _run_episode(self, training: bool = True, render: bool = None) -> Dict[str, Any]:
        """
        Run a single episode.

        Args:
            training: Whether in training mode
            render: Whether to render (overrides default)

        Returns:
            Episode statistics
        """
        if render is None:
            render = self.enable_rendering and training

        state, _ = self.env.reset()
        self.agent.reset_episode()

        total_reward = 0
        steps = 0
        episode_metrics = []
        quit_requested = False

        while True:
            # Render if enabled
            if render and self.renderer:
                # Get agent info for display
                agent_info = self.agent.get_training_info()

                # Update stats display
                stats_data = {
                    'episode': self.episode,
                    'score': total_reward,
                    'steps': steps,
                    'epsilon': agent_info.get('epsilon', 0.0),
                    'avg_score': np.mean(self.episode_scores[-100:]) if self.episode_scores else 0.0
                }

                self.renderer.render(
                    self.env.game,
                    episode=stats_data['episode'],
                    epsilon=stats_data['epsilon'],
                    total_reward=stats_data['score'],
                    steps=stats_data['steps']
                )

                # Handle events (for controls like pause, quit, speed change)
                if not self.renderer.handle_events():
                    print("Quit requested by user")
                    quit_requested = True
                    break

                # Handle pause state
                while self.renderer.paused:
                    if not self.renderer.handle_events():
                        print("Quit requested by user")
                        quit_requested = True
                        break
                    self.renderer.clock.tick(10)  # Limit CPU usage during pause

                # Check if quit was requested during pause
                if quit_requested:
                    break

            # Select and execute action
            action = self.agent.select_action(state, training=training)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Store transition if training
            if training:
                # Flatten states for storage (agent will reshape as needed)
                state_flat = state.flatten() if state.ndim > 1 else state
                next_state_flat = next_state.flatten() if next_state.ndim > 1 else next_state
                self.agent.store_transition(state_flat, action, reward, next_state_flat, terminated or truncated)

                # Update agent
                if hasattr(self.agent, 'update'):
                    metrics = self.agent.update()
                    if metrics:
                        episode_metrics.append(metrics)

            total_reward += reward
            steps += 1
            state = next_state

            # Check if episode is done
            if terminated or truncated:
                break

        return {
            'score': total_reward,
            'length': steps,
            'terminated': terminated,
            'truncated': truncated,
            'food_eaten': info.get('food_eaten', 0),
            'metrics': episode_metrics,
            'quit_requested': quit_requested
        }

    def _update_statistics(self, episode_stats: Dict[str, Any]) -> None:
        """Update training statistics."""
        score = episode_stats['score']
        length = episode_stats['length']

        self.episode_scores.append(score)
        self.episode_lengths.append(length)

        # Update stats tracker
        self.stats_tracker.add_episode(
            score=int(score),
            total_reward=score,
            episode_length=length
        )

        # Track training metrics if available
        if episode_stats['metrics']:
            avg_metrics = {}
            for key in episode_stats['metrics'][0].keys():
                avg_metrics[key] = np.mean([m[key] for m in episode_stats['metrics']])
            self.training_metrics.append(avg_metrics)

        # Update best score
        if score > self.best_score:
            self.best_score = score
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

    def _log_progress(self, episode: int, episode_stats: Dict[str, Any]) -> None:
        """Log training progress."""
        score = episode_stats['score']
        length = episode_stats['length']

        # Calculate recent averages
        recent_scores = self.episode_scores[-100:] if len(self.episode_scores) >= 100 else self.episode_scores
        avg_score = np.mean(recent_scores)
        avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)

        # Get agent-specific info
        agent_info = self.agent.get_training_info()

        print(f"Episode {episode:4d} | "
              f"Score: {score:6.1f} | "
              f"Length: {length:3d} | "
              f"Avg(100): {avg_score:6.1f} | "
              f"Best: {self.best_score:6.1f} | "
              f"Epsilon: {agent_info.get('epsilon', 0):.3f}")

    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.episodes_without_improvement >= self.training_config['patience']:
            return True

        # Check if target score is reached
        if len(self.episode_scores) >= 100:
            recent_avg = np.mean(self.episode_scores[-100:])
            if recent_avg >= self.training_config['target_score']:
                print(f"Target score {self.training_config['target_score']} reached!")
                return True

        return False

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"{self.agent_name}_checkpoint_ep{episode}.pth"
        self.agent.save_model(str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")

    def _save_final_model(self) -> None:
        """Save the final trained model."""
        final_path = self.save_dir / f"{self.agent_name}_final.pth"
        best_path = self.save_dir / f"{self.agent_name}_best.pth"

        self.agent.save_model(str(final_path))

        # Save best model if we have a good score
        if self.best_score > 0:
            self.agent.save_model(str(best_path))
            print(f"Best model saved: {best_path} (score: {self.best_score:.1f})")

        print(f"Final model saved: {final_path}")

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0

        return {
            'agent_name': self.agent_name,
            'total_episodes': len(self.episode_scores),
            'training_time': training_time,
            'best_score': self.best_score,
            'final_score': self.episode_scores[-1] if self.episode_scores else 0,
            'average_score': np.mean(self.episode_scores),
            'average_length': np.mean(self.episode_lengths),
            'convergence_episode': self._find_convergence_episode(),
            'total_steps': sum(self.episode_lengths)
        }

    def _find_convergence_episode(self) -> Optional[int]:
        """Find the episode where the agent converged (reached stable performance)."""
        if len(self.episode_scores) < 200:
            return None

        # Look for the point where the 100-episode average stabilizes
        window = 100
        threshold = 5.0  # Score threshold for considering convergence

        for i in range(window, len(self.episode_scores) - window):
            recent_avg = np.mean(self.episode_scores[i:i+window])
            if recent_avg >= threshold:
                return i

        return None

    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """
        Plot training progress.

        Args:
            save_path: Path to save the plot
        """
        if not self.episode_scores:
            print("No training data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.agent_name.upper()} Training Progress')

        # Episode scores
        axes[0, 0].plot(self.episode_scores, alpha=0.7, label='Episode Score')
        if len(self.episode_scores) >= 100:
            # Moving average
            moving_avg = [np.mean(self.episode_scores[max(0, i-99):i+1])
                         for i in range(len(self.episode_scores))]
            axes[0, 0].plot(moving_avg, label='Moving Average (100)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Episode Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)

        # Training metrics (if available)
        if self.training_metrics and 'loss' in self.training_metrics[0]:
            losses = [m['loss'] for m in self.training_metrics]
            axes[1, 0].plot(losses, color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].grid(True)

        # Epsilon decay (for DQN)
        if self.training_metrics and 'epsilon' in self.training_metrics[0]:
            epsilons = [m['epsilon'] for m in self.training_metrics]
            axes[1, 1].plot(epsilons, color='green')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved: {save_path}")

        plt.show()