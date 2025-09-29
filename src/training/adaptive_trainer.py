"""
Adaptive Grid Size Trainer for Snake RL.

Automatically increases grid size during training based on agent performance.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from src.agent.base_agent import BaseAgent
from src.environment.snake_env import SnakeEnv
from src.training.trainer import UniversalTrainer
from src.utils.config import get_training_config


class AdaptiveGridTrainer:
    """
    Adaptive trainer that automatically increases grid size during training.

    The grid size grows based on agent performance, allowing for natural
    curriculum learning progression from easy to difficult tasks.
    """

    def __init__(
        self,
        agent_type: str,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
        enable_rendering: bool = True,
        save_dir: str = "models/adaptive"
    ):
        """
        Initialize adaptive grid trainer.

        Args:
            agent_type: Type of agent ("dqn", "reinforce", "ppo")
            device: Device for training (CPU/GPU)
            config: Training configuration
            enable_rendering: Whether to enable visualization
            save_dir: Directory to save adaptive models
        """
        self.agent_type = agent_type
        self.device = device
        self.config = config or get_training_config()
        self.enable_rendering = enable_rendering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Adaptive progression parameters
        self.initial_grid_size = 5
        self.max_grid_size = 20
        self.grid_increment = 2
        self.current_grid_size = self.initial_grid_size

        # Performance criteria for advancement
        self.advancement_window = 50    # Episodes to average performance
        self.score_threshold_base = 10  # Base score threshold
        self.success_rate_threshold = 0.6  # 60% of episodes must be "successful"

        # Training tracking
        self.episode_count = 0
        self.performance_history = []
        self.grid_transitions = []
        self.agent = None
        self.current_env = None

        print(f"🔄 ADAPTIVE GRID TRAINER INITIALIZED")
        print(f"   Initial size: {self.initial_grid_size}x{self.initial_grid_size}")
        print(f"   Maximum size: {self.max_grid_size}x{self.max_grid_size}")
        print(f"   Increment: +{self.grid_increment}")
        print(f"   Advancement window: {self.advancement_window} episodes")

    def train_adaptive(
        self,
        total_episodes: int = 2000,
        episodes_per_check: int = 50,
        min_episodes_per_size: int = 100
    ) -> Dict[str, Any]:
        """
        Train agent with adaptive grid size progression.

        Args:
            total_episodes: Total episodes to train
            episodes_per_check: How often to check for advancement
            min_episodes_per_size: Minimum episodes before size can increase

        Returns:
            Training summary with grid progression history
        """
        print(f"\\n{'='*60}")
        print(f"ADAPTIVE GRID TRAINING: {self.agent_type.upper()}")
        print(f"{'='*60}")
        print(f"Total episodes: {total_episodes}")
        print(f"Episodes per check: {episodes_per_check}")
        print()

        overall_start_time = time.time()
        episodes_at_current_size = 0

        while self.episode_count < total_episodes:
            # Create/update environment for current grid size
            if self.current_env is None or self.current_env.grid_size != self.current_grid_size:
                self._setup_environment()
                episodes_at_current_size = 0

            # Train for episodes_per_check episodes
            remaining_episodes = min(episodes_per_check, total_episodes - self.episode_count)

            print(f"\\n📊 Training on {self.current_grid_size}x{self.current_grid_size} grid...")
            print(f"   Episodes {self.episode_count + 1}-{self.episode_count + remaining_episodes}")

            batch_summary = self._train_batch(remaining_episodes)
            episodes_at_current_size += remaining_episodes

            # Record performance - get scores from this batch
            self.performance_history.extend(batch_summary['episode_scores'])

            # Check for advancement (but only after minimum episodes)
            if (episodes_at_current_size >= min_episodes_per_size and
                self.current_grid_size < self.max_grid_size):

                if self._should_advance():
                    old_size = self.current_grid_size
                    self.current_grid_size = min(
                        self.max_grid_size,
                        self.current_grid_size + self.grid_increment
                    )

                    # Record transition
                    transition = {
                        'episode': self.episode_count,
                        'from_size': old_size,
                        'to_size': self.current_grid_size,
                        'episodes_at_size': episodes_at_current_size,
                        'performance_avg': np.mean(self.performance_history[-self.advancement_window:])
                    }
                    self.grid_transitions.append(transition)

                    print(f"\\n🎯 GRID SIZE ADVANCEMENT!")
                    print(f"   {old_size}x{old_size} → {self.current_grid_size}x{self.current_grid_size}")
                    print(f"   After {episodes_at_current_size} episodes")
                    print(f"   Performance: {transition['performance_avg']:.1f}")

                    # Save checkpoint at transition
                    checkpoint_path = self.save_dir / f"checkpoint_grid_{self.current_grid_size}x{self.current_grid_size}.pth"
                    self.agent.save_model(str(checkpoint_path))
                    print(f"   Checkpoint saved: {checkpoint_path}")

        overall_time = time.time() - overall_start_time

        # Save final model
        final_model_path = self.save_dir / f"{self.agent_type}_adaptive_final.pth"
        self.agent.save_model(str(final_model_path))

        # Create comprehensive summary
        summary = {
            'agent_type': self.agent_type,
            'total_episodes': self.episode_count,
            'total_training_time': overall_time,
            'final_grid_size': self.current_grid_size,
            'grid_transitions': self.grid_transitions,
            'performance_history': self.performance_history,
            'final_model_path': str(final_model_path),
            'reached_max_size': self.current_grid_size >= self.max_grid_size
        }

        self._print_adaptive_summary(summary)
        return summary

    def _setup_environment(self) -> None:
        """Setup environment and agent for current grid size."""
        print(f"🏗️  Setting up {self.current_grid_size}x{self.current_grid_size} environment...")

        # Create new environment
        self.current_env = SnakeEnv(
            grid_size=self.current_grid_size,
            max_steps=1000,
            agent_type=self.agent_type
        )

        # Create or transfer agent
        if self.agent is None:
            # Create new agent
            self.agent = self._create_agent()
            print(f"   ✨ Created new {self.agent_type.upper()} agent")
        else:
            # Transfer to new size
            self.agent = self._transfer_agent()
            print(f"   🔄 Transferred agent to new size")

    def _create_agent(self) -> BaseAgent:
        """Create new agent for current environment."""
        from src.main import create_agent

        state_shape = self.current_env.observation_space.shape
        state_dim = state_shape[0] * state_shape[1]
        action_dim = self.current_env.action_space.n

        return create_agent(self.agent_type, state_dim, action_dim, self.device)

    def _transfer_agent(self) -> BaseAgent:
        """Transfer agent from previous size to current size."""
        # Save current state
        old_state_dict = self.agent.get_model_state()

        # Create new agent for new size
        new_agent = self._create_agent()

        # Transfer compatible weights
        new_state_dict = new_agent.get_model_state()
        transferred_keys = []

        for key in old_state_dict:
            if key in new_state_dict:
                old_shape = old_state_dict[key].shape
                new_shape = new_state_dict[key].shape

                if old_shape == new_shape:
                    # Perfect match
                    new_state_dict[key] = old_state_dict[key].clone()
                    transferred_keys.append(key)
                elif len(old_shape) >= 2 and old_shape[1:] == new_shape[1:]:
                    # Input size changed but other dims match
                    if old_shape[0] <= new_shape[0]:
                        new_state_dict[key][:old_shape[0]] = old_state_dict[key]
                        transferred_keys.append(f"{key} (partial)")

        new_agent.load_model_state(new_state_dict)
        print(f"      Transferred {len(transferred_keys)} parameter groups")

        return new_agent

    def _train_batch(self, num_episodes: int) -> Dict[str, Any]:
        """Train for a batch of episodes and return performance summary."""
        trainer = UniversalTrainer(
            agent=self.agent,
            env=self.current_env,
            agent_name=f"{self.agent_type}_adaptive",
            enable_rendering=self.enable_rendering,
            save_dir=str(self.save_dir / "temp")
        )

        # Train batch
        batch_summary = trainer.train(
            num_episodes=num_episodes,
            evaluate_freq=max(num_episodes // 2, 10)
        )

        # Add episode scores to summary (trainer doesn't include them by default)
        batch_summary['episode_scores'] = trainer.episode_scores.copy()

        self.episode_count += num_episodes
        return batch_summary

    def _should_advance(self) -> bool:
        """
        Determine if agent should advance to next grid size.

        Uses recent performance history to make decision.
        """
        if len(self.performance_history) < self.advancement_window:
            return False

        recent_scores = self.performance_history[-self.advancement_window:]

        # Calculate dynamic threshold based on grid size
        # Larger grids are harder, so lower thresholds
        size_difficulty_factor = 1.0 - (self.current_grid_size - self.initial_grid_size) * 0.1
        threshold = self.score_threshold_base * size_difficulty_factor

        # Count successful episodes
        successful_episodes = sum(1 for score in recent_scores if score > threshold)
        success_rate = successful_episodes / len(recent_scores)

        # Also check average performance trend
        avg_score = np.mean(recent_scores)

        advancement_criteria = {
            'success_rate': success_rate >= self.success_rate_threshold,
            'avg_score_positive': avg_score > 0,  # At least break-even
            'threshold_met': avg_score > threshold * 0.5  # Softer threshold for average
        }

        should_advance = all(advancement_criteria.values())

        print(f"\\n📈 Performance Check:")
        print(f"   Recent average: {avg_score:.1f}")
        print(f"   Success rate: {success_rate:.1%} (need {self.success_rate_threshold:.1%})")
        print(f"   Threshold: {threshold:.1f}")
        print(f"   Should advance: {'✅ YES' if should_advance else '❌ NO'}")

        return should_advance

    def _print_adaptive_summary(self, summary: Dict[str, Any]) -> None:
        """Print comprehensive adaptive training summary."""
        print(f"\\n{'='*60}")
        print(f"ADAPTIVE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Agent: {summary['agent_type'].upper()}")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Total time: {summary['total_training_time']:.1f}s")
        print(f"Final grid size: {summary['final_grid_size']}x{summary['final_grid_size']}")
        print(f"Reached max size: {'✅ YES' if summary['reached_max_size'] else '❌ NO'}")
        print()

        print("📊 GRID SIZE PROGRESSION:")
        print("-" * 40)
        current_size = self.initial_grid_size
        print(f"Started: {current_size}x{current_size}")

        for i, transition in enumerate(summary['grid_transitions']):
            print(f"Step {i+1}: {transition['from_size']}x{transition['from_size']} → "
                  f"{transition['to_size']}x{transition['to_size']} "
                  f"(episode {transition['episode']}, avg: {transition['performance_avg']:.1f})")
            current_size = transition['to_size']

        if summary['performance_history']:
            final_avg = np.mean(summary['performance_history'][-50:])  # Last 50 episodes
            print(f"Final performance: {final_avg:.1f} (last 50 episodes)")

        print(f"\\n💾 Final model: {summary['final_model_path']}")
        print(f"🎯 Grid transitions: {len(summary['grid_transitions'])}")
        print(f"{'='*60}")