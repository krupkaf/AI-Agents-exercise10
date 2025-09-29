"""
Curriculum Learning Trainer for Snake RL.

Trains agents on progressively larger grids to improve learning success rate.
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


class CurriculumTrainer:
    """
    Curriculum learning trainer that progressively increases grid size.

    Trains agents on small grids where food finding is easier, then transfers
    learned knowledge to larger grids. This helps overcome the exploration
    problem in sparse reward environments.
    """

    def __init__(
        self,
        agent_type: str,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
        enable_rendering: bool = True,
        save_dir: str = "models/curriculum"
    ):
        """
        Initialize curriculum learning trainer.

        Args:
            agent_type: Type of agent ("dqn", "reinforce", "ppo")
            device: Device for training (CPU/GPU)
            config: Training configuration
            enable_rendering: Whether to enable visualization
            save_dir: Directory to save curriculum models
        """
        self.agent_type = agent_type
        self.device = device
        self.config = config or get_training_config()
        self.enable_rendering = enable_rendering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Default curriculum stages: [5x5, 8x8, 12x12, 16x16, 20x20]
        self.curriculum_stages = [5, 8, 12, 16, 20]
        self.episodes_per_stage = [100, 150, 200, 250, 300]  # More episodes for larger grids

        # Track training progress
        self.current_stage = 0
        self.training_history = []

    def train_curriculum(self) -> Dict[str, Any]:
        """
        Train agent through full curriculum from small to large grids.

        Returns:
            Training summary with results from all stages
        """
        print(f"\\n{'='*60}")
        print(f"CURRICULUM LEARNING: {self.agent_type.upper()} AGENT")
        print(f"{'='*60}")
        print(f"Stages: {self.curriculum_stages}")
        print(f"Episodes per stage: {self.episodes_per_stage}")
        print()

        agent = None
        overall_start_time = time.time()

        for stage_idx, (grid_size, episodes) in enumerate(zip(self.curriculum_stages, self.episodes_per_stage)):
            print(f"\\n🎯 STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: {grid_size}x{grid_size} grid")
            print(f"Episodes: {episodes}")
            print("-" * 40)

            # Create environment for this stage
            env = SnakeEnv(
                grid_size=grid_size,
                max_steps=1000,
                agent_type=self.agent_type
            )

            # Create or transfer agent
            if agent is None:
                # First stage - create new agent
                agent = self._create_agent(env)
                print(f"✨ Created new {self.agent_type.upper()} agent")
            else:
                # Transfer from previous stage
                agent = self._transfer_agent(agent, env)
                print(f"🔄 Transferred agent from {self.curriculum_stages[stage_idx-1]}x{self.curriculum_stages[stage_idx-1]} to {grid_size}x{grid_size}")

            # Train on this stage
            stage_start_time = time.time()
            trainer = UniversalTrainer(
                agent=agent,
                env=env,
                agent_name=f"{self.agent_type}_stage{stage_idx+1}",
                enable_rendering=self.enable_rendering and stage_idx >= 2,  # Only render later stages
                save_dir=str(self.save_dir / f"stage_{stage_idx+1}")
            )

            training_summary = trainer.train(
                num_episodes=episodes,
                evaluate_freq=max(episodes // 4, 25)  # Eval 4 times per stage
            )

            stage_time = time.time() - stage_start_time

            # Save stage results
            stage_results = {
                'stage': stage_idx + 1,
                'grid_size': grid_size,
                'episodes': episodes,
                'training_time': stage_time,
                'best_score': training_summary['best_score'],
                'final_score': training_summary['final_score'],
                'average_score': training_summary['average_score'],
                'convergence_episode': training_summary.get('convergence_episode')
            }
            self.training_history.append(stage_results)

            # Save model for this stage
            stage_model_path = self.save_dir / f"stage_{stage_idx+1}_model.pth"
            agent.save_model(str(stage_model_path))

            print(f"✅ Stage {stage_idx + 1} completed in {stage_time:.1f}s")
            print(f"   Best score: {training_summary['best_score']:.2f}")
            print(f"   Model saved: {stage_model_path}")

            # Early stopping if agent mastered this stage
            if training_summary['best_score'] > 50:  # Positive score threshold
                print(f"🏆 Agent mastered {grid_size}x{grid_size} grid! (score > 50)")

            self.current_stage = stage_idx + 1

        overall_time = time.time() - overall_start_time

        # Save final model
        final_model_path = self.save_dir / f"{self.agent_type}_curriculum_final.pth"
        agent.save_model(str(final_model_path))

        # Create summary
        summary = {
            'agent_type': self.agent_type,
            'total_stages': len(self.curriculum_stages),
            'total_training_time': overall_time,
            'stages_completed': self.current_stage,
            'stage_history': self.training_history,
            'final_model_path': str(final_model_path),
            'curriculum_success': self.current_stage == len(self.curriculum_stages)
        }

        self._print_curriculum_summary(summary)
        return summary

    def _create_agent(self, env: SnakeEnv) -> BaseAgent:
        """Create new agent for the first curriculum stage."""
        from src.main import create_agent

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] * state_shape[1]
        action_dim = env.action_space.n

        return create_agent(self.agent_type, state_dim, action_dim, self.device)

    def _transfer_agent(self, old_agent: BaseAgent, new_env: SnakeEnv) -> BaseAgent:
        """
        Transfer agent from smaller to larger grid.

        For convolutional networks, this works automatically.
        For fully connected networks, we need to adapt the input layer.
        """
        # Create new agent for new environment size
        new_agent = self._create_agent(new_env)

        # Transfer weights where possible
        old_state_dict = old_agent.get_model_state()
        new_state_dict = new_agent.get_model_state()

        # Transfer compatible layers (convolutional layers transfer perfectly)
        transferred_keys = []
        for key in old_state_dict:
            if key in new_state_dict:
                old_shape = old_state_dict[key].shape
                new_shape = new_state_dict[key].shape

                if old_shape == new_shape:
                    # Perfect match - transfer directly
                    new_state_dict[key] = old_state_dict[key].clone()
                    transferred_keys.append(key)
                elif len(old_shape) == len(new_shape) and old_shape[1:] == new_shape[1:]:
                    # Input dimension changed (fc layer) - partial transfer
                    if old_shape[1] < new_shape[1]:  # Expanding input
                        new_state_dict[key][:old_shape[0], :old_shape[1]] = old_state_dict[key]
                        transferred_keys.append(f"{key} (partial)")

        new_agent.load_model_state(new_state_dict)

        print(f"   Transferred {len(transferred_keys)} parameter groups")
        return new_agent

    def _print_curriculum_summary(self, summary: Dict[str, Any]) -> None:
        """Print comprehensive curriculum training summary."""
        print(f"\\n{'='*60}")
        print(f"CURRICULUM TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Agent: {summary['agent_type'].upper()}")
        print(f"Total time: {summary['total_training_time']:.1f}s")
        print(f"Stages completed: {summary['stages_completed']}/{summary['total_stages']}")
        print(f"Success: {'✅ YES' if summary['curriculum_success'] else '❌ NO'}")
        print()

        print("📊 STAGE PROGRESSION:")
        print("-" * 40)
        for stage in summary['stage_history']:
            status = "🏆" if stage['best_score'] > 50 else "📈" if stage['best_score'] > 0 else "📉"
            print(f"Stage {stage['stage']}: {stage['grid_size']}x{stage['grid_size']} | "
                  f"Best: {stage['best_score']:6.1f} | "
                  f"Time: {stage['training_time']:5.1f}s | {status}")

        print(f"\\n💾 Final model: {summary['final_model_path']}")
        print(f"{'='*60}")