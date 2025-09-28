import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Type
from pathlib import Path

from src.agent.base_agent import BaseAgent
from src.agent.dqn_agent import DQNAgent
from src.agent.reinforce_agent import REINFORCEAgent
from src.agent.ppo_agent import PPOAgent
from src.environment.snake_env import SnakeEnv
from src.training.trainer import UniversalTrainer
from src.utils.config import get_agent_config


class AgentComparison:
    """
    Framework for comparing multiple RL agents on the Snake environment.
    Provides training, evaluation, and visualization tools for agent comparison.
    """

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        results_dir: str = "results/comparison",
        models_dir: str = "models/comparison"
    ):
        """
        Initialize the comparison framework.

        Args:
            env_config: Environment configuration
            results_dir: Directory to save comparison results
            models_dir: Directory to save trained models
        """
        self.env_config = env_config or {}
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment for testing
        self.env = SnakeEnv(
            grid_size=self.env_config.get('grid_size', 20),
            max_steps=self.env_config.get('max_steps', 1000)
        )

        # Available agent types
        self.agent_types = {
            'dqn': DQNAgent,
            'reinforce': REINFORCEAgent,
            'ppo': PPOAgent
        }

        # Results storage
        self.comparison_results: Dict[str, Dict[str, Any]] = {}

    def run_comparison(
        self,
        agents_to_compare: List[str],
        episodes_per_agent: int = 1000,
        evaluation_episodes: int = 50,
        enable_rendering: bool = False,
        save_models: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a comprehensive comparison of multiple agents.

        Args:
            agents_to_compare: List of agent names to compare
            episodes_per_agent: Number of training episodes per agent
            evaluation_episodes: Number of episodes for final evaluation
            enable_rendering: Whether to enable PyGame rendering during training
            save_models: Whether to save trained models

        Returns:
            Dictionary containing comparison results for each agent
        """
        print(f"Starting agent comparison: {agents_to_compare}")
        print(f"Training episodes per agent: {episodes_per_agent}")
        print(f"Evaluation episodes: {evaluation_episodes}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        comparison_start_time = time.time()

        for agent_name in agents_to_compare:
            print(f"\n{'='*60}")
            print(f"Training {agent_name.upper()} Agent")
            print(f"{'='*60}")

            try:
                # Train agent
                training_results = self._train_single_agent(
                    agent_name=agent_name,
                    episodes=episodes_per_agent,
                    device=device,
                    enable_rendering=enable_rendering,
                    save_model=save_models
                )

                # Evaluate agent
                evaluation_results = self._evaluate_single_agent(
                    agent_name=agent_name,
                    episodes=evaluation_episodes,
                    device=device
                )

                # Combine results
                self.comparison_results[agent_name] = {
                    **training_results,
                    **evaluation_results,
                    'status': 'completed'
                }

                print(f"\n{agent_name.upper()} Results Summary:")
                print(f"  Training Episodes: {training_results['total_episodes']}")
                print(f"  Training Time: {training_results['training_time']:.1f}s")
                print(f"  Best Training Score: {training_results['best_score']:.1f}")
                print(f"  Final Evaluation Score: {evaluation_results['eval_mean_score']:.1f} ± {evaluation_results['eval_std_score']:.1f}")

            except Exception as e:
                print(f"Error training {agent_name}: {e}")
                self.comparison_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

        total_time = time.time() - comparison_start_time
        print(f"\n{'='*60}")
        print(f"Comparison completed in {total_time:.1f}s")
        print(f"{'='*60}")

        # Save comparison results
        self._save_comparison_results()

        # Generate comparison plots
        self._plot_comparison_results()

        # Print final summary
        self._print_comparison_summary()

        return self.comparison_results

    def _train_single_agent(
        self,
        agent_name: str,
        episodes: int,
        device: torch.device,
        enable_rendering: bool = False,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train a single agent.

        Args:
            agent_name: Name of the agent to train
            episodes: Number of training episodes
            device: PyTorch device
            enable_rendering: Whether to enable rendering
            save_model: Whether to save the model

        Returns:
            Training results dictionary
        """
        # Create fresh environment
        env = SnakeEnv(
            grid_size=self.env_config.get('grid_size', 20),
            max_steps=self.env_config.get('max_steps', 1000)
        )

        # Get agent configuration
        agent_config = get_agent_config(agent_name)

        # Create agent
        state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 400
        action_dim = env.action_space.n
        agent_class = self.agent_types[agent_name]

        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            config=agent_config
        )

        # Create trainer
        save_dir = self.models_dir / agent_name if save_model else None
        trainer = UniversalTrainer(
            agent=agent,
            env=env,
            agent_name=agent_name,
            enable_rendering=enable_rendering,
            save_dir=str(save_dir) if save_dir else "temp"
        )

        # Train agent
        training_results = trainer.train(num_episodes=episodes)

        return training_results

    def _evaluate_single_agent(
        self,
        agent_name: str,
        episodes: int,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Evaluate a single trained agent.

        Args:
            agent_name: Name of the agent to evaluate
            episodes: Number of evaluation episodes
            device: PyTorch device

        Returns:
            Evaluation results dictionary
        """
        # Create fresh environment for evaluation
        env = SnakeEnv(
            grid_size=self.env_config.get('grid_size', 20),
            max_steps=self.env_config.get('max_steps', 1000)
        )

        # Get agent configuration
        agent_config = get_agent_config(agent_name)

        # Create agent
        state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 400
        action_dim = env.action_space.n
        agent_class = self.agent_types[agent_name]

        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            config=agent_config
        )

        # Load trained model
        model_path = self.models_dir / agent_name / f"{agent_name}_best.pth"
        if not model_path.exists():
            model_path = self.models_dir / agent_name / f"{agent_name}_final.pth"

        if model_path.exists():
            agent.load_model(str(model_path))
            print(f"Loaded model: {model_path}")
        else:
            print(f"Warning: No trained model found for {agent_name}")

        # Create trainer for evaluation
        trainer = UniversalTrainer(
            agent=agent,
            env=env,
            agent_name=agent_name,
            enable_rendering=False,
            save_dir="temp"
        )

        # Evaluate agent
        eval_results = trainer.evaluate(num_episodes=episodes, render=False)

        # Prefix keys for clarity
        return {f"eval_{k}": v for k, v in eval_results.items()}

    def _save_comparison_results(self) -> None:
        """Save comparison results to JSON file."""
        results_file = self.results_dir / "comparison_results.json"

        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for agent_name, results in self.comparison_results.items():
            serializable_results[agent_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[agent_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_results[agent_name][key] = value.item()
                else:
                    serializable_results[agent_name][key] = value

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"Comparison results saved: {results_file}")

    def _plot_comparison_results(self) -> None:
        """Generate comprehensive comparison plots."""
        # Filter successful results
        successful_results = {
            name: results for name, results in self.comparison_results.items()
            if results.get('status') == 'completed'
        }

        if not successful_results:
            print("No successful training results to plot")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Agent Comparison Results', fontsize=16)

        # 1. Training Performance
        agent_names = list(successful_results.keys())
        best_scores = [results['best_score'] for results in successful_results.values()]
        final_scores = [results['final_score'] for results in successful_results.values()]

        x_pos = np.arange(len(agent_names))
        width = 0.35

        axes[0, 0].bar(x_pos - width/2, best_scores, width, label='Best Score', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, final_scores, width, label='Final Score', alpha=0.8)
        axes[0, 0].set_xlabel('Agent')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Training Scores')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name.upper() for name in agent_names])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Evaluation Performance
        eval_means = [results['eval_mean_score'] for results in successful_results.values()]
        eval_stds = [results['eval_std_score'] for results in successful_results.values()]

        axes[0, 1].bar(agent_names, eval_means, yerr=eval_stds, capsize=5, alpha=0.8)
        axes[0, 1].set_xlabel('Agent')
        axes[0, 1].set_ylabel('Evaluation Score')
        axes[0, 1].set_title('Final Evaluation Performance')
        axes[0, 1].set_xticklabels([name.upper() for name in agent_names])
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Training Efficiency
        training_times = [results['training_time'] for results in successful_results.values()]
        total_episodes = [results['total_episodes'] for results in successful_results.values()]

        axes[0, 2].bar(agent_names, training_times, alpha=0.8)
        axes[0, 2].set_xlabel('Agent')
        axes[0, 2].set_ylabel('Training Time (seconds)')
        axes[0, 2].set_title('Training Efficiency')
        axes[0, 2].set_xticklabels([name.upper() for name in agent_names])
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Episode Lengths
        avg_lengths = [results['average_length'] for results in successful_results.values()]

        axes[1, 0].bar(agent_names, avg_lengths, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Agent')
        axes[1, 0].set_ylabel('Average Episode Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xticklabels([name.upper() for name in agent_names])
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Success Rates
        success_rates = [results.get('eval_success_rate', 0) * 100 for results in successful_results.values()]

        axes[1, 1].bar(agent_names, success_rates, alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Agent')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Evaluation Success Rate')
        axes[1, 1].set_xticklabels([name.upper() for name in agent_names])
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance Summary Table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')

        # Create summary table
        table_data = []
        headers = ['Agent', 'Best Score', 'Eval Score', 'Success Rate', 'Training Time']

        for name, results in successful_results.items():
            row = [
                name.upper(),
                f"{results['best_score']:.1f}",
                f"{results['eval_mean_score']:.1f}±{results['eval_std_score']:.1f}",
                f"{results.get('eval_success_rate', 0)*100:.1f}%",
                f"{results['training_time']:.0f}s"
            ]
            table_data.append(row)

        table = axes[1, 2].table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        axes[1, 2].set_title('Performance Summary')

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / "agent_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {plot_path}")

        plt.show()

    def _print_comparison_summary(self) -> None:
        """Print a detailed comparison summary."""
        print(f"\n{'='*80}")
        print("AGENT COMPARISON SUMMARY")
        print(f"{'='*80}")

        successful_results = {
            name: results for name, results in self.comparison_results.items()
            if results.get('status') == 'completed'
        }

        if not successful_results:
            print("No agents completed training successfully.")
            return

        # Find best performers
        best_training = max(successful_results.items(), key=lambda x: x[1]['best_score'])
        best_evaluation = max(successful_results.items(), key=lambda x: x[1]['eval_mean_score'])
        most_efficient = min(successful_results.items(), key=lambda x: x[1]['training_time'])

        print(f"\n🏆 BEST TRAINING PERFORMANCE:")
        print(f"   Agent: {best_training[0].upper()}")
        print(f"   Best Score: {best_training[1]['best_score']:.1f}")

        print(f"\n🎯 BEST EVALUATION PERFORMANCE:")
        print(f"   Agent: {best_evaluation[0].upper()}")
        print(f"   Evaluation Score: {best_evaluation[1]['eval_mean_score']:.1f} ± {best_evaluation[1]['eval_std_score']:.1f}")

        print(f"\n⚡ MOST EFFICIENT TRAINING:")
        print(f"   Agent: {most_efficient[0].upper()}")
        print(f"   Training Time: {most_efficient[1]['training_time']:.1f}s")

        print(f"\n📊 DETAILED RESULTS:")
        print(f"{'Agent':<12} {'Best Score':<12} {'Eval Score':<15} {'Success Rate':<12} {'Time (s)':<10}")
        print("-" * 65)

        for name, results in successful_results.items():
            eval_score = f"{results['eval_mean_score']:.1f}±{results['eval_std_score']:.1f}"
            success_rate = f"{results.get('eval_success_rate', 0)*100:.1f}%"

            print(f"{name.upper():<12} {results['best_score']:<12.1f} {eval_score:<15} {success_rate:<12} {results['training_time']:<10.0f}")

        # Failed agents
        failed_results = {
            name: results for name, results in self.comparison_results.items()
            if results.get('status') == 'failed'
        }

        if failed_results:
            print(f"\n❌ FAILED AGENTS:")
            for name, results in failed_results.items():
                print(f"   {name.upper()}: {results.get('error', 'Unknown error')}")

        print(f"\n{'='*80}")

    def load_comparison_results(self, results_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load comparison results from a JSON file.

        Args:
            results_file: Path to the results JSON file

        Returns:
            Dictionary containing comparison results
        """
        with open(results_file, 'r') as f:
            self.comparison_results = json.load(f)

        print(f"Loaded comparison results from: {results_file}")
        return self.comparison_results

    def compare_specific_agents(
        self,
        agent_configs: Dict[str, Dict[str, Any]],
        episodes_per_agent: int = 500
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare agents with specific configurations.

        Args:
            agent_configs: Dictionary mapping agent names to their configurations
            episodes_per_agent: Number of training episodes per agent

        Returns:
            Comparison results
        """
        print("Running custom agent comparison...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for agent_name, config in agent_configs.items():
            print(f"\nTraining {agent_name} with custom config...")

            # Create environment
            env = SnakeEnv(
                grid_size=self.env_config.get('grid_size', 20),
                max_steps=self.env_config.get('max_steps', 1000)
            )

            # Create agent with custom config
            state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 400
            action_dim = env.action_space.n
            agent_class = self.agent_types[config['agent_type']]

            agent = agent_class(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                config=config
            )

            # Train and evaluate
            trainer = UniversalTrainer(
                agent=agent,
                env=env,
                agent_name=agent_name,
                enable_rendering=False,
                save_dir=str(self.models_dir / agent_name)
            )

            training_results = trainer.train(num_episodes=episodes_per_agent)
            eval_results = trainer.evaluate(num_episodes=20)

            self.comparison_results[agent_name] = {
                **training_results,
                **{f"eval_{k}": v for k, v in eval_results.items()},
                'config': config,
                'status': 'completed'
            }

        self._save_comparison_results()
        self._plot_comparison_results()
        self._print_comparison_summary()

        return self.comparison_results