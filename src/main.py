#!/usr/bin/env python3
"""
Main script for training and testing Snake RL agents.
Supports multiple algorithms: DQN, REINFORCE, PPO.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.snake_env import SnakeEnv
from src.agent.dqn_agent import DQNAgent
from src.training.trainer import UniversalTrainer
from src.utils.config import (
    get_agent_config,
    get_env_config,
    get_device,
    EnvironmentConfig
)


def create_agent(agent_type: str, state_dim: int, action_dim: int, device: torch.device):
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent ('dqn', 'reinforce', 'ppo')
        state_dim: Dimension of state space
        action_dim: Number of actions
        device: PyTorch device

    Returns:
        Initialized agent
    """
    config = get_agent_config(agent_type)

    if agent_type == 'dqn':
        return DQNAgent(state_dim, action_dim, device, config)
    elif agent_type == 'reinforce':
        # TODO: Implement REINFORCE agent
        raise NotImplementedError("REINFORCE agent not yet implemented")
    elif agent_type == 'ppo':
        # TODO: Implement PPO agent
        raise NotImplementedError("PPO agent not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_agent(args):
    """Train an agent."""
    print(f"Training {args.agent.upper()} agent...")
    print(f"Episodes: {args.episodes}")
    print(f"Headless mode: {args.headless}")

    # Get device
    device = get_device()

    # Create environment
    env = SnakeEnv(
        grid_size=EnvironmentConfig.GRID_SIZE,
        max_steps=EnvironmentConfig.MAX_STEPS
    )

    # Get dimensions
    state_shape = env.observation_space.shape  # (grid_size, grid_size)
    state_dim = state_shape[0] * state_shape[1]  # Flattened size
    action_dim = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create agent
    agent = create_agent(args.agent, state_dim, action_dim, device)

    # Load model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading model from: {args.load_model}")
            agent.load_model(args.load_model)
        else:
            print(f"Warning: Model file not found: {args.load_model}")

    # Create trainer
    trainer = UniversalTrainer(
        agent=agent,
        env=env,
        agent_name=args.agent,
        enable_rendering=not args.headless,
        save_dir=args.save_dir
    )

    # Train
    training_summary = trainer.train(
        num_episodes=args.episodes,
        evaluate_freq=args.eval_freq
    )

    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for key, value in training_summary.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:8.2f}")
        else:
            print(f"{key:20s}: {value}")

    # Save training plot
    if args.save_plots:
        plot_path = Path(args.save_dir) / f"{args.agent}_training_progress.png"
        trainer.plot_training_progress(str(plot_path))


def test_agent(args):
    """Test a trained agent."""
    print(f"Testing {args.agent.upper()} agent...")

    if not args.load_model:
        print("Error: --load-model is required for testing")
        return

    if not os.path.exists(args.load_model):
        print(f"Error: Model file not found: {args.load_model}")
        return

    # Get device
    device = get_device()

    # Create environment
    env = SnakeEnv(
        grid_size=EnvironmentConfig.GRID_SIZE,
        max_steps=EnvironmentConfig.MAX_STEPS
    )

    # Get dimensions
    state_shape = env.observation_space.shape  # (grid_size, grid_size)
    state_dim = state_shape[0] * state_shape[1]  # Flattened size
    action_dim = env.action_space.n

    # Create and load agent
    agent = create_agent(args.agent, state_dim, action_dim, device)
    agent.load_model(args.load_model)
    agent.set_training_mode(False)

    # Create trainer for evaluation
    trainer = UniversalTrainer(
        agent=agent,
        env=env,
        agent_name=args.agent,
        enable_rendering=True,  # Always render during testing
        save_dir=args.save_dir
    )

    # Evaluate
    eval_stats = trainer.evaluate(num_episodes=args.test_episodes, render=True)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in eval_stats.items():
        print(f"{key:15s}: {value:8.2f}")


def compare_agents(args):
    """Compare multiple agents."""
    print("Comparing multiple agents...")
    print(f"Episodes per agent: {args.episodes}")

    # Agents to compare
    agents_to_compare = ['dqn']  # Start with DQN, add others when implemented

    results = {}

    for agent_type in agents_to_compare:
        print(f"\n{'='*20} Training {agent_type.upper()} {'='*20}")

        # Modify args for this agent
        args_copy = argparse.Namespace(**vars(args))
        args_copy.agent = agent_type
        args_copy.save_dir = f"{args.save_dir}/{agent_type}"

        try:
            # Train agent
            train_agent(args_copy)

            # Test agent
            args_copy.mode = 'test'
            args_copy.load_model = f"{args_copy.save_dir}/{agent_type}_best.pth"
            args_copy.test_episodes = 10

            # Only test if model exists
            if os.path.exists(args_copy.load_model):
                test_agent(args_copy)

        except Exception as e:
            print(f"Error training {agent_type}: {e}")
            continue

    print(f"\n{'='*50}")
    print("COMPARISON COMPLETED")
    print("="*50)


def _add_arguments(parser):
    """Add all command line arguments to the parser."""
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Mode to run (train or test)')

    # Agent selection
    parser.add_argument('--agent', choices=['dqn', 'reinforce', 'ppo'], default='dqn',
                        help='Type of agent to use')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train/test')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluation frequency during training')

    # Model management
    parser.add_argument('--load-model', type=str,
                        help='Path to load model from')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')

    # Testing parameters
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of episodes for testing')

    # Visualization options
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization (faster training)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training progress plots')

    # Special modes
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all available agents')

    # Demo mode
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with random agent')


def main():
    """Main function with argument parsing."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        print("Snake RL Training and Testing")
        print("=" * 40)
        print("No arguments provided. Here's how to use this script:\n")

        # Create parser to show help
        parser = argparse.ArgumentParser(description='Snake RL Training and Testing')

        # Add all arguments (will be added below)
        _add_arguments(parser)

        # Show help and exit
        parser.print_help()

        print("\nCommon usage examples:")
        print("  # Train DQN agent for 100 episodes")
        print("  uv run python src/main.py --agent dqn --episodes 100")
        print("")
        print("  # Train without visualization (faster)")
        print("  uv run python src/main.py --agent dqn --episodes 500 --headless")
        print("")
        print("  # Test a trained model")
        print("  uv run python src/main.py --mode test --agent dqn --load-model models/dqn_best.pth")
        print("")
        print("  # Run demo with random agent")
        print("  uv run python src/main.py --demo")
        print("")
        return

    parser = argparse.ArgumentParser(description='Snake RL Training and Testing')
    _add_arguments(parser)

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Handle special modes
    if args.compare_all:
        compare_agents(args)
        return

    if args.demo:
        print("Running demo with random agent...")
        # Import and run the existing demo
        from demo.visualization import demo_pygame_visualization
        demo_pygame_visualization()
        return

    # Run normal mode
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
