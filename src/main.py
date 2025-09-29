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
from typing import Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.snake_env import SnakeEnv
from src.agent.dqn_agent import DQNAgent
from src.agent.reinforce_agent import REINFORCEAgent
from src.agent.ppo_agent import PPOAgent
from src.agent.human_agent import HumanAgent
from src.training.trainer import UniversalTrainer
from src.training.comparison import AgentComparison
from src.utils.config import (
    get_agent_config,
    get_env_config,
    get_device,
    EnvironmentConfig
)
from src.utils.model_utils import (
    detect_model_type,
    detect_grid_size,
    validate_model_agent_compatibility,
    suggest_correct_agent_type,
    get_model_info
)


def create_agent(agent_type: str, state_dim: int, action_dim: int, device: torch.device, grid_size: Optional[int] = None):
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent ('dqn', 'reinforce', 'ppo', 'human')
        state_dim: Dimension of state space
        action_dim: Number of actions
        device: PyTorch device
        grid_size: Grid size (optional, for verification purposes)

    Returns:
        Initialized agent
    """
    if agent_type == 'human':
        return HumanAgent()

    config = get_agent_config(agent_type)

    if agent_type == 'dqn':
        return DQNAgent(state_dim, action_dim, device, config)
    elif agent_type == 'reinforce':
        return REINFORCEAgent(state_dim, action_dim, device, config)
    elif agent_type == 'ppo':
        return PPOAgent(state_dim, action_dim, device, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def adaptive_train_agent(args):
    """Train agent using adaptive grid size (grows during training)."""
    from src.training.adaptive_trainer import AdaptiveGridTrainer

    print(f"Adaptive training {args.agent.upper()} agent...")
    print(f"Grid size: 5x5 → automatically grows based on performance")
    print(f"Headless mode: {args.headless}")

    # Get device
    device = get_device()

    # Create adaptive trainer
    adaptive_trainer = AdaptiveGridTrainer(
        agent_type=args.agent,
        device=device,
        enable_rendering=not args.headless,
        save_dir=args.save_dir + "/adaptive"
    )

    # Run adaptive training
    summary = adaptive_trainer.train_adaptive(
        total_episodes=args.episodes,
        episodes_per_check=50,
        min_episodes_per_size=100
    )

    print(f"\\n🎯 Adaptive training completed!")
    print(f"Final grid size: {summary['final_grid_size']}x{summary['final_grid_size']}")
    print(f"Grid transitions: {len(summary['grid_transitions'])}")
    print(f"Final model: {summary['final_model_path']}")
    return summary


def curriculum_train_agent(args):
    """Train agent using curriculum learning (small to large grids)."""
    from src.training.curriculum_trainer import CurriculumTrainer

    print(f"Curriculum training {args.agent.upper()} agent...")
    print(f"Stages: 5x5 → 8x8 → 12x12 → 16x16 → 20x20")
    print(f"Headless mode: {args.headless}")

    # Get device
    device = get_device()

    # Create curriculum trainer
    curriculum_trainer = CurriculumTrainer(
        agent_type=args.agent,
        device=device,
        enable_rendering=not args.headless,
        save_dir=args.save_dir + "/curriculum"
    )

    # Run curriculum training
    summary = curriculum_trainer.train_curriculum()

    print(f"\\n🎓 Curriculum training completed!")
    print(f"Final model: {summary['final_model_path']}")
    return summary


def train_agent(args):
    """Train an agent."""
    print(f"Training {args.agent.upper()} agent...")
    print(f"Episodes: {args.episodes}")
    print(f"Headless mode: {args.headless}")

    # Get device
    device = get_device()

    # Create environment
    env = SnakeEnv(
        grid_size=args.grid_size,
        max_steps=EnvironmentConfig.MAX_STEPS,
        agent_type=args.agent
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
    if not args.load_model:
        print("Error: --load-model is required for testing")
        return

    if not os.path.exists(args.load_model):
        print(f"Error: Model file not found: {args.load_model}")
        return

    # Auto-detect agent type if requested
    if args.auto_detect:
        detected_type = detect_model_type(args.load_model)
        if detected_type is None:
            print(f"Error: Could not auto-detect model type from {args.load_model}")
            return
        args.agent = detected_type
        print(f"Auto-detected agent type: {detected_type}")

    # Auto-detect grid size from model
    detected_grid_size = detect_grid_size(args.load_model)
    if detected_grid_size is not None:
        if args.grid_size != detected_grid_size:
            print(f"Grid size mismatch detected!")
            print(f"  Command line grid size: {args.grid_size}")
            print(f"  Model grid size: {detected_grid_size}")
            print(f"  Using model grid size: {detected_grid_size}")
        args.grid_size = detected_grid_size
    else:
        print(f"Warning: Could not auto-detect grid size from model, using: {args.grid_size}")

    print(f"Testing {args.agent.upper()} agent...")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")

    # Detect model type and check compatibility
    detected_type = detect_model_type(args.load_model)
    if detected_type is None:
        print(f"Warning: Could not detect model type from {args.load_model}")
        print("Proceeding with specified agent type, but this may fail...")
    elif detected_type != args.agent:
        print(f"Error: Model type mismatch!")
        print(f"  Specified agent: {args.agent}")
        print(f"  Detected model type: {detected_type}")
        print(f"  Suggestion: Use '--agent {detected_type}' or '--auto-detect' instead")
        return

    print(f"Model type verified: {detected_type}")

    # Get device
    device = get_device()

    # Create environment
    env = SnakeEnv(
        grid_size=args.grid_size,
        max_steps=EnvironmentConfig.MAX_STEPS,
        agent_type=args.agent
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


def human_control(args):
    """Start human-controlled Snake game."""
    print("Starting human-controlled Snake game...")
    print("Use Arrow keys to control the snake")
    print("SPACE - Pause/Resume, +/- - Speed, S - Screenshot, ESC - Skip Episode, Q - Quit")

    # Create environment
    env = SnakeEnv(
        grid_size=args.grid_size,
        max_steps=EnvironmentConfig.MAX_STEPS,
        agent_type=args.agent
    )

    # Create human agent
    agent = HumanAgent()

    # Create trainer for visualization
    trainer = UniversalTrainer(
        agent=agent,
        env=env,
        agent_name="human",
        enable_rendering=True,
        save_dir=args.save_dir
    )

    # Run human control with special flag
    trainer.run_human_control(args.episodes)


def compare_agents(args):
    """Compare multiple agents using the AgentComparison framework."""
    print("Starting comprehensive agent comparison...")

    # Create comparison framework
    comparison = AgentComparison(
        env_config={
            'grid_size': args.grid_size,
            'max_steps': EnvironmentConfig.MAX_STEPS
        },
        results_dir=f"{args.save_dir}/comparison",
        models_dir=f"{args.save_dir}/comparison/models"
    )

    # All available agents
    agents_to_compare = ['dqn', 'reinforce', 'ppo']

    # Run comprehensive comparison
    results = comparison.run_comparison(
        agents_to_compare=agents_to_compare,
        episodes_per_agent=args.episodes,
        evaluation_episodes=args.test_episodes,
        enable_rendering=not args.headless,
        save_models=True
    )

    print(f"\n{'='*60}")
    print("COMPREHENSIVE COMPARISON COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {args.save_dir}/comparison/")
    print("Check the comparison plots and JSON results for detailed analysis.")


def inspect_model(args):
    """Inspect a model file and show detailed information."""
    if not args.load_model:
        print("Error: --load-model is required for model inspection")
        return

    print(f"Inspecting model: {args.load_model}")
    print("=" * 50)

    model_info = get_model_info(args.load_model)

    if not model_info['exists']:
        print(f"Error: Model file not found: {args.load_model}")
        return

    if 'error' in model_info:
        print(f"Error loading model: {model_info['error']}")
        return

    print(f"File size: {model_info['file_size_mb']:.2f} MB")
    print(f"Detected agent type: {model_info['agent_type'] or 'Unknown'}")
    print(f"Detected grid size: {model_info['grid_size'] or 'Unknown'}")
    print(f"Model keys: {len(model_info['keys'])}")

    print("\nModel structure:")
    for key in model_info['keys']:
        print(f"  - {key}")

    if model_info['agent_type']:
        print(f"\nUsage:")
        print(f"  uv run python src/main.py --mode test --agent {model_info['agent_type']} --load-model {args.load_model}")
        print(f"  # Or use auto-detection:")
        print(f"  uv run python src/main.py --mode test --auto-detect --load-model {args.load_model}")
        if model_info['grid_size']:
            print(f"  # Grid size will be auto-detected as: {model_info['grid_size']}x{model_info['grid_size']}")
    else:
        print(f"\nWarning: Could not determine agent type. This may not be a valid model file.")


def _add_arguments(parser):
    """Add all command line arguments to the parser."""
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test', 'human', 'curriculum', 'adaptive'], default='train',
                        help='Mode to run (train, test, human, curriculum, or adaptive)')

    # Agent selection
    parser.add_argument('--agent', choices=['dqn', 'reinforce', 'ppo'], default='dqn',
                        help='Type of agent to use')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train/test')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluation frequency during training')
    parser.add_argument('--grid-size', type=int, default=20,
                        help='Size of the game grid (default: 20x20)')

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

    # Auto-detection mode
    parser.add_argument('--auto-detect', action='store_true',
                        help='Automatically detect agent type from model file (test mode only)')

    # Model inspection
    parser.add_argument('--inspect-model', action='store_true',
                        help='Inspect model file and show detailed information')


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
        print("  uv run python src/main.py --mode test --agent dqn --load-model models/dqn_final.pth")
        print("")
        print("  # Test with auto-detection")
        print("  uv run python src/main.py --mode test --auto-detect --load-model models/reinforce_final.pth")
        print("")
        print("  # Inspect a model file")
        print("  uv run python src/main.py --inspect-model --load-model models/ppo_final.pth")
        print("")
        print("  # Run demo with random agent")
        print("  uv run python src/main.py --demo")
        print("")
        print("  # Play Snake manually with arrow keys (human control)")
        print("  uv run python src/main.py --mode human")
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

    if args.inspect_model:
        inspect_model(args)
        return

    # Run normal mode
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args)
    elif args.mode == 'human':
        human_control(args)
    elif args.mode == 'curriculum':
        curriculum_train_agent(args)
    elif args.mode == 'adaptive':
        adaptive_train_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
