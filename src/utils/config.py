"""Configuration settings for Snake RL project."""

import torch
from typing import Dict, Any


# Environment Configuration
class EnvironmentConfig:
    """Configuration for Snake environment."""

    GRID_SIZE = 20
    WINDOW_SIZE = 800
    WINDOW_HEIGHT = 600
    MAX_STEPS = 1000

    # Rendering
    FPS = 60
    CELL_SIZE = WINDOW_SIZE // GRID_SIZE

    # Colors (compatibility with existing colors.py)
    BACKGROUND_COLOR = (40, 40, 40)
    GRID_COLOR = (60, 60, 60)


# DQN Agent Configuration
class DQNConfig:
    """Configuration for DQN agent."""

    # Network architecture
    HIDDEN_SIZE = 512
    LEARNING_RATE = 0.001

    # Training parameters
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 10000
    MIN_REPLAY_SIZE = 1000
    TARGET_UPDATE_FREQ = 100

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.99

    # Optimizer
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0


# REINFORCE Agent Configuration
class REINFORCEConfig:
    """Configuration for REINFORCE agent."""

    # Network architecture
    HIDDEN_SIZE = 256
    LEARNING_RATE = 0.003

    # Training parameters
    GAMMA = 0.99

    # Baseline (for variance reduction)
    USE_BASELINE = True
    BASELINE_LEARNING_RATE = 0.001

    # Optimizer
    WEIGHT_DECAY = 1e-4


# PPO Agent Configuration
class PPOConfig:
    """Configuration for PPO agent."""

    # Network architecture
    HIDDEN_SIZE = 256
    LEARNING_RATE = 0.0003

    # Training parameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RATIO = 0.2
    VALUE_LOSS_COEFF = 0.5
    ENTROPY_COEFF = 0.01

    # PPO specific
    PPO_EPOCHS = 4
    MINI_BATCH_SIZE = 64
    MAX_GRAD_NORM = 0.5

    # Experience collection
    ROLLOUT_LENGTH = 128


# Training Configuration
class TrainingConfig:
    """Configuration for training process."""

    # Training parameters
    MAX_EPISODES = 1000
    SAVE_FREQUENCY = 100
    EVAL_FREQUENCY = 50
    EVAL_EPISODES = 10

    # Early stopping
    TARGET_SCORE = 50  # Target average score for Snake
    PATIENCE = 200  # Episodes without improvement

    # Logging
    LOG_FREQUENCY = 10
    TENSORBOARD_LOG_DIR = "runs"

    # Model saving
    MODEL_DIR = "models"
    BEST_MODEL_SUFFIX = "_best.pth"
    CHECKPOINT_SUFFIX = "_checkpoint.pth"


# Visualization Configuration
class VisualizationConfig:
    """Configuration for visualization and rendering."""

    # PyGame rendering
    ENABLE_RENDERING = True
    RENDER_SPEED = 60  # FPS during training
    EVAL_RENDER_SPEED = 30  # Slower FPS during evaluation

    # Statistics display
    STATS_UPDATE_FREQ = 1  # Episodes
    PLOT_UPDATE_FREQ = 10  # Episodes

    # Screenshot and recording
    SCREENSHOT_DIR = "results/screenshots"
    VIDEO_DIR = "results/videos"
    ENABLE_SCREENSHOTS = True
    SCREENSHOT_FREQ = 100  # Episodes

    # Stats tracking
    STATS_WINDOW_SIZE = 100  # For moving averages


def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """
    Get configuration dictionary for specified agent type.

    Args:
        agent_type: Type of agent ('dqn', 'reinforce', 'ppo')

    Returns:
        Configuration dictionary
    """
    configs = {
        'dqn': {
            'hidden_size': DQNConfig.HIDDEN_SIZE,
            'learning_rate': DQNConfig.LEARNING_RATE,
            'gamma': DQNConfig.GAMMA,
            'batch_size': DQNConfig.BATCH_SIZE,
            'replay_buffer_size': DQNConfig.REPLAY_BUFFER_SIZE,
            'min_replay_size': DQNConfig.MIN_REPLAY_SIZE,
            'target_update_freq': DQNConfig.TARGET_UPDATE_FREQ,
            'epsilon_start': DQNConfig.EPSILON_START,
            'epsilon_end': DQNConfig.EPSILON_END,
            'epsilon_decay': DQNConfig.EPSILON_DECAY,
            'weight_decay': DQNConfig.WEIGHT_DECAY,
            'grad_clip_norm': DQNConfig.GRAD_CLIP_NORM
        },
        'reinforce': {
            'hidden_size': REINFORCEConfig.HIDDEN_SIZE,
            'learning_rate': REINFORCEConfig.LEARNING_RATE,
            'gamma': REINFORCEConfig.GAMMA,
            'use_baseline': REINFORCEConfig.USE_BASELINE,
            'baseline_learning_rate': REINFORCEConfig.BASELINE_LEARNING_RATE,
            'weight_decay': REINFORCEConfig.WEIGHT_DECAY
        },
        'ppo': {
            'hidden_size': PPOConfig.HIDDEN_SIZE,
            'learning_rate': PPOConfig.LEARNING_RATE,
            'gamma': PPOConfig.GAMMA,
            'gae_lambda': PPOConfig.GAE_LAMBDA,
            'clip_ratio': PPOConfig.CLIP_RATIO,
            'value_loss_coeff': PPOConfig.VALUE_LOSS_COEFF,
            'entropy_coeff': PPOConfig.ENTROPY_COEFF,
            'ppo_epochs': PPOConfig.PPO_EPOCHS,
            'mini_batch_size': PPOConfig.MINI_BATCH_SIZE,
            'max_grad_norm': PPOConfig.MAX_GRAD_NORM,
            'rollout_length': PPOConfig.ROLLOUT_LENGTH
        }
    }

    if agent_type not in configs:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(configs.keys())}")

    return configs[agent_type]


def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Returns:
        PyTorch device (CUDA if available, otherwise CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


def get_env_config() -> Dict[str, Any]:
    """
    Get environment configuration dictionary.

    Returns:
        Environment configuration
    """
    return {
        'grid_size': EnvironmentConfig.GRID_SIZE,
        'window_size': EnvironmentConfig.WINDOW_SIZE,
        'window_height': EnvironmentConfig.WINDOW_HEIGHT,
        'max_steps': EnvironmentConfig.MAX_STEPS,
        'fps': EnvironmentConfig.FPS,
        'cell_size': EnvironmentConfig.CELL_SIZE
    }


def get_training_config() -> Dict[str, Any]:
    """
    Get training configuration dictionary.

    Returns:
        Training configuration
    """
    return {
        'max_episodes': TrainingConfig.MAX_EPISODES,
        'save_frequency': TrainingConfig.SAVE_FREQUENCY,
        'eval_frequency': TrainingConfig.EVAL_FREQUENCY,
        'eval_episodes': TrainingConfig.EVAL_EPISODES,
        'target_score': TrainingConfig.TARGET_SCORE,
        'patience': TrainingConfig.PATIENCE,
        'log_frequency': TrainingConfig.LOG_FREQUENCY,
        'model_dir': TrainingConfig.MODEL_DIR
    }


def get_visualization_config() -> Dict[str, Any]:
    """
    Get visualization configuration dictionary.

    Returns:
        Visualization configuration
    """
    return {
        'enable_rendering': VisualizationConfig.ENABLE_RENDERING,
        'render_speed': VisualizationConfig.RENDER_SPEED,
        'eval_render_speed': VisualizationConfig.EVAL_RENDER_SPEED,
        'stats_update_freq': VisualizationConfig.STATS_UPDATE_FREQ,
        'plot_update_freq': VisualizationConfig.PLOT_UPDATE_FREQ,
        'stats_window_size': VisualizationConfig.STATS_WINDOW_SIZE,
        'screenshot_dir': VisualizationConfig.SCREENSHOT_DIR,
        'enable_screenshots': VisualizationConfig.ENABLE_SCREENSHOTS,
        'screenshot_freq': VisualizationConfig.SCREENSHOT_FREQ
    }


# Default configurations for easy import
DEFAULT_CONFIGS = {
    'env': get_env_config(),
    'training': get_training_config(),
    'visualization': get_visualization_config(),
    'dqn': get_agent_config('dqn'),
    'reinforce': get_agent_config('reinforce'),
    'ppo': get_agent_config('ppo')
}