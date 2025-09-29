"""
Utility functions for model management and type detection.
"""

import torch
from typing import Optional, Dict, Any
import os


def detect_model_type(model_path: str) -> Optional[str]:
    """
    Detect the type of RL agent based on the model file structure.

    Args:
        model_path: Path to the model file

    Returns:
        Agent type ('dqn', 'reinforce', 'ppo') or None if detection fails
    """
    if not os.path.exists(model_path):
        return None

    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if not isinstance(checkpoint, dict):
            return None

        keys = set(checkpoint.keys())

        # DQN agent detection - has both q_network and target_network
        if 'q_network_state_dict' in keys and 'target_network_state_dict' in keys:
            return 'dqn'

        # REINFORCE agent detection - has policy_net and baseline_net
        elif 'policy_net_state_dict' in keys and 'baseline_net_state_dict' in keys:
            return 'reinforce'

        # PPO agent detection - has actor_critic network
        elif 'actor_critic_state_dict' in keys:
            return 'ppo'

        # Unknown format
        else:
            return None

    except Exception:
        # Failed to load or parse model
        return None


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a model file.

    Args:
        model_path: Path to the model file

    Returns:
        Dictionary with model information
    """
    info = {
        'path': model_path,
        'exists': os.path.exists(model_path),
        'agent_type': None,
        'grid_size': None,
        'keys': [],
        'file_size_mb': 0.0
    }

    if not info['exists']:
        return info

    try:
        # File size
        info['file_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)

        # Load and analyze checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            info['keys'] = list(checkpoint.keys())
            info['agent_type'] = detect_model_type(model_path)
            info['grid_size'] = detect_grid_size(model_path)

    except Exception as e:
        info['error'] = str(e)

    return info


def validate_model_agent_compatibility(model_path: str, agent_type: str) -> bool:
    """
    Check if a model file is compatible with the specified agent type.

    Args:
        model_path: Path to the model file
        agent_type: Type of agent ('dqn', 'reinforce', 'ppo')

    Returns:
        True if compatible, False otherwise
    """
    detected_type = detect_model_type(model_path)
    return detected_type == agent_type.lower()


def detect_grid_size(model_path: str) -> Optional[int]:
    """
    Detect the grid size used in the model based on network architecture.

    Args:
        model_path: Path to the model file

    Returns:
        Grid size (e.g., 7, 20) or None if detection fails
    """
    if not os.path.exists(model_path):
        return None

    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if not isinstance(checkpoint, dict):
            return None

        # Try to detect grid size from different agent types
        agent_type = detect_model_type(model_path)

        if agent_type == 'dqn':
            # For DQN, check q_network state dict
            if 'q_network_state_dict' in checkpoint:
                state_dict = checkpoint['q_network_state_dict']
                return _extract_grid_size_from_state_dict(state_dict, 'dqn')

        elif agent_type == 'reinforce':
            # For REINFORCE, check policy_net state dict
            if 'policy_net_state_dict' in checkpoint:
                state_dict = checkpoint['policy_net_state_dict']
                return _extract_grid_size_from_state_dict(state_dict, 'reinforce')

        elif agent_type == 'ppo':
            # For PPO, check actor_critic state dict
            if 'actor_critic_state_dict' in checkpoint:
                state_dict = checkpoint['actor_critic_state_dict']
                return _extract_grid_size_from_state_dict(state_dict, 'ppo')

        return None

    except Exception:
        return None


def _extract_grid_size_from_state_dict(state_dict: Dict[str, torch.Tensor], agent_type: str) -> Optional[int]:
    """
    Extract grid size from network state dictionary based on layer shapes.

    Args:
        state_dict: PyTorch state dictionary
        agent_type: Type of agent ('dqn', 'reinforce', 'ppo')

    Returns:
        Grid size or None if detection fails
    """
    try:
        # Different agents have different fully connected layer names
        if agent_type == 'dqn':
            fc1_key = 'fc1.weight'
        elif agent_type == 'reinforce':
            fc1_key = 'fc1.weight'
        elif agent_type == 'ppo':
            fc1_key = 'shared_fc.weight'
        else:
            return None

        if fc1_key not in state_dict:
            return None

        # Get the input size of the first fully connected layer
        fc1_weight = state_dict[fc1_key]
        conv_output_size = fc1_weight.shape[1]  # Input dimension to fc1

        # Calculate grid size based on the conv output size
        # For all agents: conv_output_size = final_channels * grid_size * grid_size

        if agent_type == 'dqn':
            # DQN: 64 channels after conv3
            final_channels = 64
        else:
            # REINFORCE and PPO: 32 channels after conv2
            final_channels = 32

        # grid_size^2 = conv_output_size / final_channels
        grid_size_squared = conv_output_size // final_channels
        grid_size = int(grid_size_squared ** 0.5)

        # Verify it's a perfect square
        if grid_size * grid_size == grid_size_squared:
            return grid_size
        else:
            return None

    except Exception:
        return None


def suggest_correct_agent_type(model_path: str) -> Optional[str]:
    """
    Suggest the correct agent type for a given model file.

    Args:
        model_path: Path to the model file

    Returns:
        Suggested agent type or None if detection fails
    """
    return detect_model_type(model_path)