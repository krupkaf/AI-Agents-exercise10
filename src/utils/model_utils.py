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


def suggest_correct_agent_type(model_path: str) -> Optional[str]:
    """
    Suggest the correct agent type for a given model file.

    Args:
        model_path: Path to the model file

    Returns:
        Suggested agent type or None if detection fails
    """
    return detect_model_type(model_path)