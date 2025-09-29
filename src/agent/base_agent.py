from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple


class BaseAgent(ABC):
    """Abstract base class for all reinforcement learning agents."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            device: PyTorch device (cpu/cuda)
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.config = config

        # Training state
        self.training_step = 0
        self.episode_count = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store a transition for learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        Update the agent's policy/value function.

        Returns:
            Dictionary of training metrics (loss, etc.)
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the agent's model to file.

        Args:
            filepath: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load the agent's model from file.

        Args:
            filepath: Path to load the model from
        """
        pass

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get the current model state dictionary.

        Returns:
            Model state dictionary for transfer learning
        """
        pass

    @abstractmethod
    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from dictionary.

        Args:
            state_dict: Model state dictionary
        """
        pass

    def reset_episode(self) -> None:
        """Reset any episode-specific state."""
        self.episode_count += 1

    def get_training_info(self) -> Dict[str, Any]:
        """
        Get current training information.

        Returns:
            Dictionary with training statistics
        """
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }

    def set_training_mode(self, training: bool) -> None:
        """Set agent to training or evaluation mode."""
        pass  # Override in subclasses if needed