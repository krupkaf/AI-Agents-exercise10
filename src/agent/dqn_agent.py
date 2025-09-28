import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import Dict, Any, Optional

from src.agent.base_agent import BaseAgent
from src.agent.neural_networks import DQNNetwork
from src.agent.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and target network.
    Implements the DQN algorithm for discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space (grid_size for Snake)
            action_dim: Number of possible actions
            device: PyTorch device
            config: Configuration dictionary with hyperparameters
        """
        super().__init__(state_dim, action_dim, device, config)

        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        self.min_replay_size = config.get('min_replay_size', 1000)
        self.hidden_size = config.get('hidden_size', 512)

        # Current epsilon for exploration
        self.epsilon = self.epsilon_start

        # Networks
        self.grid_size = int(np.sqrt(state_dim))  # Assume square grid
        self.q_network = DQNNetwork(
            grid_size=self.grid_size,
            action_dim=action_dim,
            hidden_size=self.hidden_size
        ).to(device)

        self.target_network = DQNNetwork(
            grid_size=self.grid_size,
            action_dim=action_dim,
            hidden_size=self.hidden_size
        ).to(device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, device)

        # Training metrics
        self.total_steps = 0
        self.loss_history = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation (2D grid or flattened)
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action for exploration
            return random.randrange(self.action_dim)

        # Convert state to proper grid shape and get Q-values
        state_tensor = self._preprocess_state(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        Update the Q-network using experience replay.

        Returns:
            Dictionary with training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'loss': 0.0, 'epsilon': self.epsilon}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Preprocess states
        states = self._preprocess_states_batch(states)
        next_states = self._preprocess_states_batch(next_states)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network periodically
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Note: Epsilon decay moved to reset_episode() to be per-episode instead of per-step

        # Store loss for tracking
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return {
            'loss': loss_value,
            'epsilon': self.epsilon,
            'q_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item(),
            'buffer_size': len(self.replay_buffer)
        }

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess single state for network input.

        Args:
            state: State array (2D grid or flattened)

        Returns:
            Preprocessed state tensor
        """
        # Ensure state is in 2D grid format
        if state.ndim == 1:
            # Flattened state - reshape to grid
            state_grid = state.reshape(self.grid_size, self.grid_size)
        else:
            # Already 2D grid
            state_grid = state

        # Convert to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state_grid).unsqueeze(0).to(self.device)

        return state_tensor

    def _preprocess_states_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Preprocess batch of states for network input.

        Args:
            states: Batch of flattened states

        Returns:
            Preprocessed states tensor
        """
        batch_size = states.shape[0]

        # Reshape to grid format using the stored grid_size
        states_grid = states.view(batch_size, self.grid_size, self.grid_size)

        return states_grid

    def save_model(self, filepath: str) -> None:
        """
        Save agent's model and training state.

        Args:
            filepath: Path to save the model
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load agent's model and training state.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])

    def set_training_mode(self, training: bool) -> None:
        """Set agent to training or evaluation mode."""
        if training:
            self.q_network.train()
            # Restore original epsilon if it was saved
            if hasattr(self, '_saved_epsilon'):
                self.epsilon = self._saved_epsilon
                delattr(self, '_saved_epsilon')
        else:
            self.q_network.eval()
            # Save current epsilon and use minimal epsilon during evaluation
            self._saved_epsilon = self.epsilon
            self.epsilon = 0.01

    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        base_info = super().get_training_info()

        dqn_info = {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer_size,
            'avg_loss_recent': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            'target_network_updates': self.total_steps // self.target_update_freq
        }

        return {**base_info, **dqn_info}

    def reset_episode(self) -> None:
        """Reset episode-specific state and decay epsilon."""
        super().reset_episode()

        # Decay epsilon per episode (not per step)
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a given state (for debugging/analysis).

        Args:
            state: State observation

        Returns:
            Q-values for all actions
        """
        state_tensor = self._preprocess_state(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.cpu().numpy().flatten()

    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        return self.replay_buffer.get_statistics()