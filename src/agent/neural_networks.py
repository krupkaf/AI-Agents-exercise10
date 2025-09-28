import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for processing grid-based state representation.
    Uses convolutional layers for spatial feature extraction.
    """

    def __init__(self, grid_size: int = 20, action_dim: int = 4, hidden_size: int = 512):
        """
        Initialize DQN network.

        Args:
            grid_size: Size of the grid (20x20 for Snake)
            action_dim: Number of actions (4 for Snake: up, right, down, left)
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()

        self.grid_size = grid_size
        self.action_dim = action_dim

        # Convolutional layers for spatial feature extraction
        # Input: (batch_size, 1, grid_size, grid_size) - single channel grid
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size after convolutions
        # With padding=1 and stride=1, spatial dimensions are preserved
        conv_output_size = 64 * grid_size * grid_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
               or (batch_size, 1, grid_size, grid_size)

        Returns:
            Q-values for each action of shape (batch_size, action_dim)
        """
        # Ensure input has correct shape: (batch_size, 1, grid_size, grid_size)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation on final layer (Q-values can be negative)

        return x


class PolicyNetwork(nn.Module):
    """
    Policy network for REINFORCE and PPO agents.
    Outputs action probabilities for policy gradient methods.
    """

    def __init__(self, grid_size: int = 20, action_dim: int = 4, hidden_size: int = 256):
        """
        Initialize policy network.

        Args:
            grid_size: Size of the grid (20x20 for Snake)
            action_dim: Number of actions
            hidden_size: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()

        self.grid_size = grid_size
        self.action_dim = action_dim

        # Convolutional layers (lighter than DQN for faster training)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size
        conv_output_size = 32 * grid_size * grid_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)

        Returns:
            Action probabilities of shape (batch_size, action_dim)
        """
        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Policy output with softmax for action probabilities
        action_logits = self.policy_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs


class ValueNetwork(nn.Module):
    """
    Value network for PPO agent (critic).
    Estimates state values for advantage calculation.
    """

    def __init__(self, grid_size: int = 20, hidden_size: int = 256):
        """
        Initialize value network.

        Args:
            grid_size: Size of the grid
            hidden_size: Size of hidden layers
        """
        super(ValueNetwork, self).__init__()

        self.grid_size = grid_size

        # Convolutional layers (shared architecture with policy network)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size
        conv_output_size = 32 * grid_size * grid_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)

        Returns:
            State values of shape (batch_size, 1)
        """
        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Value output
        value = self.value_head(x)

        return value


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Shares convolutional features between policy and value heads.
    """

    def __init__(self, grid_size: int = 20, action_dim: int = 4, hidden_size: int = 256):
        """
        Initialize Actor-Critic network.

        Args:
            grid_size: Size of the grid
            action_dim: Number of actions
            hidden_size: Size of hidden layers
        """
        super(ActorCriticNetwork, self).__init__()

        self.grid_size = grid_size
        self.action_dim = action_dim

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size
        conv_output_size = 32 * grid_size * grid_size

        # Shared fully connected layer
        self.shared_fc = nn.Linear(conv_output_size, hidden_size)

        # Separate heads for actor and critic
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.

        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)

        Returns:
            Tuple of (action_probabilities, state_values)
        """
        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Shared fully connected layer
        shared_features = F.relu(self.shared_fc(x))
        shared_features = self.dropout(shared_features)

        # Actor branch (policy)
        actor_features = F.relu(self.actor_fc(shared_features))
        actor_features = self.dropout(actor_features)
        action_logits = self.policy_head(actor_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic branch (value)
        critic_features = F.relu(self.critic_fc(shared_features))
        critic_features = self.dropout(critic_features)
        state_value = self.value_head(critic_features)

        return action_probs, state_value