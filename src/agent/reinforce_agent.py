import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque

from .base_agent import BaseAgent
from .neural_networks import PolicyNetwork, ValueNetwork


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE agent with baseline for variance reduction.
    Implements policy gradient methods using Monte Carlo returns.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize REINFORCE agent.

        Args:
            state_dim: Dimension of state space (grid_size for Snake)
            action_dim: Number of possible actions
            device: PyTorch device
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, device, config)

        # Extract config parameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.baseline_weight = config.get('baseline_weight', 0.5)
        self.hidden_size = config.get('hidden_size', 256)
        self.entropy_weight = config.get('entropy_weight', 0.01)

        # Initialize networks
        grid_size = int(np.sqrt(state_dim)) if state_dim == 400 else 20
        self.policy_net = PolicyNetwork(
            grid_size=grid_size,
            action_dim=action_dim,
            hidden_size=self.hidden_size
        ).to(device)

        # Baseline network (value function for variance reduction)
        self.baseline_net = ValueNetwork(
            grid_size=grid_size,
            hidden_size=self.hidden_size
        ).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        self.baseline_optimizer = optim.Adam(
            self.baseline_net.parameters(),
            lr=self.learning_rate
        )

        # Episode storage
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        self.episode_values: List[torch.Tensor] = []
        self.episode_entropies: List[torch.Tensor] = []

        # Training metrics
        self.policy_losses = deque(maxlen=100)
        self.baseline_losses = deque(maxlen=100)
        self.avg_returns = deque(maxlen=100)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using policy network.

        Args:
            state: Current state observation
            training: Whether in training mode

        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_net(state_tensor)

            if training:
                # Sample from probability distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                # Store log probability and entropy for training
                log_prob = action_dist.log_prob(action)
                entropy = action_dist.entropy()

                self.episode_log_probs.append(log_prob)
                self.episode_entropies.append(entropy)

                # Store baseline value
                baseline_value = self.baseline_net(state_tensor)
                self.episode_values.append(baseline_value.squeeze())

                return action.item()
            else:
                # Greedy action selection for evaluation
                return torch.argmax(action_probs, dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition reward.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.episode_rewards.append(reward)

    def update(self) -> Dict[str, float]:
        """
        Update policy and baseline networks using REINFORCE with baseline.

        Returns:
            Dictionary of training metrics
        """
        if not self.episode_rewards:
            return {}

        # Calculate discounted returns
        returns = self._calculate_returns()

        # Convert to tensors
        log_probs = torch.stack(self.episode_log_probs)
        values = torch.stack(self.episode_values)
        entropies = torch.stack(self.episode_entropies)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Calculate advantages (returns - baseline)
        advantages = returns_tensor - values.detach()

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages).mean()

        # Entropy bonus for exploration
        entropy_bonus = entropies.mean()
        total_policy_loss = policy_loss - self.entropy_weight * entropy_bonus

        # Baseline loss (MSE between predicted and actual returns)
        baseline_loss = nn.MSELoss()(values, returns_tensor)

        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        # Update baseline network
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), max_norm=0.5)
        self.baseline_optimizer.step()

        # Store metrics
        self.policy_losses.append(policy_loss.item())
        self.baseline_losses.append(baseline_loss.item())
        self.avg_returns.append(np.mean(returns))

        # Clear episode data
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.episode_entropies.clear()

        self.training_step += 1

        return {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy': entropy_bonus.item(),
            'avg_return': np.mean(returns),
            'episode_length': len(returns)
        }

    def _calculate_returns(self) -> List[float]:
        """
        Calculate discounted returns using Monte Carlo.

        Returns:
            List of discounted returns for each step
        """
        returns = []
        discounted_sum = 0

        # Calculate returns backwards (more numerically stable)
        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        return returns

    def reset_episode(self) -> None:
        """Reset episode-specific data."""
        super().reset_episode()
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.episode_entropies.clear()

    def save_model(self, filepath: str) -> None:
        """
        Save agent models.

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'baseline_net_state_dict': self.baseline_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load agent models.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.baseline_net.load_state_dict(checkpoint['baseline_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])

        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    def get_training_info(self) -> Dict[str, Any]:
        """
        Get current training information.

        Returns:
            Dictionary with training statistics
        """
        base_info = super().get_training_info()

        additional_info = {
            'avg_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0.0,
            'avg_baseline_loss': np.mean(self.baseline_losses) if self.baseline_losses else 0.0,
            'avg_return': np.mean(self.avg_returns) if self.avg_returns else 0.0,
        }

        return {**base_info, **additional_info}

    def set_training_mode(self, training: bool) -> None:
        """Set networks to training or evaluation mode."""
        self.policy_net.train(training)
        self.baseline_net.train(training)