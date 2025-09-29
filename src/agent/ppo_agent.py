import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque

from .base_agent import BaseAgent
from .neural_networks import ActorCriticNetwork


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent with actor-critic architecture.
    Implements clipped surrogate objective with GAE for advantage estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            device: PyTorch device
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, device, config)

        # Extract config parameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_weight = config.get('entropy_weight', 0.01)
        self.value_weight = config.get('value_weight', 0.5)
        self.hidden_size = config.get('hidden_size', 256)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # PPO-specific parameters
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 2048)

        # Initialize network
        grid_size = int(np.sqrt(state_dim))
        self.actor_critic = ActorCriticNetwork(
            grid_size=grid_size,
            action_dim=action_dim,
            hidden_size=self.hidden_size
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate
        )

        # Experience buffer
        self.states_buffer: List[np.ndarray] = []
        self.actions_buffer: List[int] = []
        self.rewards_buffer: List[float] = []
        self.log_probs_buffer: List[torch.Tensor] = []
        self.values_buffer: List[torch.Tensor] = []
        self.dones_buffer: List[bool] = []

        # Episode tracking
        self.episode_states: List[np.ndarray] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_values: List[torch.Tensor] = []
        self.episode_dones: List[bool] = []

        # Training metrics
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        self.clip_fractions = deque(maxlen=100)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using actor-critic network.

        Args:
            state: Current state observation
            training: Whether in training mode

        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad() if not training else torch.enable_grad():
            action_probs, state_value = self.actor_critic(state_tensor)

            if training:
                # Sample from probability distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Store for episode
                self.episode_states.append(state)
                self.episode_actions.append(action.item())
                self.episode_log_probs.append(log_prob)
                self.episode_values.append(state_value.squeeze())

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
        Store transition reward and done flag.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)

        # If episode is done, add to buffer
        if done:
            self._add_episode_to_buffer()

    def _add_episode_to_buffer(self) -> None:
        """Add completed episode to experience buffer."""
        self.states_buffer.extend(self.episode_states)
        self.actions_buffer.extend(self.episode_actions)
        self.rewards_buffer.extend(self.episode_rewards)
        self.log_probs_buffer.extend(self.episode_log_probs)
        self.values_buffer.extend(self.episode_values)
        self.dones_buffer.extend(self.episode_dones)

        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_values.clear()
        self.episode_dones.clear()

        # Maintain buffer size
        if len(self.states_buffer) > self.buffer_size:
            excess = len(self.states_buffer) - self.buffer_size
            self.states_buffer = self.states_buffer[excess:]
            self.actions_buffer = self.actions_buffer[excess:]
            self.rewards_buffer = self.rewards_buffer[excess:]
            self.log_probs_buffer = self.log_probs_buffer[excess:]
            self.values_buffer = self.values_buffer[excess:]
            self.dones_buffer = self.dones_buffer[excess:]

    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.

        Returns:
            Dictionary of training metrics
        """
        if len(self.states_buffer) < self.batch_size:
            return {}

        # Calculate advantages and returns using GAE
        advantages, returns = self._calculate_gae()

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states_buffer)).to(self.device)
        actions = torch.LongTensor(self.actions_buffer).to(self.device)
        old_log_probs = torch.stack(self.log_probs_buffer).detach().to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clip_fraction = 0

        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass
                action_probs, state_values = self.actor_critic(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)

                # Calculate new log probabilities and entropy
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()

                # Calculate probability ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)

                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_weight * value_loss -
                    self.entropy_weight * entropy
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()

                # Calculate clip fraction
                clip_fraction = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean()
                total_clip_fraction += clip_fraction.item()

        # Average metrics over all updates
        num_updates = self.ppo_epochs * len(range(0, len(states), self.batch_size))
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates

        # Store metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.clip_fractions.append(avg_clip_fraction)

        # Clear buffer after update
        self._clear_buffer()

        self.training_step += 1

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy_loss,
            'clip_fraction': avg_clip_fraction,
            'buffer_size': len(states)
        }

    def _calculate_gae(self) -> Tuple[List[float], List[float]]:
        """
        Calculate Generalized Advantage Estimation (GAE).

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        # Convert values to numpy for easier computation
        values = [v.cpu().item() for v in self.values_buffer]

        # Add final value estimate for bootstrap
        if self.states_buffer:
            last_state = torch.FloatTensor(self.states_buffer[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value = self.actor_critic(last_state)
                values.append(last_value.cpu().item() if not self.dones_buffer[-1] else 0.0)

        # Calculate GAE
        gae = 0
        for i in reversed(range(len(self.rewards_buffer))):
            if i == len(self.rewards_buffer) - 1:
                next_value = values[i + 1]
            else:
                next_value = values[i + 1]

            delta = self.rewards_buffer[i] + self.gamma * next_value * (1 - self.dones_buffer[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones_buffer[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return advantages, returns

    def _clear_buffer(self) -> None:
        """Clear experience buffer."""
        self.states_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()
        self.log_probs_buffer.clear()
        self.values_buffer.clear()
        self.dones_buffer.clear()

    def reset_episode(self) -> None:
        """Reset episode-specific data."""
        super().reset_episode()
        # Episode data is managed through store_transition and _add_episode_to_buffer

    def save_model(self, filepath: str) -> None:
        """
        Save agent model.

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load agent model.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
            'avg_value_loss': np.mean(self.value_losses) if self.value_losses else 0.0,
            'avg_entropy': np.mean(self.entropy_losses) if self.entropy_losses else 0.0,
            'avg_clip_fraction': np.mean(self.clip_fractions) if self.clip_fractions else 0.0,
            'buffer_size': len(self.states_buffer)
        }

        return {**base_info, **additional_info}

    def set_training_mode(self, training: bool) -> None:
        """Set network to training or evaluation mode."""
        self.actor_critic.train(training)

    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state dictionary for transfer learning."""
        return self.actor_critic.state_dict()

    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """Load model state from dictionary for transfer learning."""
        self.actor_critic.load_state_dict(state_dict, strict=False)