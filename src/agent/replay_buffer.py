import numpy as np
import torch
import random
from collections import deque
from typing import Tuple, List, NamedTuple


class Experience(NamedTuple):
    """Named tuple for storing experience transitions."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN agent.
    Stores transitions and provides random sampling for training.
    """

    def __init__(self, capacity: int, device: torch.device):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            device: PyTorch device for tensor operations
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} experiences, cannot sample {batch_size}")

        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)

        # Unpack and convert to tensors efficiently
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        # Convert to tensors in one operation
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'reward_std': 0.0
            }

        rewards = [exp.reward for exp in self.buffer]
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards)
        }


class EpisodeBuffer:
    """
    Buffer for storing complete episodes.
    Used by REINFORCE and PPO agents for full episode training.
    """

    def __init__(self, device: torch.device):
        """
        Initialize episode buffer.

        Args:
            device: PyTorch device for tensor operations
        """
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []  # For actor-critic methods

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float = None,
        value: float = None
    ) -> None:
        """
        Add a new transition to the current episode.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action (for policy gradient)
            value: State value (for actor-critic)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if log_prob is not None:
            self.log_probs.append(log_prob)

        if value is not None:
            self.values.append(value)

    def get_returns(self, gamma: float = 0.99) -> List[float]:
        """
        Calculate discounted returns for the episode.

        Args:
            gamma: Discount factor

        Returns:
            List of discounted returns for each timestep
        """
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def get_advantages(self, gamma: float = 0.99, lambda_gae: float = 0.95) -> List[float]:
        """
        Calculate GAE (Generalized Advantage Estimation) advantages.

        Args:
            gamma: Discount factor
            lambda_gae: GAE lambda parameter

        Returns:
            List of advantage values for each timestep
        """
        if not self.values:
            raise ValueError("Values not stored in buffer. Use get_returns() instead.")

        advantages = []
        advantage = 0

        # Calculate advantages using GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            advantage = delta + gamma * lambda_gae * advantage
            advantages.insert(0, advantage)

        return advantages

    def get_tensors(self, normalize_advantages: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Convert episode data to tensors.

        Args:
            normalize_advantages: Whether to normalize advantages

        Returns:
            Tuple of tensors (states, actions, returns, log_probs, advantages)
        """
        if not self.states:
            raise ValueError("Episode buffer is empty")

        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)

        # Calculate returns
        returns = self.get_returns()
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Prepare log_probs and advantages if available
        log_probs_tensor = None
        advantages_tensor = None

        if self.log_probs:
            log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)

        if self.values:
            advantages = self.get_advantages()
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)

            # Normalize advantages if requested
            if normalize_advantages and len(advantages) > 1:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                    advantages_tensor.std() + 1e-8
                )

        return states, actions, returns_tensor, log_probs_tensor, advantages_tensor

    def clear(self) -> None:
        """Clear the episode buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        """Return episode length."""
        return len(self.states)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.states) == 0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Advanced version of replay buffer that samples experiences based on their TD error.
    """

    def __init__(self, capacity: int, device: torch.device, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            device: PyTorch device for tensor operations
            alpha: Prioritization exponent (0 = uniform sampling, 1 = full prioritization)
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None
    ) -> None:
        """
        Add a new experience with priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority value (if None, uses max priority)
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

        if priority is None:
            priority = self.max_priority

        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch with importance sampling weights.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} experiences, cannot sample {batch_size}")

        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize for stability

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        # Convert to tensors efficiently
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        # Convert to tensors in one operation
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        return states, actions, rewards, next_states, dones, weights_tensor, indices

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)