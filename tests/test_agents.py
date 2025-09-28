import pytest
import torch
import numpy as np
from src.agent.dqn_agent import DQNAgent
from src.agent.reinforce_agent import REINFORCEAgent
from src.agent.ppo_agent import PPOAgent
from src.utils.config import get_agent_config


class TestAgents:
    """Test suite for all agent implementations."""

    @pytest.fixture
    def agent_config(self):
        """Common setup for agent tests."""
        return {
            'state_dim': 400,  # 20x20 grid flattened
            'action_dim': 4,   # UP, RIGHT, DOWN, LEFT
            'device': torch.device('cpu')
        }

    def test_dqn_agent_creation(self, agent_config):
        """Test DQN agent creation and basic functionality."""
        config = get_agent_config('dqn')
        agent = DQNAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            device=agent_config['device'],
            config=config
        )

        assert agent.state_dim == 400
        assert agent.action_dim == 4
        assert agent.device == torch.device('cpu')

        # Test action selection
        state = np.random.random((20, 20))
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 4

        # Test evaluation mode
        action_eval = agent.select_action(state, training=False)
        assert isinstance(action_eval, int)
        assert 0 <= action_eval < 4

    def test_reinforce_agent_creation(self, agent_config):
        """Test REINFORCE agent creation and basic functionality."""
        config = get_agent_config('reinforce')
        agent = REINFORCEAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            device=agent_config['device'],
            config=config
        )

        assert agent.state_dim == 400
        assert agent.action_dim == 4
        assert agent.device == torch.device('cpu')

        # Test action selection
        state = np.random.random((20, 20))
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 4

        # Test evaluation mode
        action_eval = agent.select_action(state, training=False)
        assert isinstance(action_eval, int)
        assert 0 <= action_eval < 4

    def test_ppo_agent_creation(self, agent_config):
        """Test PPO agent creation and basic functionality."""
        config = get_agent_config('ppo')
        agent = PPOAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            device=agent_config['device'],
            config=config
        )

        assert agent.state_dim == 400
        assert agent.action_dim == 4
        assert agent.device == torch.device('cpu')

        # Test action selection
        state = np.random.random((20, 20))
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 4

        # Test evaluation mode
        action_eval = agent.select_action(state, training=False)
        assert isinstance(action_eval, int)
        assert 0 <= action_eval < 4

    def test_agent_transition_storage(self, agent_config):
        """Test that agents can store transitions."""
        agents = [
            ('dqn', DQNAgent),
            ('reinforce', REINFORCEAgent),
            ('ppo', PPOAgent)
        ]

        for agent_name, agent_class in agents:
            config = get_agent_config(agent_name)
            agent = agent_class(
                state_dim=agent_config['state_dim'],
                action_dim=agent_config['action_dim'],
                device=agent_config['device'],
                config=config
            )

            # Test storing a transition
            state = np.random.random(400)
            action = 1
            reward = 10.0
            next_state = np.random.random(400)
            done = False

            # This should not raise an exception
            agent.store_transition(state, action, reward, next_state, done)

    def test_agent_episode_reset(self, agent_config):
        """Test that agents properly reset episode data."""
        agents = [
            ('dqn', DQNAgent),
            ('reinforce', REINFORCEAgent),
            ('ppo', PPOAgent)
        ]

        for agent_name, agent_class in agents:
            config = get_agent_config(agent_name)
            agent = agent_class(
                state_dim=agent_config['state_dim'],
                action_dim=agent_config['action_dim'],
                device=agent_config['device'],
                config=config
            )

            initial_episode_count = agent.episode_count

            # Reset episode
            agent.reset_episode()

            # Episode count should increase
            assert agent.episode_count == initial_episode_count + 1

    def test_agent_training_info(self, agent_config):
        """Test that agents return training information."""
        agents = [
            ('dqn', DQNAgent),
            ('reinforce', REINFORCEAgent),
            ('ppo', PPOAgent)
        ]

        for agent_name, agent_class in agents:
            config = get_agent_config(agent_name)
            agent = agent_class(
                state_dim=agent_config['state_dim'],
                action_dim=agent_config['action_dim'],
                device=agent_config['device'],
                config=config
            )

            info = agent.get_training_info()

            # All agents should have these basic fields
            assert 'training_step' in info
            assert 'episode_count' in info
            assert isinstance(info['training_step'], int)
            assert isinstance(info['episode_count'], int)

    def test_agent_training_mode(self, agent_config):
        """Test that agents can switch between training and evaluation modes."""
        agents = [
            ('dqn', DQNAgent),
            ('reinforce', REINFORCEAgent),
            ('ppo', PPOAgent)
        ]

        for agent_name, agent_class in agents:
            config = get_agent_config(agent_name)
            agent = agent_class(
                state_dim=agent_config['state_dim'],
                action_dim=agent_config['action_dim'],
                device=agent_config['device'],
                config=config
            )

            # This should not raise an exception
            agent.set_training_mode(True)
            agent.set_training_mode(False)

    def test_agent_update_reinforce(self, agent_config):
        """Test REINFORCE agent update after episode completion."""
        config = get_agent_config('reinforce')
        agent = REINFORCEAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            device=agent_config['device'],
            config=config
        )

        # Simulate a complete episode with proper action-reward pairing
        state = np.random.random((20, 20))

        # Take actions and store rewards (action selection and reward storage must match)
        for step in range(5):
            action = agent.select_action(state, training=True)
            reward = -0.1 if step < 4 else 10.0
            done = step == 4
            agent.store_transition(state.flatten(), action, reward, state.flatten(), done)

        # Update should work and return metrics
        metrics = agent.update()

        if metrics:  # Only check if update was performed
            assert isinstance(metrics, dict)
            assert 'policy_loss' in metrics
            assert 'baseline_loss' in metrics

    def test_agent_update_ppo(self, agent_config):
        """Test PPO agent update after sufficient experience."""
        config = get_agent_config('ppo')
        config['buffer_size'] = 50  # Small buffer for testing
        agent = PPOAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            device=agent_config['device'],
            config=config
        )

        # Simulate multiple short episodes to fill buffer
        state = np.random.random((20, 20))

        for episode in range(3):
            for step in range(10):
                action = agent.select_action(state, training=True)
                reward = -0.1 if step < 9 else 10.0
                done = step == 9
                agent.store_transition(state.flatten(), action, reward, state.flatten(), done)

        # Update should work if enough experience collected
        metrics = agent.update()

        if metrics:  # Only check if update was performed
            assert isinstance(metrics, dict)
            # PPO specific metrics
            assert 'policy_loss' in metrics
            assert 'value_loss' in metrics