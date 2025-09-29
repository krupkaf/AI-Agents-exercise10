import pytest
import numpy as np
from src.environment.snake_env import SnakeEnv
from src.environment.game_logic import SnakeGame
from src.environment.constants import UP, RIGHT, DOWN, LEFT


class TestSnakeGame:
    def test_initialization(self):
        game = SnakeGame(grid_size=10)
        assert game.grid_size == 10
        assert game.initial_length == 1
        assert len(game.snake) == 1
        assert game.snake[0] == (5, 5)
        assert game.direction == RIGHT
        assert game.score == 0
        assert not game.game_over

    def test_reset(self):
        game = SnakeGame(grid_size=10)
        game.score = 5
        game.steps = 10
        game.game_over = True

        state = game.reset(seed=42)

        assert game.score == 0
        assert game.steps == 0
        assert not game.game_over
        assert len(game.snake) == 1
        assert isinstance(state, np.ndarray)
        assert state.shape == (10, 10)

    def test_movement(self):
        game = SnakeGame(grid_size=10, seed=42)
        initial_head = game.snake[0]

        state, reward, done, info = game.step(RIGHT)

        new_head = game.snake[0]
        assert new_head[0] == initial_head[0] + 1
        assert new_head[1] == initial_head[1]
        assert not done
        # With new survival-first system: survival bonus (+0.1) + new cell bonus (+0.5) + distance penalty (-0.3) = 0.3
        assert reward == 0.3

    def test_food_eating(self):
        game = SnakeGame(grid_size=10, seed=42)

        game.food = (game.snake[0][0] + 1, game.snake[0][1])
        initial_length = len(game.snake)

        state, reward, done, info = game.step(RIGHT)

        assert len(game.snake) == initial_length + 1
        # With new progressive reward system: REWARD_FOOD_BASE + (score * FOOD_SCORE_MULTIPLIER) = 100.0 + (1 * 10.0) = 110.0
        assert reward == 110.0
        assert game.score == 1

    def test_wall_collision(self):
        game = SnakeGame(grid_size=10)
        game.snake = [(0, 5)]

        state, reward, done, info = game.step(LEFT)

        assert done
        assert reward == -200.0  # New REWARD_DEATH_BASE for quick death
        assert game.game_over

    def test_self_collision(self):
        game = SnakeGame(grid_size=10)
        game.snake = [(5, 5), (4, 5), (3, 5), (3, 4), (4, 4), (5, 4)]

        state, reward, done, info = game.step(LEFT)

        assert done
        assert reward == -200.0  # New REWARD_DEATH_BASE for quick death
        assert game.game_over

    def test_state_representation(self):
        game = SnakeGame(grid_size=5, seed=42)
        state = game.get_state()

        assert state.shape == (5, 5)
        assert state.dtype == np.int32

        head_x, head_y = game.snake[0]
        assert state[head_y, head_x] == 2

        food_x, food_y = game.food
        assert state[food_y, food_x] == 3

    def test_custom_initial_length(self):
        game = SnakeGame(grid_size=10, initial_length=3)
        assert len(game.snake) == 3
        assert game.snake[0] == (5, 5)  # Head at center
        assert game.snake[1] == (4, 5)  # Body leftward
        assert game.snake[2] == (3, 5)  # Tail leftward

    def test_initial_length_too_large(self):
        with pytest.raises(ValueError):
            SnakeGame(grid_size=5, initial_length=25)


class TestSnakeEnv:
    def test_initialization(self):
        env = SnakeEnv(grid_size=20)

        assert env.grid_size == 20
        assert env.action_space.n == 4
        assert env.observation_space.shape == (20, 20)

    def test_reset(self):
        env = SnakeEnv(grid_size=10)

        observation, info = env.reset(seed=42)

        assert observation.shape == (10, 10)
        assert "score" in info
        assert "steps" in info
        assert "snake_length" in info
        assert info["score"] == 0
        assert info["snake_length"] == 1

    def test_step(self):
        env = SnakeEnv(grid_size=10)
        env.reset(seed=42)

        observation, reward, terminated, truncated, info = env.step(1)

        assert observation.shape == (10, 10)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "score" in info
        assert "steps" in info

    def test_invalid_action(self):
        env = SnakeEnv(grid_size=10)
        env.reset()

        with pytest.raises(ValueError):
            env.step(5)

    def test_max_steps_truncation(self):
        env = SnakeEnv(grid_size=10, max_steps=5)
        env.reset(seed=42)

        for _ in range(5):
            observation, reward, terminated, truncated, info = env.step(1)

        assert truncated
        assert info["truncated"]

    def test_action_meanings(self):
        env = SnakeEnv()
        meanings = env.get_action_meanings()

        expected = ["UP", "RIGHT", "DOWN", "LEFT"]
        assert meanings == expected

    def test_state_info(self):
        env = SnakeEnv(grid_size=10)
        env.reset(seed=42)

        state_info = env.get_state_info()

        required_keys = [
            "head_position", "food_position", "snake_body",
            "direction", "score", "steps", "game_over"
        ]

        for key in required_keys:
            assert key in state_info

    def test_render_rgb_array(self):
        env = SnakeEnv(grid_size=5, render_mode="rgb_array")
        env.reset(seed=42)

        rgb_array = env.render()

        assert rgb_array is not None
        assert rgb_array.shape == (5, 5, 3)
        assert rgb_array.dtype == np.uint8

    def test_oscillation_prevention_after_food(self):
        """Test that oscillation is prevented after food consumption."""
        # Create game with controlled setup
        game = SnakeGame(grid_size=5, agent_type='dqn')
        game.snake = [(1, 1)]
        game.food = (1, 0)

        # Eat food - snake grows to length 2
        state, reward, done, info = game.step(UP)
        assert len(game.snake) == 2
        assert not done

        # Try to oscillate - should detect collision immediately
        state, reward, done, info = game.step(DOWN)
        assert done, "Snake should collide when trying to move to occupied body position"

    def test_normal_movement_after_food(self):
        """Test that normal movement still works after food consumption."""
        game = SnakeGame(grid_size=5, agent_type='dqn')
        game.snake = [(2, 2)]
        game.food = (2, 1)

        # Eat food
        state, reward, done, info = game.step(UP)
        assert len(game.snake) == 2
        assert not done

        # Normal movements should work fine
        normal_moves = [LEFT, DOWN, RIGHT]
        for action in normal_moves:
            state, reward, done, info = game.step(action)
            assert not done, f"Normal movement {action} should not cause collision"


if __name__ == "__main__":
    pytest.main([__file__])