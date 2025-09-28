import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from .game_logic import SnakeGame


class SnakeEnv(gym.Env):
    """Gymnasium environment wrapper for Snake game.

    Provides standardized RL interface for Snake game with configurable
    grid size, step limits, and rendering modes.
    """
    # Metadata for Gymnasium framework - describes environment capabilities
    metadata = {
        "render_modes": ["rgb_array"],  # List of supported rendering modes
        "render_fps": 10               # Suggested FPS for human-readable rendering
    }

    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        """Initialize Snake environment.

        Args:
            grid_size: Size of square game grid (default: 20x20)
            max_steps: Maximum steps before truncation (default: 1000)
            render_mode: Rendering mode string as required by Gymnasium standard:
                - "rgb_array": Returns RGB image array when render() is called
                - None: No rendering, render() returns None (headless mode)
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Define action space: 4 discrete actions (UP=0, RIGHT=1, DOWN=2, LEFT=3)
        self.action_space = spaces.Discrete(4)

        # Define observation space: 2D grid with encoded game state
        # low=0, high=3: Each grid cell can contain values 0, 1, 2, or 3
        # Values: 0=empty, 1=snake_body, 2=snake_head, 3=food
        # Note: No static obstacles (4) in current implementation - only field boundaries
        self.observation_space = spaces.Box(
            low=0,      # Minimum value in any grid cell
            high=3,     # Maximum value in any grid cell
            shape=(grid_size, grid_size),
            dtype=np.int32  # Integer values for categorical data
        )

        self.game = SnakeGame(grid_size=grid_size)
        self.current_step = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducible resets
            options: Additional reset options (unused)

        Returns:
            observation: Initial game state as grid array
            info: Dictionary with game statistics
        """
        super().reset(seed=seed)

        self.game.reset(seed)
        self.current_step = 0

        observation = self.game.get_state()
        info = {
            "score": self.game.get_score(),
            "steps": self.game.get_steps(),
            "snake_length": len(self.game.get_snake_body()),
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            observation: New game state as grid array
            reward: Reward for this step (see constants.py for reward values)
            terminated: True if game ended (collision/death)
            truncated: True if max steps reached
            info: Dictionary with game statistics
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        observation, reward, terminated, info = self.game.step(action)
        self.current_step += 1

        truncated = self.current_step >= self.max_steps

        info.update({
            "steps": self.game.get_steps(),
            "snake_length": len(self.game.get_snake_body()),
            "terminated": terminated,
            "truncated": truncated,
        })

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render current game state based on render_mode.

        The render_mode determines what this method returns:
        - "rgb_array": Returns (grid_size, grid_size, 3) RGB image array
        - None: Returns None (for headless training without visualization)

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_rgb_array(self) -> np.ndarray:
        """Convert game state to RGB array for rendering.

        Returns:
            RGB array (grid_size, grid_size, 3) with colored game elements
        """
        rgb_array = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        state = self.game.get_state()

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if state[y, x] == 1.0:
                    rgb_array[y, x] = [0, 255, 0]
                elif state[y, x] == 2.0:
                    rgb_array[y, x] = [0, 200, 0]
                elif state[y, x] == 3.0:
                    rgb_array[y, x] = [255, 0, 0]
                else:
                    rgb_array[y, x] = [0, 0, 0]

        return rgb_array

    def close(self):
        """Clean up resources when environment is closed."""
        pass

    def get_action_meanings(self) -> list:
        """Get human-readable action names.

        Returns:
            List of action names corresponding to action indices
        """
        return ["UP", "RIGHT", "DOWN", "LEFT"]

    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed game state information.

        Returns:
            Dictionary with complete game state details
        """
        return {
            "head_position": self.game.get_head_position(),
            "food_position": self.game.get_food_position(),
            "snake_body": self.game.get_snake_body(),
            "direction": self.game.get_direction(),
            "score": self.game.get_score(),
            "steps": self.game.get_steps(),
            "game_over": self.game.is_game_over(),
        }