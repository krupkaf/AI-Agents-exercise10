"""Human-controlled agent for manual Snake game play."""

import pygame
import numpy as np
from typing import Optional, Dict, Any
from src.environment.constants import UP, RIGHT, DOWN, LEFT


class HumanAgent:
    """Human player agent controlled via keyboard input."""

    def __init__(self):
        """Initialize human agent."""
        self.last_action = RIGHT  # Default starting direction
        self.pending_action = None  # Store action until next step
        self.training_step = 0
        self.episode_count = 0

    def get_action(self, state, **kwargs) -> int:
        """Get action from keyboard input.

        Args:
            state: Current game state (not used for human control)
            **kwargs: Additional arguments (not used)

        Returns:
            Action as integer
        """
        # Check for keyboard input
        keys = pygame.key.get_pressed()

        # Determine action based on pressed keys
        # Use pending action if set, otherwise continue with last action
        if self.pending_action is not None:
            action = self.pending_action
            self.pending_action = None
        elif keys[pygame.K_UP]:
            action = UP
        elif keys[pygame.K_DOWN]:
            action = DOWN
        elif keys[pygame.K_LEFT]:
            action = LEFT
        elif keys[pygame.K_RIGHT]:
            action = RIGHT
        else:
            # No key pressed, continue with last action
            action = self.last_action

        self.last_action = action
        return action

    def handle_keydown_event(self, event) -> None:
        """Handle discrete keydown events for more responsive control.

        Args:
            event: pygame.KEYDOWN event
        """
        if event.key == pygame.K_UP:
            self.pending_action = UP
        elif event.key == pygame.K_DOWN:
            self.pending_action = DOWN
        elif event.key == pygame.K_LEFT:
            self.pending_action = LEFT
        elif event.key == pygame.K_RIGHT:
            self.pending_action = RIGHT

    def save_model(self, filepath: str) -> None:
        """Human agent doesn't need to save models."""
        pass

    def load_model(self, filepath: str) -> None:
        """Human agent doesn't need to load models."""
        pass

    def train_step(self, *args, **kwargs) -> dict:
        """Human agent doesn't train."""
        return {}

    def update_target_network(self) -> None:
        """Human agent doesn't have target network."""
        pass

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action based on keyboard input."""
        return self.get_action(state)

    def store_transition(self, state, action, reward, next_state, done, **kwargs) -> None:
        """Human agent doesn't store transitions."""
        pass

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """Human agent doesn't update."""
        return {}

    def set_training_mode(self, training: bool) -> None:
        """Set training mode (no effect for human agent)."""
        pass