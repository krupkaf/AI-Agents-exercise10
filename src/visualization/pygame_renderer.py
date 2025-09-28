"""PyGame renderer for Snake game visualization."""

import pygame
import numpy as np
from typing import Optional, Tuple, List
from src.environment.game_logic import SnakeGame
from src.visualization.colors import *


class PyGameRenderer:
    """Real-time PyGame renderer for Snake game."""

    def __init__(self, grid_size: int = 20, window_size: Tuple[int, int] = (800, 600)):
        """Initialize PyGame renderer.

        Args:
            grid_size: Size of the game grid
            window_size: Window size (width, height)
        """
        self.grid_size = grid_size
        self.window_size = window_size
        self.window_width, self.window_height = window_size

        # Calculate game area dimensions
        self.game_area_width = 500
        self.game_area_height = 500
        self.cell_size = self.game_area_width // grid_size

        # Panel dimensions
        self.panel_width = self.window_width - self.game_area_width
        self.panel_x = self.game_area_width

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Snake RL Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)

        self.paused = False
        self.fps = 10

    def render(self, game: SnakeGame, episode: int = 0, epsilon: float = 0.0,
               total_reward: float = 0.0, steps: int = 0) -> None:
        """Render the current game state.

        Args:
            game: SnakeGame instance
            episode: Current episode number
            epsilon: Current epsilon value
            total_reward: Total reward for current episode
            steps: Steps taken in current episode
        """
        self.screen.fill(BACKGROUND)

        # Draw game grid
        self._draw_grid()

        # Draw game elements
        self._draw_food(game.get_food_position())
        self._draw_snake(game.get_snake_body())

        # Draw info panel
        self._draw_info_panel(game, episode, epsilon, total_reward, steps)

        # Draw pause indicator if paused
        if self.paused:
            self._draw_pause_indicator()

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self) -> None:
        """Draw the game grid."""
        for x in range(0, self.game_area_width + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.game_area_height))

        for y in range(0, self.game_area_height + 1, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.game_area_width, y))

    def _draw_snake(self, snake_body: List[Tuple[int, int]]) -> None:
        """Draw the snake.

        Args:
            snake_body: List of snake body positions
        """
        for i, (x, y) in enumerate(snake_body):
            rect = pygame.Rect(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )

            # Head is different color
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect)

    def _draw_food(self, food_pos: Tuple[int, int]) -> None:
        """Draw the food.

        Args:
            food_pos: Food position (x, y)
        """
        x, y = food_pos
        rect = pygame.Rect(
            x * self.cell_size + 2,
            y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, FOOD_COLOR, rect)

    def _draw_info_panel(self, game: SnakeGame, episode: int, epsilon: float,
                        total_reward: float, steps: int) -> None:
        """Draw the information panel.

        Args:
            game: SnakeGame instance
            episode: Current episode number
            epsilon: Current epsilon value
            total_reward: Total reward for current episode
            steps: Steps taken in current episode
        """
        # Panel background
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, PANEL_BACKGROUND, panel_rect)

        # Title
        title_text = self.title_font.render("Snake RL", True, TEXT_COLOR)
        self.screen.blit(title_text, (self.panel_x + 10, 20))

        y_offset = 80
        line_height = 30

        # Game info
        info_lines = [
            f"Episode: {episode}",
            f"Score: {game.get_score()}",
            f"Steps: {steps}",
            f"Total Reward: {total_reward:.1f}",
            "",
            f"Epsilon: {epsilon:.3f}",
            f"FPS: {self.fps}",
            "",
            f"Grid Size: {self.grid_size}x{self.grid_size}",
            f"Snake Length: {len(game.get_snake_body())}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "UP/DOWN - Speed",
            "S - Screenshot",
            "Q - Quit"
        ]

        for i, line in enumerate(info_lines):
            if line:  # Skip empty lines
                text = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(text, (self.panel_x + 10, y_offset + i * line_height))

    def _draw_pause_indicator(self) -> None:
        """Draw pause indicator overlay."""
        overlay = pygame.Surface((self.game_area_width, self.game_area_height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        pause_text = self.title_font.render("PAUSED", True, WHITE)
        text_rect = pause_text.get_rect(center=(self.game_area_width // 2, self.game_area_height // 2))
        self.screen.blit(pause_text, text_rect)

    def handle_events(self) -> bool:
        """Handle PyGame events.

        Returns:
            False if quit event received, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.fps = self.fps + 5
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 5)
                elif event.key == pygame.K_s:
                    self._save_screenshot()

        return True

    def _save_screenshot(self) -> None:
        """Save a screenshot of the current game state."""
        import os
        import datetime

        # Create screenshots directory if it doesn't exist
        os.makedirs("results/screenshots", exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/screenshots/snake_{timestamp}.png"

        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved: {filename}")

    def is_paused(self) -> bool:
        """Check if renderer is paused."""
        return self.paused

    def close(self) -> None:
        """Close the renderer and clean up resources."""
        pygame.quit()