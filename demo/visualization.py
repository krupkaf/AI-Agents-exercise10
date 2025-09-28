"""Demo script for PyGame visualization with random agent."""

import time
import random
import os
from src.environment.game_logic import SnakeGame
from src.visualization.pygame_renderer import PyGameRenderer

# Configuration variables
GRID_SIZE = int(os.getenv('SNAKE_GRID_SIZE', 20))
WINDOW_WIDTH = int(os.getenv('SNAKE_WINDOW_WIDTH', 800))
WINDOW_HEIGHT = int(os.getenv('SNAKE_WINDOW_HEIGHT', 600))
MAX_EPISODES = int(os.getenv('SNAKE_MAX_EPISODES', 5))
SEED = int(os.getenv('SNAKE_SEED', 42))
INITIAL_FPS = int(os.getenv('SNAKE_FPS', 3))
INITIAL_LENGTH = int(os.getenv('SNAKE_INITIAL_LENGTH', 10))


def demo_pygame_visualization():
    """Demo PyGame visualization with random agent."""
    print(f"Configuration: Grid={GRID_SIZE}x{GRID_SIZE}, Window={WINDOW_WIDTH}x{WINDOW_HEIGHT}, FPS={INITIAL_FPS}, Length={INITIAL_LENGTH}")

    game = SnakeGame(grid_size=GRID_SIZE, initial_length=INITIAL_LENGTH, seed=SEED)
    renderer = PyGameRenderer(grid_size=GRID_SIZE, window_size=(WINDOW_WIDTH, WINDOW_HEIGHT))

    # Set initial FPS
    renderer.fps = INITIAL_FPS

    episode = 1
    epsilon = 1.0
    total_reward = 0.0
    steps = 0

    print("Testing PyGame visualization...")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  UP/DOWN - Adjust speed")
    print("  S - Screenshot")
    print("  Q - Quit")

    running = True
    game_running = True

    while running:
        # Handle PyGame events
        running = renderer.handle_events()

        if not renderer.is_paused() and game_running:
            # Random action for testing
            action = random.choice([0, 1, 2, 3])  # UP, RIGHT, DOWN, LEFT
            state, reward, done, info = game.step(action)

            total_reward += reward
            steps += 1

            if done:
                print(f"Episode {episode} finished - Score: {game.get_score()}, "
                      f"Steps: {steps}, Reward: {total_reward:.1f}")

                # Reset for next episode
                episode += 1
                epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon
                total_reward = 0.0
                steps = 0
                game.reset()

                # Stop after configured number of episodes
                if episode > MAX_EPISODES:
                    game_running = False

        # Render current state
        renderer.render(game, episode, epsilon, total_reward, steps)

        if not game_running:
            # Wait a bit before closing
            time.sleep(2)
            break

    renderer.close()
    print("Visualization test completed!")


if __name__ == "__main__":
    demo_pygame_visualization()