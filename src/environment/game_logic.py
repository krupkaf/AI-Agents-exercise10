import random
import numpy as np
from typing import List, Tuple, Optional

from .constants import (
    UP, RIGHT, DOWN, LEFT, EMPTY, SNAKE_BODY, SNAKE_HEAD, FOOD,
    REWARD_STEP, REWARD_SURVIVAL, REWARD_SURVIVAL_PG, REWARD_NEW_CELL, RECENT_VISIT_WINDOW,
    REWARD_REVISIT, REWARD_OSCILLATE, REWARD_CLOSER, REWARD_FARTHER,
    REWARD_DEATH_BASE, REWARD_FOOD_BASE, SURVIVAL_THRESHOLD, DEATH_REDUCTION_RATE,
    FOOD_SCORE_MULTIPLIER, MIN_DEATH_PENALTY_RATIO
)


class SnakeGame:
    """Core Snake game logic implementation.

    Handles game state, movement, collision detection, and scoring.
    Uses integer grid coordinates and direction constants.
    """

    def __init__(self, grid_size: int = 20, initial_length: int = 1, seed: Optional[int] = None, agent_type: str = "dqn"):
        """Initialize Snake game.

        Args:
            grid_size: Size of square game grid (default: 20x20)
            initial_length: Initial length of snake (default: 1)
            seed: Optional random seed for reproducible games
            agent_type: Type of agent ("dqn", "ppo", "reinforce") for reward tuning
        """
        self.grid_size = grid_size
        self.initial_length = initial_length
        self.agent_type = agent_type
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset game to initial state.

        Args:
            seed: Optional random seed for reproducible resets

        Returns:
            Initial game state as grid array
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.snake = self._create_initial_snake()
        self.direction = RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.game_over = False

        # Track visited positions for anti-oscillation rewards
        self.visited_positions = set()  # All positions ever visited
        self.position_history = []      # Last few positions for oscillation detection
        self.recent_positions = []      # Recent positions for exploration bonus tracking

        # Track distance to food for guidance rewards
        self.previous_distance_to_food = self._manhattan_distance(self.snake[0], self.food)

        return self.get_state()

    def _create_initial_snake(self) -> List[Tuple[int, int]]:
        """Create initial snake with specified length.

        Creates snake starting from center, extending leftward horizontally.
        Validates that snake fits within grid.

        Returns:
            List of (x, y) positions from head to tail

        Raises:
            ValueError: If initial_length is too large for grid_size
        """
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        # Validate snake fits in grid horizontally
        if self.initial_length > self.grid_size:
            raise ValueError(f"Initial length {self.initial_length} too large for grid size {self.grid_size}")

        # Create snake extending leftward from center
        snake = []
        for i in range(self.initial_length):
            x = center_x - i
            if x < 0:
                raise ValueError(f"Initial length {self.initial_length} too large for horizontal placement")
            snake.append((x, center_y))

        return snake

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step with given action.

        Args:
            action: Direction to move (UP=0, RIGHT=1, DOWN=2, LEFT=3)

        Returns:
            observation: Current game state as grid array
            reward: Reward for this step (see constants.py for reward values)
            done: True if game ended (collision occurred)
            info: Dictionary with game statistics
        """
        if self.game_over:
            return self.get_state(), 0, True, {"score": self.score}

        self.direction = action
        self.steps += 1

        head_x, head_y = self.snake[0]

        if self.direction == UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == LEFT:
            new_head = (head_x - 1, head_y)

        # Start with survival bonus (algorithm-specific)
        if self.agent_type in ["ppo", "reinforce"]:
            reward = REWARD_SURVIVAL_PG  # Higher bonus for policy gradient methods
        else:
            reward = REWARD_SURVIVAL     # Standard bonus for DQN

        if self._is_collision(new_head):
            self.game_over = True
            reward = self._calculate_death_penalty()
            return self.get_state(), reward, True, {"score": self.score}

        # Bonus for exploring recently unvisited cells (prevents oscillation abuse)
        if new_head not in self.recent_positions:
            reward += REWARD_NEW_CELL

        # Apply position-based penalties (but lighter)
        position_penalty = self._calculate_position_penalty(new_head)
        reward += position_penalty

        # Apply distance-based reward for food guidance (stronger)
        distance_reward = self._calculate_distance_reward(new_head)
        reward += distance_reward

        self.snake.insert(0, new_head)

        # Update position tracking
        self._update_position_tracking(new_head)

        if new_head == self.food:
            self.score += 1
            reward = self._calculate_food_reward()
            self.food = self._spawn_food()
            # Update distance tracking for new food position
            self.previous_distance_to_food = self._manhattan_distance(self.snake[0], self.food)
        else:
            self.snake.pop()

        return self.get_state(), reward, self.game_over, {"score": self.score}

    def _is_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position would result in collision.

        Args:
            position: Grid position (x, y) to check

        Returns:
            True if collision would occur (wall or self-collision)
        """
        x, y = position

        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True

        # Check collision with snake body (excluding tail which will be removed)
        if position in self.snake[:-1]:
            return True

        return False

    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food at random empty position.

        Returns:
            Food position (x, y) that doesn't overlap with snake
        """
        while True:
            food_pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if food_pos not in self.snake:
                return food_pos

    def get_state(self) -> np.ndarray:
        """Generate grid state representation using defined constants."""
        state = np.full((self.grid_size, self.grid_size), EMPTY, dtype=np.int32)

        # Mark snake body segments
        for segment in self.snake:
            x, y = segment
            state[y, x] = SNAKE_BODY

        # Mark snake head (overrides body at head position)
        head_x, head_y = self.snake[0]
        state[head_y, head_x] = SNAKE_HEAD

        # Mark food
        food_x, food_y = self.food
        state[food_y, food_x] = FOOD

        return state

    def get_head_position(self) -> Tuple[int, int]:
        """Get current snake head position.

        Returns:
            Head position as (x, y) coordinates
        """
        return self.snake[0]

    def get_food_position(self) -> Tuple[int, int]:
        """Get current food position.

        Returns:
            Food position as (x, y) coordinates
        """
        return self.food

    def get_snake_body(self) -> List[Tuple[int, int]]:
        """Get complete snake body positions.

        Returns:
            List of (x, y) positions from head to tail
        """
        return self.snake.copy()

    def get_direction(self) -> int:
        """Get current movement direction.

        Returns:
            Current direction (UP=0, RIGHT=1, DOWN=2, LEFT=3)
        """
        return self.direction

    def is_game_over(self) -> bool:
        """Check if game has ended.

        Returns:
            True if collision occurred and game ended
        """
        return self.game_over

    def get_score(self) -> int:
        """Get current game score.

        Returns:
            Number of food items eaten
        """
        return self.score

    def get_steps(self) -> int:
        """Get number of steps taken.

        Returns:
            Total steps since last reset
        """
        return self.steps

    def _calculate_position_penalty(self, new_head: Tuple[int, int]) -> float:
        """Calculate penalty for visiting specific positions.

        Args:
            new_head: New head position to evaluate

        Returns:
            Penalty value (negative float)
        """
        penalty = 0.0

        # Check if returning to previously visited position
        if new_head in self.visited_positions:
            penalty += REWARD_REVISIT

        # Check if oscillating (returning to position from 2 steps ago)
        if len(self.position_history) >= 2 and new_head == self.position_history[-2]:
            penalty += REWARD_OSCILLATE

        return penalty

    def _update_position_tracking(self, new_head: Tuple[int, int]) -> None:
        """Update position tracking for penalty calculations.

        Args:
            new_head: New head position that was just moved to
        """
        # Add to visited positions set
        self.visited_positions.add(new_head)

        # Add to position history and keep only last 3 positions
        self.position_history.append(new_head)
        if len(self.position_history) > 3:
            self.position_history.pop(0)

        # Add to recent positions and keep only last RECENT_VISIT_WINDOW positions
        self.recent_positions.append(new_head)
        if len(self.recent_positions) > RECENT_VISIT_WINDOW:
            self.recent_positions.pop(0)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Manhattan distance as float
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_distance_reward(self, new_head: Tuple[int, int]) -> float:
        """Calculate reward based on distance change to food.

        Args:
            new_head: New head position to evaluate

        Returns:
            Distance-based reward (positive if closer, negative if farther)
        """
        # Calculate new distance with the proposed move
        new_distance = self._manhattan_distance(new_head, self.food)

        # Compare with previous distance
        if new_distance < self.previous_distance_to_food:
            # Moving closer to food
            distance_reward = REWARD_CLOSER
        elif new_distance > self.previous_distance_to_food:
            # Moving farther from food
            distance_reward = REWARD_FARTHER
        else:
            # Same distance (rare, but possible)
            distance_reward = 0.0

        # Update previous distance for next step
        self.previous_distance_to_food = new_distance

        return distance_reward

    def _calculate_death_penalty(self) -> float:
        """Calculate death penalty based on survival time.

        Longer survival reduces the death penalty to encourage exploration.

        Returns:
            Death penalty value (negative float)
        """
        if self.steps < SURVIVAL_THRESHOLD:
            # Full penalty for quick death
            return REWARD_DEATH_BASE
        else:
            # Reduce penalty based on survival time
            survival_bonus = (self.steps - SURVIVAL_THRESHOLD) * DEATH_REDUCTION_RATE
            reduced_penalty = REWARD_DEATH_BASE + survival_bonus  # Adding positive bonus to negative penalty

            # Ensure penalty doesn't become too small
            min_penalty = REWARD_DEATH_BASE * MIN_DEATH_PENALTY_RATIO
            return max(min_penalty, reduced_penalty)

    def _calculate_food_reward(self) -> float:
        """Calculate food reward based on current score.

        Higher scores increase the value of each food item to encourage growth.

        Returns:
            Food reward value (positive float)
        """
        return REWARD_FOOD_BASE + (self.score * FOOD_SCORE_MULTIPLIER)