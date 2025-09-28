"""Constants for Snake game state encoding."""

# Direction constants (integers matching action indices)
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Grid state encoding values (integers, cast to int32 in arrays)
EMPTY = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
FOOD = 3

# Reward values (float for precise calculations)
REWARD_FOOD = 20.0          # Reward for eating food
REWARD_DEATH = -10.0        # Penalty for collision/death
REWARD_STEP = -0.001        # Small penalty for each step
REWARD_REVISIT = -1.0       # Penalty for returning to previously visited position
REWARD_OSCILLATE = -3.0     # Heavy penalty for returning to position from 2 steps ago
REWARD_CLOSER = 0.1         # Reward for moving closer to food
REWARD_FARTHER = -0.1       # Penalty for moving farther from food