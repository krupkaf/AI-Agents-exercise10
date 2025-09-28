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
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01