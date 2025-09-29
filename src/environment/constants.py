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

# Reward values - "Survival First" philosophy
REWARD_STEP = 0.0            # No penalty for movement
REWARD_SURVIVAL = 0.1        # Bonus for each step survived (base for DQN)
REWARD_SURVIVAL_PG = 0.5     # Higher survival bonus for policy gradient methods (PPO/REINFORCE)
REWARD_DEATH_BASE = -200.0   # High death penalty, but survival bonuses compensate
REWARD_FOOD_BASE = 100.0     # High food reward
REWARD_NEW_CELL = 0.5        # Bonus for exploring recently unvisited cells
RECENT_VISIT_WINDOW = 20     # Cells visited in last N steps don't give exploration bonus
REWARD_REVISIT = -0.05       # Small penalty for revisiting (sometimes necessary)
REWARD_OSCILLATE = -0.8      # Light penalty for oscillation (but not prohibitive)
REWARD_PATTERN_REPEAT = -5.0 # Strong penalty for repeating movement patterns
REWARD_CLOSER = 1.0          # Strong reward for getting closer to food
REWARD_FARTHER = -0.3        # Moderate penalty for moving away from food

# Pattern detection parameters
PATTERN_DETECTION_WINDOW = 8  # Number of recent moves to analyze for patterns

# Progressive reward system parameters
SURVIVAL_THRESHOLD = 50      # Minimum steps before death penalty reduction
DEATH_REDUCTION_RATE = 0.5   # Rate of death penalty reduction per step over threshold
FOOD_SCORE_MULTIPLIER = 10.0  # Multiplier for progressive food reward (base + score * multiplier)
MIN_DEATH_PENALTY_RATIO = 0.2  # Minimum death penalty as ratio of base penalty

