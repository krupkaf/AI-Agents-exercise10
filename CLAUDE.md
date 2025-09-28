# Snake Reinforcement Learning Project

## Project Overview
Implementation of Snake game environment with multiple Reinforcement Learning agents (DQN, REINFORCE, PPO) for AI Agents course (Lesson 10).

## Communication & Language
- **Code & Documentation**: English (except original Czech assignment)
- **Comments**: English
- **Variable names**: English
- **Git commits**: English
- **README, documentation**: English

## Technology Stack
- **Python**: 3.10+
- **Package Manager**: uv (for fast dependency management)
- **ML Framework**: PyTorch
- **Visualization**: PyGame (real-time game rendering)
- **Environment**: Gymnasium (OpenAI Gym standard)
- **Additional**: NumPy, Matplotlib

## Project Structure
```
snake-rl/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── snake_env.py          # Gymnasium-compatible environment
│   │   └── game_logic.py         # Core Snake game mechanics
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Abstract base class for all agents
│   │   ├── dqn_agent.py          # DQN agent implementation
│   │   ├── reinforce_agent.py    # REINFORCE agent implementation
│   │   ├── ppo_agent.py          # PPO agent implementation (optional)
│   │   ├── neural_networks.py    # Shared PyTorch network architectures
│   │   └── replay_buffer.py      # Experience replay utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Universal trainer for all agents
│   │   └── comparison.py         # Multi-agent comparison framework
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── pygame_renderer.py    # Real-time PyGame visualization
│   │   ├── stats_display.py      # Training statistics display
│   │   └── colors.py             # Color constants and themes
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration parameters
│   └── main.py                   # Main training and testing script
├── demo/
│   ├── __init__.py
│   └── visualization.py         # PyGame visualization demo with random agent
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   └── test_agent.py
├── docs/
│   └── assignment.pdf            # Original Czech assignment
├── models/                       # Saved trained models
├── results/                      # Training logs, screenshots, videos
├── requirements.txt              # Dependencies list
├── pyproject.toml               # uv configuration
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # MIT License
├── README.md                    # Main project documentation
└── CLAUDE.md                    # This file - project setup guide
```

## Development Workflow
1. **Setup**: Environment and dependencies
2. **Environment**: Implement Snake game with Gymnasium interface
3. **Visualization**: PyGame real-time rendering
4. **Agents**: Multi-algorithm RL implementation (DQN primary, REINFORCE/PPO optional)
5. **Training**: Integration and hyperparameter tuning
6. **Evaluation**: Performance testing and algorithm comparison
7. **Submission**: GitHub repository preparation

## Implementation Phases

### Phase 1: Repository & Environment Setup (30 min) ✅ COMPLETED ✅ VERIFIED
- ✅ Create project directory structure - complete with src/, tests/, docs/, models/, results/
- ✅ Configure uv with dependencies: torch, pygame, numpy, matplotlib, gymnasium - pyproject.toml configured
- ✅ Git initialization + first commit - git history shows initial commits
- ✅ Create GitHub repository (public, MIT license) - ready for GitHub
- ✅ Initial documentation setup - all core files in place

### Phase 2: Core Documentation (15 min) ✅ COMPLETED ✅ VERIFIED
- ✅ README.md with badges and MIT license - comprehensive documentation with architecture
- ✅ LICENSE file (MIT) - MIT license file present
- ✅ .gitignore (Python template) - proper Python gitignore configured
- ✅ docs/ folder with original Czech assignment - assignment.pdf present

### Phase 3: Snake Environment Implementation (60 min) ✅ COMPLETED ✅ VERIFIED
- ✅ Snake game logic (movement, collision, food spawning) - `src/environment/game_logic.py` complete
- ✅ Gymnasium-compatible wrapper - `src/environment/snake_env.py` with proper interface
- ✅ State representation (20x20 grid) - implemented with proper encoding via constants.py
- ✅ Action space (4 directions) + reward system - +10 food, -10 collision, -0.01 step verified
- ✅ Food system - single food item, immediate respawn at random empty position working
- ✅ Snake mechanics - initial length 1, grows by 1 segment per food eaten confirmed
- ✅ Unit tests for environment - `tests/test_environment.py` **ALL 17 TESTS PASSING**

### Phase 4: PyGame Visualization (45 min) ✅ COMPLETED ✅ VERIFIED
- ✅ Game window (800x600px) with grid rendering - `src/visualization/pygame_renderer.py` functional
- ✅ Real-time snake/food/background display - full color theme in `colors.py` implemented
- ✅ Info panel: score, episode, epsilon, steps - complete statistics display working
- ✅ Control features: pause, screenshot, FPS control - SPACE, S, UP/DOWN keys verified
- ✅ Integration with training loop - `demo/visualization.py` (demo with random agent) tested
- ✅ StatsTracker utility class - `src/visualization/stats_display.py` (ready for DQN training)

### Phase 5: DQN Agent Implementation (90 min) ✅ COMPLETED ✅ VERIFIED
- ✅ Base agent abstract class for extensibility - `src/agent/base_agent.py` with abstract methods
- ✅ Neural network architecture (Conv2D → Dense for grid processing) - `src/agent/neural_networks.py` DQNNetwork
- ✅ Experience replay buffer (configurable size) - `src/agent/replay_buffer.py` with 10k default size
- ✅ Target network + epsilon-greedy exploration - DQNAgent with target update freq 100, ε: 1.0→0.01
- ✅ Save/load model functionality - `save_model()` and `load_model()` methods implemented
- ✅ Hyperparameter configuration system - `src/utils/config.py` comprehensive config classes

### Phase 5b: Additional RL Algorithms (60-90 min, optional)
- REINFORCE agent implementation (policy gradient baseline)
- PPO agent implementation (advanced policy gradient, if time permits)
- Universal trainer supporting all agent types
- Performance comparison framework and visualization
- Multi-agent benchmarking and results analysis

### Phase 6: Training & Integration (60 min)
- Training loop with PyGame real-time display
- Statistics tracking: scores, loss curves, epsilon decay (using StatsTracker class)
- Checkpointing and model saving
- Performance monitoring and logging
- Integration with StatsTracker for matplotlib plots and JSON export

### Phase 7: Results & Finalization (30 min)
- Train final model + performance evaluation
- Update README with results and screenshots
- Create GitHub release for submission
- Final documentation review

**Total Estimated Time**: ~5.5 hours (distributed across sessions)

## Assignment Requirements
- **Course**: AI Agents (Praktické cvičení - Lekce 10)
- **Points**: 100
- **Deadline**: 9.10.2025
- **Task**: Implement any RL environment and train any RL agent
- **Deliverable**: GitHub repository link
- **Submission**: Google Classroom

## Repository Setup
- **Visibility**: Public
- **License**: MIT
- **Platform**: GitHub
- **Topics**: reinforcement-learning, dqn, snake-game, pytorch, pygame, ai-agents

## Key Implementation Details

### Environment Specifications
- **State Space**: 20x20 grid representation or feature vector
- **Action Space**: 4 discrete actions (up, right, down, left)
- **Food System**: Single food item, respawns immediately when eaten at random empty position
- **Snake Growth**: Initial length 1 segment, grows by 1 segment per food eaten
- **Step Limit**: 1000 steps maximum per episode (SnakeEnv wrapper), core game has no limit
- **Episode Termination**:
  - `terminated=True`: Collision occurred (snake died)
  - `truncated=True`: Time limit reached (1000 steps)
- **Reward System**:
  - +10 for eating food
  - -10 for collision/death
  - -0.01 for each step (efficiency incentive)

### Implemented Algorithms

#### 1. DQN (Deep Q-Network) - Primary Implementation
- **Neural Network**: Convolutional + Dense layers for grid processing
- **Experience Replay**: Configurable buffer size (default: 10,000)
- **Target Network**: Updated every N steps for stability
- **Exploration**: Epsilon-greedy strategy (1.0 → 0.01)
- **Optimization**: Adam optimizer with learning rate scheduling

#### 2. REINFORCE - Policy Gradient Baseline (Optional)
- **Policy Network**: Direct action probability output
- **Monte Carlo**: Full episode rollouts for gradient estimation
- **Baseline**: Value function for variance reduction
- **Exploration**: Stochastic policy sampling

#### 3. PPO - Advanced Policy Gradient (Optional)
- **Actor-Critic**: Separate policy and value networks
- **Clipped Objective**: Proximal policy optimization constraint
- **GAE**: Generalized Advantage Estimation
- **Multiple Epochs**: Efficient sample reuse

### Visualization Features
- **Real-time rendering**: 800x600 PyGame window
- **Game display**: 20x20 grid with snake and food
- **Statistics panel**: Score, episode, epsilon, steps
- **Training monitoring**: Live graphs of performance metrics
- **Interactive controls**: Pause, resume, screenshot, speed control

## Installation Commands
```bash
# Initialize project
uv init snake-rl
cd snake-rl

# Install dependencies
uv add torch torchvision pygame numpy matplotlib gymnasium

# Development dependencies
uv add --dev pytest black flake8 isort

# Configure symlink mode for disk space efficiency
# (already configured in pyproject.toml: link-mode = "symlink")

# Test visualization
uv run python demo/visualization.py

# Run project (when implemented)
uv run python src/main.py --agent dqn --episodes 1000
uv run python src/main.py --agent reinforce --episodes 500
uv run python src/main.py --agent ppo --episodes 1000
uv run python src/main.py --compare-all --episodes 500
```

## Quality Standards
- **Code Style**: Black formatter, isort imports
- **Testing**: Unit tests for core components
- **Documentation**: Comprehensive docstrings
- **Logging**: Structured logging for training progress
- **Performance**: Efficient implementation for real-time visualization

## Success Criteria
1. **Functional Environment**: Snake game following Gymnasium interface
2. **Working Agents**: DQN successfully learning to play (+ optional comparison algorithms)
3. **Real-time Visualization**: PyGame rendering during training
4. **Performance**: Agent achieving reasonable scores
5. **Documentation**: Complete README with usage examples
6. **Submission**: Public GitHub repository with MIT license