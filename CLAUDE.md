# Snake Reinforcement Learning Project

## Project Overview
Implementation of Snake game environment with Deep Q-Network (DQN) agent for AI Agents course (Lesson 10).

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
│   │   ├── dqn_agent.py          # Main DQN agent implementation
│   │   ├── neural_network.py     # PyTorch neural network
│   │   └── replay_buffer.py      # Experience replay buffer
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── pygame_renderer.py    # Real-time PyGame visualization
│   │   ├── stats_display.py      # Training statistics display
│   │   └── colors.py             # Color constants and themes
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration parameters
│   └── main.py                   # Main training and testing script
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
4. **Agent**: DQN implementation with experience replay
5. **Training**: Integration and hyperparameter tuning
6. **Evaluation**: Performance testing and documentation
7. **Submission**: GitHub repository preparation

## Implementation Phases

### Phase 1: Repository & Environment Setup (30 min)
- Create project directory structure
- Configure uv with dependencies: torch, pygame, numpy, matplotlib, gymnasium
- Git initialization + first commit
- Create GitHub repository (public, MIT license)
- Initial documentation setup

### Phase 2: Core Documentation (15 min)
- README.md with badges and MIT license
- LICENSE file (MIT)
- .gitignore (Python template)
- docs/ folder with original Czech assignment

### Phase 3: Snake Environment Implementation (60 min)
- Snake game logic (movement, collision, food spawning)
- Gymnasium-compatible wrapper
- State representation (20x20 grid)
- Action space (4 directions) + reward system
- Unit tests for environment

### Phase 4: PyGame Visualization (45 min)
- Game window (800x600px) with grid rendering
- Real-time snake/food/background display
- Info panel: score, episode, epsilon, steps
- Control features: pause, screenshot, FPS control
- Integration with training loop

### Phase 5: DQN Agent (90 min)
- Neural network architecture (Conv2D → Dense)
- Experience replay buffer (configurable size)
- Target network + epsilon-greedy exploration
- Save/load model functionality
- Hyperparameter configuration

### Phase 6: Training & Integration (60 min)
- Training loop with PyGame real-time display
- Statistics tracking: scores, loss curves, epsilon decay
- Checkpointing and model saving
- Performance monitoring and logging

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
- **Reward System**:
  - +10 for eating food
  - -10 for collision/death
  - -1 for each step (efficiency incentive)

### DQN Agent Features
- **Neural Network**: Convolutional + Dense layers
- **Experience Replay**: Configurable buffer size (default: 10,000)
- **Target Network**: Updated every N steps for stability
- **Exploration**: Epsilon-greedy strategy (1.0 → 0.01)
- **Optimization**: Adam optimizer with learning rate scheduling

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

# Run project
uv run python src/main.py --mode train --episodes 1000
```

## Quality Standards
- **Code Style**: Black formatter, isort imports
- **Testing**: Unit tests for core components
- **Documentation**: Comprehensive docstrings
- **Logging**: Structured logging for training progress
- **Performance**: Efficient implementation for real-time visualization

## Success Criteria
1. **Functional Environment**: Snake game following Gymnasium interface
2. **Working Agent**: DQN successfully learning to play
3. **Real-time Visualization**: PyGame rendering during training
4. **Performance**: Agent achieving reasonable scores
5. **Documentation**: Complete README with usage examples
6. **Submission**: Public GitHub repository with MIT license