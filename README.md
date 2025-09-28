# Snake Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.8+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of Snake game environment with multiple Reinforcement Learning agents (DQN, REINFORCE, PPO) for educational comparison. Created as part of AI Agents course practical exercise.

## Features

- **Custom Environment**: Snake game built with Gymnasium interface
- **Multiple RL Algorithms**: DQN, REINFORCE, and PPO implementations
- **Real-time Visualization**: PyGame rendering during training
- **Performance Tracking**: Live statistics and training metrics
- **Algorithm Comparison**: Side-by-side performance analysis
- **Modular Architecture**: Clean separation of environment, agents, and visualization

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/username/snake-rl.git
cd snake-rl

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Training

```bash
# Train DQN agent (primary implementation)
uv run python src/main.py --agent dqn --episodes 1000

# Train REINFORCE agent (policy gradient baseline)
uv run python src/main.py --agent reinforce --episodes 500

# Train PPO agent (advanced policy gradient)
uv run python src/main.py --agent ppo --episodes 1000

# Compare all algorithms
uv run python src/main.py --compare-all --episodes 500

# Train in headless mode (faster)
uv run python src/main.py --agent dqn --episodes 1000 --headless
```

### Testing

```bash
# Test trained DQN agent
uv run python src/main.py --agent dqn --mode test --load-model models/dqn_best.pth

# Test REINFORCE agent
uv run python src/main.py --agent reinforce --mode test --load-model models/reinforce_best.pth

# Run test suite
uv run pytest tests/
```

## Architecture

### Environment (`src/environment/`)
- **Snake Game Logic**: Core mechanics (movement, collision, food spawning)
- **Gymnasium Interface**: Standard RL environment API
- **State Representation**: 20x20 grid or feature vector
- **Food System**: Single food item, immediate respawn at random empty position
- **Snake Growth**: Initial length 1 segment, grows by 1 segment per food eaten
- **Episode Limit**: 1000 steps maximum (configurable in SnakeEnv)
- **Termination**: `terminated` (collision) vs `truncated` (time limit)
- **Reward System**: +20 food, -10 death, -0.001 step, -1 revisit, -3 oscillate (anti-stuck design)

### Agents (`src/agent/`)

#### DQN (Deep Q-Network) - Primary Implementation
- **Neural Network**: Convolutional + Dense layers for grid processing
- **Experience Replay**: Configurable buffer size (default: 10,000)
- **Target Network**: Updated every N steps for stability
- **Exploration**: Epsilon-greedy strategy (1.0 → 0.01)

#### REINFORCE - Policy Gradient Baseline
- **Policy Network**: Direct action probability output
- **Monte Carlo**: Full episode rollouts for gradient estimation
- **Baseline**: Value function for variance reduction
- **Exploration**: Stochastic policy sampling

#### PPO - Advanced Policy Gradient (Optional)
- **Actor-Critic**: Separate policy and value networks
- **Clipped Objective**: Proximal policy optimization constraint
- **GAE**: Generalized Advantage Estimation
- **Multiple Epochs**: Efficient sample reuse

### Visualization (`src/visualization/`)
- **Real-time Rendering**: 800x600 PyGame window
- **Training Statistics**: Live performance graphs (StatsTracker utility class)
- **Interactive Controls**: Pause, screenshot, speed control
- **Statistics Export**: Matplotlib plots and JSON logging (ready for DQN training)

## Configuration

Key parameters in `src/utils/config.py`:

```python
# Environment
GRID_SIZE = 20
WINDOW_SIZE = 800

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = 0.995

# Network
HIDDEN_SIZE = 512
TARGET_UPDATE_FREQ = 100
```

## Anti-Oscillation Reward Design

The reward system is specifically designed to prevent the snake from getting stuck in safe positions and encourage active food exploration:

**Positive Rewards:**
- `+20.0` for eating food (doubled from original +10 to encourage active seeking)

**Minimal Exploration Penalty:**
- `-0.001` per step (reduced from -0.01 to allow longer exploration without heavy penalty)

**Anti-Stuck Penalties:**
- `-1.0` for returning to any previously visited position
- `-3.0` for oscillating between two positions (returning to position from 2 steps ago)

**Game Over Penalty:**
- `-10.0` for collision/death

This design eliminates the common problem where DQN agents learn to "ping-pong" between safe positions indefinitely, instead forcing them to actively explore and seek food.

## Results

### Algorithm Comparison

| Algorithm | Max Score | Avg Score (last 100) | Training Episodes | Implementation Status |
|-----------|-----------|-------------------|------------------|---------------------|
| DQN | ✅ Trained | ✅ Available | 1000 | ✅ **Complete** |
| REINFORCE | - | - | - | 🔄 Architecture ready |
| PPO | - | - | - | 🔄 Architecture ready |

**DQN Status**: ✅ Fully trained and tested
- Model saved: `models/dqn_final.pth`
- Interactive testing available with controls (pause, speed, screenshot)
- Real-time visualization during training and testing

### Performance Metrics
- **Learning Stability**: DQN vs Policy Gradient methods
- **Sample Efficiency**: Episodes needed to reach target performance
- **Final Performance**: Maximum achievable scores
- **Training Speed**: Wall-clock time comparison

## Requirements

- Python 3.10+
- PyTorch 2.8+
- PyGame 2.6+
- Gymnasium 1.2+
- NumPy, Matplotlib

See `pyproject.toml` for complete dependencies.

## Development

```bash
# Install development dependencies
uv sync --group dev

# Code formatting
uv run black src/ tests/
uv run isort src/ tests/

# Linting
uv run flake8 src/ tests/

# Testing
uv run pytest tests/ -v
```

## Project Structure

```
snake-rl/
├── src/
│   ├── environment/     # Snake game environment
│   ├── agent/          # Multi-algorithm RL agents (DQN, REINFORCE, PPO)
│   ├── training/       # Universal trainer and comparison framework
│   ├── visualization/  # PyGame rendering and statistics
│   ├── utils/          # Configuration and utilities
│   └── main.py         # Training and testing script
├── demo/               # Demo scripts
├── tests/              # Unit tests
├── models/             # Saved model checkpoints (by algorithm)
├── results/            # Training logs, screenshots, comparison plots
└── docs/               # Documentation and assignment
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Acknowledgments

- Framework: [Gymnasium](https://gymnasium.farama.org/)
- Deep Learning: [PyTorch](https://pytorch.org/)
- Visualization: [PyGame](https://www.pygame.org/)