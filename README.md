# Snake Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.8+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of Snake game environment with Deep Q-Network (DQN) agent for reinforcement learning. Created as part of AI Agents course practical exercise.

## Features

- **Custom Environment**: Snake game built with Gymnasium interface
- **DQN Agent**: Deep Q-Network with experience replay and target network
- **Real-time Visualization**: PyGame rendering during training
- **Performance Tracking**: Live statistics and training metrics
- **Modular Architecture**: Clean separation of environment, agent, and visualization

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
# Train DQN agent with real-time visualization
uv run python src/main.py --mode train --episodes 1000

# Train in headless mode (faster)
uv run python src/main.py --mode train --episodes 1000 --headless

# Resume training from checkpoint
uv run python src/main.py --mode train --load-model models/checkpoint.pth
```

### Testing

```bash
# Test trained agent
uv run python src/main.py --mode test --load-model models/best_model.pth

# Run test suite
uv run pytest tests/
```

## Architecture

### Environment (`src/environment/`)
- **Snake Game Logic**: Core mechanics (movement, collision, food spawning)
- **Gymnasium Interface**: Standard RL environment API
- **State Representation**: 20x20 grid or feature vector
- **Reward System**: +10 food, -10 death, -1 step

### Agent (`src/agent/`)
- **DQN Network**: Convolutional + Dense layers
- **Experience Replay**: Configurable buffer size
- **Target Network**: Stable Q-learning
- **Exploration**: Epsilon-greedy strategy

### Visualization (`src/visualization/`)
- **Real-time Rendering**: 800x600 PyGame window
- **Training Statistics**: Live performance graphs
- **Interactive Controls**: Pause, screenshot, speed control

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

## Results

| Metric | Value |
|--------|--------|
| Max Score | TBD |
| Avg Score (last 100) | TBD |
| Training Episodes | TBD |
| Training Time | TBD |

*Results will be updated after training completion.*

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
│   ├── agent/          # DQN agent implementation
│   ├── visualization/  # PyGame rendering
│   ├── utils/          # Configuration and utilities
│   └── main.py         # Training and testing script
├── tests/              # Unit tests
├── models/             # Saved model checkpoints
├── results/            # Training logs and screenshots
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