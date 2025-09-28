"""Training package for Snake RL project."""

from .trainer import UniversalTrainer
from .comparison import AgentComparison

__all__ = [
    'UniversalTrainer',
    'AgentComparison'
]