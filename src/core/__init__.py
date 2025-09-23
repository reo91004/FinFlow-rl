# src/core/__init__.py

"""
FinFlow Core Module

Core training and environment components
"""

from .env import PortfolioEnv
from .replay import PrioritizedReplayBuffer, Transition
from .networks import DirichletActor, QNetwork, ValueNetwork
from .objectives import PortfolioObjective
from .iql import IQLAgent
from .td3bc import TD3BCAgent
from .offline_trainer import OfflineTrainer
from .offline_dataset import OfflineDataset

__all__ = [
    # Environment
    'PortfolioEnv',

    # Replay Buffer
    'PrioritizedReplayBuffer',
    'Transition',

    # Networks
    'DirichletActor',
    'QNetwork',
    'ValueNetwork',

    # Objectives
    'PortfolioObjective',

    # Agents
    'IQLAgent',
    'TD3BCAgent',

    # Trainers
    'OfflineTrainer',

    # Datasets
    'OfflineDataset',
]