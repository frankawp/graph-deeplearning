"""
分布式训练模块
基于DistDGL的分布式图神经网络训练
"""

from .train_dist import DistributedTrainer
from .utils import setup_distributed, cleanup_distributed

__all__ = ['DistributedTrainer', 'setup_distributed', 'cleanup_distributed']
