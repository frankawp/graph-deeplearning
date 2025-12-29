"""
图分区模块
使用DGL的分布式分区工具进行图分区
"""

from .dgl_partitioner import DGLPartitioner
from .partition_dispatch import PartitionDispatcher

__all__ = ['DGLPartitioner', 'PartitionDispatcher']
