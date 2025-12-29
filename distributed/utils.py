"""
分布式训练工具函数
"""

import os
import torch
import torch.distributed as dist
import dgl
from typing import Dict, Optional
import socket
import logging


def setup_distributed(backend: str = 'gloo',
                      init_method: Optional[str] = None,
                      rank: Optional[int] = None,
                      world_size: Optional[int] = None) -> bool:
    """
    设置分布式训练环境

    Args:
        backend: 通信后端 ('gloo' for CPU, 'nccl' for GPU)
        init_method: 初始化方法 (如 'tcp://host:port')
        rank: 当前进程rank
        world_size: 总进程数

    Returns:
        是否成功初始化
    """
    # 从环境变量获取参数
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if init_method is None:
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        init_method = f'tcp://{master_addr}:{master_port}'

    if world_size == 1:
        print("单机模式，跳过分布式初始化")
        return False

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        print(f"分布式环境初始化成功: rank={rank}, world_size={world_size}")
        return True
    except Exception as e:
        print(f"分布式环境初始化失败: {e}")
        return False


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_ip() -> str:
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def sync_tensor(tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
    """
    同步所有进程的张量

    Args:
        tensor: 要同步的张量
        op: 聚合操作 ('sum', 'mean', 'max', 'min')

    Returns:
        同步后的张量
    """
    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()

    if op == 'sum':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == 'mean':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    elif op == 'max':
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == 'min':
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)

    return tensor


def broadcast_object(obj, src: int = 0):
    """
    从源进程广播Python对象

    Args:
        obj: 要广播的对象
        src: 源进程rank

    Returns:
        广播后的对象
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """判断是否是主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """获取当前进程rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class DistributedLogger:
    """分布式日志记录器"""

    def __init__(self, name: str, log_dir: str = './logs'):
        """
        初始化日志记录器

        Args:
            name: 日志名称
            log_dir: 日志目录
        """
        self.rank = get_rank()
        self.is_main = is_main_process()

        # 只有主进程创建文件日志
        if self.is_main:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'{name}.log')

            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)

            # 文件处理器
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        else:
            self.logger = None

    def info(self, msg: str):
        """记录INFO级别日志"""
        if self.is_main and self.logger:
            self.logger.info(msg)

    def warning(self, msg: str):
        """记录WARNING级别日志"""
        if self.is_main and self.logger:
            self.logger.warning(msg)

    def error(self, msg: str):
        """记录ERROR级别日志"""
        if self.is_main and self.logger:
            self.logger.error(msg)

    def debug(self, msg: str):
        """记录DEBUG级别日志"""
        if self.is_main and self.logger:
            self.logger.debug(msg)


class EarlyStopping:
    """早停机制"""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        初始化早停

        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善值
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前分数

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def compute_metrics(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    num_classes: int) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数

    Returns:
        指标字典
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
