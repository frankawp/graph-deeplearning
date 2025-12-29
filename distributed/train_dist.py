"""
DistDGL分布式训练脚本
支持32节点CPU集群的异构图分布式训练
"""

import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import dgl
from dgl.distributed import DistGraph, DistDataLoader, node_split
from dgl.dataloading import NeighborSampler
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hetero_sage import HeteroSAGEForNodeClassification
from models.hetero_gat import HeteroGATForNodeClassification
from models.rgcn import RGCNForNodeClassification


class DistributedTrainer:
    """DistDGL分布式训练器"""

    def __init__(self,
                 config: Dict,
                 graph: DistGraph,
                 model: nn.Module):
        """
        初始化训练器

        Args:
            config: 训练配置
            graph: 分布式图
            model: 模型
        """
        self.config = config
        self.graph = graph
        self.model = model
        self.device = torch.device('cpu')

        # 获取分布式信息
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 设置优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0)
        )

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 训练统计
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def setup_data_loaders(self) -> Tuple[DistDataLoader, DistDataLoader, DistDataLoader]:
        """设置数据加载器"""
        target_ntype = self.config.get('target_node_type', 'user')
        pb = self.graph.get_partition_book()

        # 获取训练/验证/测试节点
        train_nids = node_split(
            self.graph.ndata['train_mask'][target_ntype],
            pb, ntype=target_ntype,
            force_even=True
        )
        val_nids = node_split(
            self.graph.ndata['val_mask'][target_ntype],
            pb, ntype=target_ntype
        )
        test_nids = node_split(
            self.graph.ndata['test_mask'][target_ntype],
            pb, ntype=target_ntype
        )

        if self.rank == 0:
            print(f"训练节点: {len(train_nids):,}")
            print(f"验证节点: {len(val_nids):,}")
            print(f"测试节点: {len(test_nids):,}")

        # 邻居采样器
        fanouts = self.config.get('fanouts', [15, 10, 5])
        sampler = NeighborSampler(fanouts)

        # 创建数据加载器
        batch_size = self.config.get('batch_size', 1024)

        train_loader = DistDataLoader(
            self.graph,
            {target_ntype: train_nids},
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        val_loader = DistDataLoader(
            self.graph,
            {target_ntype: val_nids},
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        test_loader = DistDataLoader(
            self.graph,
            {target_ntype: test_nids},
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DistDataLoader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        target_ntype = self.config.get('target_node_type', 'user')

        for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            # 获取输入特征和标签
            batch_inputs = {
                ntype: self.graph.ndata['feat'][ntype][ids]
                for ntype, ids in input_nodes.items()
                if 'feat' in self.graph.ndata and ntype in self.graph.ndata['feat']
            }
            batch_labels = self.graph.ndata['label'][target_ntype][output_nodes[target_ntype]]

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(blocks, batch_inputs)
            loss = self.criterion(logits, batch_labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * len(batch_labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)

        # 聚合所有worker的统计
        avg_loss = self._sync_metric(total_loss, total_samples)
        avg_acc = self._sync_metric(total_correct, total_samples)

        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DistDataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        target_ntype = self.config.get('target_node_type', 'user')

        for input_nodes, output_nodes, blocks in data_loader:
            batch_inputs = {
                ntype: self.graph.ndata['feat'][ntype][ids]
                for ntype, ids in input_nodes.items()
                if 'feat' in self.graph.ndata and ntype in self.graph.ndata['feat']
            }
            batch_labels = self.graph.ndata['label'][target_ntype][output_nodes[target_ntype]]

            logits = self.model(blocks, batch_inputs)
            preds = logits.argmax(dim=1)

            total_correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_labels)

        avg_acc = self._sync_metric(total_correct, total_samples)

        return {'accuracy': avg_acc}

    def _sync_metric(self, value: float, count: int) -> float:
        """同步所有worker的指标"""
        if not dist.is_initialized():
            return value / count if count > 0 else 0.0

        tensor = torch.tensor([value, count], dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor[0].item() / tensor[1].item() if tensor[1].item() > 0 else 0.0

    def train(self,
              train_loader: DistDataLoader,
              val_loader: DistDataLoader,
              test_loader: DistDataLoader) -> Dict:
        """完整训练流程"""
        num_epochs = self.config.get('num_epochs', 100)
        checkpoint_dir = self.config.get('checkpoint_path', './checkpoints')

        if self.rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"\n开始训练，共 {num_epochs} 个epoch...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.evaluate(val_loader)

            # 更新学习率
            self.scheduler.step(val_metrics['accuracy'])

            epoch_time = time.time() - start_time

            # 打印进度
            if self.rank == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Train Acc: {train_metrics['accuracy']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Time: {epoch_time:.1f}s")

                # 保存最佳模型
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_epoch = epoch
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_model.pt'),
                        epoch
                    )

            # 定期保存检查点
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                if self.rank == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt'),
                        epoch
                    )

        # 测试
        if self.rank == 0:
            print(f"\n加载最佳模型 (epoch {self.best_epoch+1})...")
            self.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))

        test_metrics = self.evaluate(test_loader)

        if self.rank == 0:
            print(f"\n=== 最终结果 ===")
            print(f"最佳验证准确率: {self.best_val_acc:.4f} (epoch {self.best_epoch+1})")
            print(f"测试准确率: {test_metrics['accuracy']:.4f}")

        return {
            'best_val_acc': self.best_val_acc,
            'test_acc': test_metrics['accuracy'],
            'best_epoch': self.best_epoch
        }

    def save_checkpoint(self, path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)


def build_model(config: Dict, graph: DistGraph) -> nn.Module:
    """构建模型"""
    model_type = config.get('model_type', 'sage')
    target_ntype = config.get('target_node_type', 'user')

    # 获取特征维度
    in_feats = {}
    for ntype in graph.ntypes:
        if 'feat' in graph.ndata and ntype in graph.ndata['feat']:
            in_feats[ntype] = graph.ndata['feat'][ntype].shape[1]
        else:
            in_feats[ntype] = config.get('hidden_channels', 256)

    # 获取边类型
    etypes = list(graph.canonical_etypes)

    if model_type == 'sage':
        model = HeteroSAGEForNodeClassification(
            in_feats=in_feats,
            hidden_feats=config.get('hidden_channels', 256),
            num_classes=config.get('num_classes', 10),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.5),
            target_ntype=target_ntype,
            etypes=etypes
        )
    elif model_type == 'gat':
        model = HeteroGATForNodeClassification(
            in_feats=in_feats,
            hidden_feats=config.get('hidden_channels', 256),
            num_classes=config.get('num_classes', 10),
            num_layers=config.get('num_layers', 3),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.5),
            target_ntype=target_ntype,
            etypes=etypes
        )
    elif model_type == 'rgcn':
        model = RGCNForNodeClassification(
            in_feats=in_feats,
            hidden_feats=config.get('hidden_channels', 256),
            num_classes=config.get('num_classes', 10),
            num_rels=len(etypes),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5),
            target_ntype=target_ntype,
            etypes=etypes
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    return model


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='DistDGL分布式训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--graph-name', type=str, default='hetero_graph', help='图名称')
    parser.add_argument('--part-config', type=str, required=True, help='分区配置文件')
    parser.add_argument('--ip-config', type=str, required=True, help='IP配置文件')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 初始化分布式环境
    dgl.distributed.initialize(args.ip_config)

    # 加载分布式图
    print("加载分布式图...")
    graph = DistGraph(args.graph_name, part_config=args.part_config)
    print(f"图信息: {graph.num_nodes()} 节点, {graph.num_edges()} 边")

    # 构建模型
    model = build_model(config, graph)
    print(f"模型: {type(model).__name__}")

    # 创建训练器
    trainer = DistributedTrainer(config, graph, model)

    # 设置数据加载器
    train_loader, val_loader, test_loader = trainer.setup_data_loaders()

    # 开始训练
    results = trainer.train(train_loader, val_loader, test_loader)

    print("\n训练完成！")


if __name__ == '__main__':
    main()
