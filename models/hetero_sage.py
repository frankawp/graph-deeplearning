"""
异构图GraphSAGE模型
支持多种节点类型和边类型的消息传递
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv, SAGEConv
from typing import Dict, List, Tuple, Optional


class HeteroGraphSAGE(nn.Module):
    """异构图GraphSAGE模型"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 out_feats: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 aggregator_type: str = 'mean',
                 activation: nn.Module = F.relu,
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化模型

        Args:
            in_feats: 各节点类型的输入特征维度 {ntype: dim}
            hidden_feats: 隐藏层特征维度
            out_feats: 输出特征维度
            num_layers: GNN层数
            dropout: Dropout比例
            aggregator_type: 聚合类型 ('mean', 'gcn', 'pool', 'lstm')
            activation: 激活函数
            etypes: 边类型列表 [(src, rel, dst), ...]
        """
        super().__init__()

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.etypes = etypes

        # 输入投影层 - 将不同维度的输入统一到hidden_feats
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_feats.items():
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_feats)

        # 异构图卷积层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 每层为每种边类型创建一个SAGEConv
            if i == num_layers - 1:
                # 最后一层输出out_feats
                layer = self._create_hetero_conv(hidden_feats, out_feats, aggregator_type)
            else:
                layer = self._create_hetero_conv(hidden_feats, hidden_feats, aggregator_type)
            self.layers.append(layer)

        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_feats if i < num_layers - 1 else out_feats)
            for i in range(num_layers)
        ])

    def _create_hetero_conv(self,
                            in_feats: int,
                            out_feats: int,
                            aggregator_type: str) -> HeteroGraphConv:
        """创建异构图卷积层"""
        conv_dict = {}

        if self.etypes:
            for etype in self.etypes:
                conv_dict[etype] = SAGEConv(
                    in_feats,
                    out_feats,
                    aggregator_type=aggregator_type
                )

        return HeteroGraphConv(conv_dict, aggregate='sum')

    def forward(self,
                blocks: List[dgl.DGLHeteroGraph],
                x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            blocks: 采样得到的子图列表（从输入到输出）
            x: 输入节点特征 {ntype: tensor}

        Returns:
            输出节点特征 {ntype: tensor}
        """
        # 输入投影
        h = {}
        for ntype, feat in x.items():
            if ntype in self.input_proj:
                h[ntype] = self.input_proj[ntype](feat)
            else:
                h[ntype] = feat

        # 消息传递
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

            # 归一化和激活（最后一层不激活）
            if i < self.num_layers - 1:
                h = {ntype: self.norms[i](feat) for ntype, feat in h.items()}
                h = {ntype: self.activation(feat) for ntype, feat in h.items()}
                h = {ntype: F.dropout(feat, p=self.dropout, training=self.training)
                     for ntype, feat in h.items()}

        return h

    def inference(self,
                  graph: dgl.DGLHeteroGraph,
                  x: Dict[str, torch.Tensor],
                  batch_size: int = 1024,
                  device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """
        全图推理（用于评估）

        Args:
            graph: 完整图
            x: 输入节点特征
            batch_size: 批大小
            device: 设备

        Returns:
            所有节点的输出特征
        """
        # 输入投影
        h = {}
        for ntype, feat in x.items():
            if ntype in self.input_proj:
                h[ntype] = self.input_proj[ntype](feat.to(device))
            else:
                h[ntype] = feat.to(device)

        # 逐层推理
        for i, layer in enumerate(self.layers):
            next_h = {}

            for ntype in graph.ntypes:
                if ntype not in h:
                    continue

                # 对每种节点类型进行批处理推理
                num_nodes = graph.num_nodes(ntype)
                y = torch.zeros(num_nodes, self.hidden_feats if i < self.num_layers - 1 else self.out_feats)

                for start in range(0, num_nodes, batch_size):
                    end = min(start + batch_size, num_nodes)
                    batch_nodes = {ntype: torch.arange(start, end)}

                    # 采样子图
                    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                    _, _, blocks = sampler.sample(graph, batch_nodes)

                    # 前向传播
                    block_h = {nt: h[nt][blocks[0].srcnodes[nt].data[dgl.NID]]
                              for nt in h if nt in blocks[0].srctypes}
                    out = layer(blocks[0], block_h)

                    if ntype in out:
                        y[start:end] = out[ntype].cpu()

                next_h[ntype] = y

            # 归一化和激活
            if i < self.num_layers - 1:
                h = {ntype: self.norms[i](feat) for ntype, feat in next_h.items()}
                h = {ntype: self.activation(feat) for ntype, feat in h.items()}
            else:
                h = next_h

        return h


class HeteroSAGEForNodeClassification(nn.Module):
    """用于节点分类的异构GraphSAGE"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 num_classes: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 target_ntype: str = 'user',
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化模型

        Args:
            in_feats: 各节点类型的输入特征维度
            hidden_feats: 隐藏层特征维度
            num_classes: 分类类别数
            num_layers: GNN层数
            dropout: Dropout比例
            target_ntype: 目标节点类型
            etypes: 边类型列表
        """
        super().__init__()

        self.target_ntype = target_ntype

        # GNN编码器
        self.gnn = HeteroGraphSAGE(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_layers=num_layers,
            dropout=dropout,
            etypes=etypes
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats // 2, num_classes)
        )

    def forward(self,
                blocks: List[dgl.DGLHeteroGraph],
                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            blocks: 采样得到的子图列表
            x: 输入节点特征

        Returns:
            目标节点的分类logits
        """
        h = self.gnn(blocks, x)

        # 只对目标节点类型进行分类
        if self.target_ntype in h:
            return self.classifier(h[self.target_ntype])
        else:
            raise ValueError(f"目标节点类型 {self.target_ntype} 不在输出中")


class HeteroSAGEForLinkPrediction(nn.Module):
    """用于链接预测的异构GraphSAGE"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化模型

        Args:
            in_feats: 各节点类型的输入特征维度
            hidden_feats: 隐藏层特征维度
            num_layers: GNN层数
            dropout: Dropout比例
            etypes: 边类型列表
        """
        super().__init__()

        # GNN编码器
        self.gnn = HeteroGraphSAGE(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_layers=num_layers,
            dropout=dropout,
            etypes=etypes
        )

        # 边预测MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 1)
        )

    def forward(self,
                blocks: List[dgl.DGLHeteroGraph],
                x: Dict[str, torch.Tensor],
                pos_graph: dgl.DGLHeteroGraph,
                neg_graph: dgl.DGLHeteroGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            blocks: 采样得到的子图列表
            x: 输入节点特征
            pos_graph: 正样本边图
            neg_graph: 负样本边图

        Returns:
            (正样本分数, 负样本分数)
        """
        h = self.gnn(blocks, x)

        # 计算正样本分数
        pos_scores = self._compute_edge_scores(h, pos_graph)

        # 计算负样本分数
        neg_scores = self._compute_edge_scores(h, neg_graph)

        return pos_scores, neg_scores

    def _compute_edge_scores(self,
                             h: Dict[str, torch.Tensor],
                             graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        """计算边的分数"""
        scores = []

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            src, dst = graph.edges(etype=etype)

            if src_type in h and dst_type in h:
                src_h = h[src_type][src]
                dst_h = h[dst_type][dst]
                edge_h = torch.cat([src_h, dst_h], dim=1)
                score = self.predictor(edge_h).squeeze(-1)
                scores.append(score)

        return torch.cat(scores) if scores else torch.tensor([])
