"""
R-GCN (Relational Graph Convolutional Network)
专门为异构图设计的关系图卷积网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv, HeteroGraphConv
from typing import Dict, List, Tuple, Optional


class RGCN(nn.Module):
    """R-GCN模型"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 out_feats: int,
                 num_rels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 regularizer: str = 'basis',
                 num_bases: int = 8,
                 self_loop: bool = True,
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化R-GCN模型

        Args:
            in_feats: 各节点类型的输入特征维度
            hidden_feats: 隐藏层特征维度
            out_feats: 输出特征维度
            num_rels: 关系类型数量
            num_layers: GNN层数
            dropout: Dropout比例
            regularizer: 正则化方式 ('basis' 或 'bdd')
            num_bases: 基分解的基数量
            self_loop: 是否添加自环
            etypes: 边类型列表
        """
        super().__init__()

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.dropout = dropout
        self.etypes = etypes

        # 输入投影层
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_feats.items():
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_feats)

        # R-GCN层
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                layer = self._create_rgcn_layer(
                    hidden_feats, out_feats, num_rels,
                    regularizer, num_bases, self_loop
                )
            else:
                layer = self._create_rgcn_layer(
                    hidden_feats, hidden_feats, num_rels,
                    regularizer, num_bases, self_loop
                )
            self.layers.append(layer)

        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_feats if i < num_layers - 1 else out_feats)
            for i in range(num_layers)
        ])

    def _create_rgcn_layer(self,
                           in_feats: int,
                           out_feats: int,
                           num_rels: int,
                           regularizer: str,
                           num_bases: int,
                           self_loop: bool) -> HeteroGraphConv:
        """创建R-GCN层"""
        # 使用HeteroGraphConv包装RelGraphConv
        conv_dict = {}

        if self.etypes:
            for i, etype in enumerate(self.etypes):
                conv_dict[etype] = RelGraphConv(
                    in_feats,
                    out_feats,
                    num_rels,
                    regularizer=regularizer,
                    num_bases=num_bases,
                    self_loop=self_loop,
                    dropout=self.dropout
                )

        return HeteroGraphConv(conv_dict, aggregate='sum')

    def forward(self,
                blocks: List[dgl.DGLHeteroGraph],
                x: Dict[str, torch.Tensor],
                edge_types: Optional[Dict[Tuple, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            blocks: 采样得到的子图列表
            x: 输入节点特征
            edge_types: 边类型张量（用于RelGraphConv）

        Returns:
            输出节点特征
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

            # 归一化和激活
            h = {ntype: self.norms[i](feat) for ntype, feat in h.items()}

            if i < self.num_layers - 1:
                h = {ntype: F.relu(feat) for ntype, feat in h.items()}
                h = {ntype: F.dropout(feat, p=self.dropout, training=self.training)
                     for ntype, feat in h.items()}

        return h


class RGCNForNodeClassification(nn.Module):
    """用于节点分类的R-GCN"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 num_classes: int,
                 num_rels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 target_ntype: str = 'user',
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化模型

        Args:
            in_feats: 各节点类型的输入特征维度
            hidden_feats: 隐藏层特征维度
            num_classes: 分类类别数
            num_rels: 关系类型数量
            num_layers: GNN层数
            dropout: Dropout比例
            target_ntype: 目标节点类型
            etypes: 边类型列表
        """
        super().__init__()

        self.target_ntype = target_ntype

        # R-GCN编码器
        self.rgcn = RGCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_rels=num_rels,
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
        """前向传播"""
        h = self.rgcn(blocks, x)

        if self.target_ntype in h:
            return self.classifier(h[self.target_ntype])
        else:
            raise ValueError(f"目标节点类型 {self.target_ntype} 不在输出中")


class EntityClassifier(nn.Module):
    """实体分类器 - 可用于多种节点类型的分类"""

    def __init__(self,
                 in_feats: int,
                 num_classes: int,
                 hidden_feats: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        初始化分类器

        Args:
            in_feats: 输入特征维度
            num_classes: 类别数
            hidden_feats: 隐藏层维度
            num_layers: MLP层数
            dropout: Dropout比例
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_feats, hidden_feats))
            else:
                layers.append(nn.Linear(hidden_feats, hidden_feats))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_feats, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.mlp(x)
