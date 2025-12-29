"""
异构图GAT模型
使用图注意力机制的异构图神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv, GATConv
from typing import Dict, List, Tuple, Optional


class HeteroGAT(nn.Module):
    """异构图GAT模型"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 out_feats: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.5,
                 attn_dropout: float = 0.1,
                 negative_slope: float = 0.2,
                 residual: bool = True,
                 etypes: Optional[List[Tuple[str, str, str]]] = None):
        """
        初始化模型

        Args:
            in_feats: 各节点类型的输入特征维度 {ntype: dim}
            hidden_feats: 隐藏层特征维度
            out_feats: 输出特征维度
            num_layers: GNN层数
            num_heads: 注意力头数
            dropout: Dropout比例
            attn_dropout: 注意力Dropout比例
            negative_slope: LeakyReLU负斜率
            residual: 是否使用残差连接
            etypes: 边类型列表
        """
        super().__init__()

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.etypes = etypes

        # 输入投影层
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_feats.items():
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_feats)

        # 异构图注意力层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_feats
            else:
                in_dim = hidden_feats * num_heads

            if i == num_layers - 1:
                # 最后一层
                layer = self._create_gat_layer(
                    in_dim, out_feats, 1,  # 最后一层使用1个头
                    attn_dropout, negative_slope, residual=False
                )
            else:
                layer = self._create_gat_layer(
                    in_dim, hidden_feats, num_heads,
                    attn_dropout, negative_slope, residual
                )
            self.layers.append(layer)

        # 层归一化
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(hidden_feats * num_heads))

    def _create_gat_layer(self,
                          in_feats: int,
                          out_feats: int,
                          num_heads: int,
                          attn_dropout: float,
                          negative_slope: float,
                          residual: bool) -> HeteroGraphConv:
        """创建异构图注意力层"""
        conv_dict = {}

        if self.etypes:
            for etype in self.etypes:
                conv_dict[etype] = GATConv(
                    in_feats,
                    out_feats,
                    num_heads,
                    feat_drop=self.dropout,
                    attn_drop=attn_dropout,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=None,
                    allow_zero_in_degree=True
                )

        return HeteroGraphConv(conv_dict, aggregate='sum')

    def forward(self,
                blocks: List[dgl.DGLHeteroGraph],
                x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            blocks: 采样得到的子图列表
            x: 输入节点特征

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

            # 展平多头输出
            h = {ntype: feat.flatten(1) for ntype, feat in h.items()}

            # 归一化和激活（最后一层不激活）
            if i < self.num_layers - 1:
                h = {ntype: self.norms[i](feat) for ntype, feat in h.items()}
                h = {ntype: F.elu(feat) for ntype, feat in h.items()}
                h = {ntype: F.dropout(feat, p=self.dropout, training=self.training)
                     for ntype, feat in h.items()}

        return h


class HeteroGATForNodeClassification(nn.Module):
    """用于节点分类的异构GAT"""

    def __init__(self,
                 in_feats: Dict[str, int],
                 hidden_feats: int,
                 num_classes: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
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
            num_heads: 注意力头数
            dropout: Dropout比例
            target_ntype: 目标节点类型
            etypes: 边类型列表
        """
        super().__init__()

        self.target_ntype = target_ntype

        # GAT编码器
        self.gat = HeteroGAT(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=hidden_feats,
            num_layers=num_layers,
            num_heads=num_heads,
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
        h = self.gat(blocks, x)

        if self.target_ntype in h:
            return self.classifier(h[self.target_ntype])
        else:
            raise ValueError(f"目标节点类型 {self.target_ntype} 不在输出中")
