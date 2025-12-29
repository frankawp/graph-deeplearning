"""
TorchServe模型Handler
处理异构图神经网络的在线推理请求
"""

import os
import json
import torch
import dgl
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HeteroGNNHandler:
    """异构图神经网络推理Handler"""

    def __init__(self):
        """初始化Handler"""
        self.model = None
        self.config = None
        self.feature_service = None
        self.graph_service = None
        self.initialized = False

    def initialize(self, context: Dict):
        """
        初始化模型和服务

        Args:
            context: TorchServe上下文，包含模型目录等信息
        """
        properties = context.get('system_properties', {})
        model_dir = properties.get('model_dir', '.')

        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # 加载模型
        model_path = os.path.join(model_dir, 'model.pt')
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()

        # 初始化特征服务
        if self.config.get('redis_host'):
            from .feature_sync import FeatureSyncService
            self.feature_service = FeatureSyncService(
                redis_host=self.config['redis_host'],
                redis_port=self.config.get('redis_port', 6379)
            )

        # 初始化图查询服务
        if self.config.get('graph_service_endpoints'):
            from .graph_query_service import GraphQueryService
            self.graph_service = GraphQueryService(
                endpoints=self.config['graph_service_endpoints']
            )

        self.initialized = True
        logger.info("Handler初始化完成")

    def preprocess(self, data: List[Dict]) -> Dict:
        """
        预处理推理请求

        Args:
            data: 请求数据列表

        Returns:
            预处理后的批数据
        """
        batch = {
            'node_ids': [],
            'node_types': [],
            'task_type': None
        }

        for item in data:
            body = item.get('body', item)
            if isinstance(body, bytes):
                body = json.loads(body.decode('utf-8'))
            elif isinstance(body, str):
                body = json.loads(body)

            batch['node_ids'].append(body['node_id'])
            batch['node_types'].append(body.get('node_type', 'user'))
            batch['task_type'] = body.get('task_type', 'node_classify')

        return batch

    def inference(self, batch: Dict) -> List[Dict]:
        """
        执行推理

        Args:
            batch: 预处理后的批数据

        Returns:
            推理结果列表
        """
        node_ids = batch['node_ids']
        node_types = batch['node_types']
        task_type = batch['task_type']

        results = []

        for node_id, node_type in zip(node_ids, node_types):
            try:
                # 获取节点特征
                features = self._get_node_features(node_id, node_type)

                # 获取邻居子图
                subgraph = self._get_subgraph(node_id, node_type)

                # 构建mini-batch
                blocks = self._build_blocks(subgraph, node_id, node_type)

                # 模型推理
                with torch.no_grad():
                    if task_type == 'node_classify':
                        result = self._node_classification(blocks, features, node_type)
                    elif task_type == 'link_predict':
                        result = self._link_prediction(blocks, features, batch)
                    else:
                        result = {'error': f'未知的任务类型: {task_type}'}

                result['node_id'] = node_id
                result['node_type'] = node_type
                results.append(result)

            except Exception as e:
                logger.error(f"推理失败 node_id={node_id}: {e}")
                results.append({
                    'node_id': node_id,
                    'node_type': node_type,
                    'error': str(e)
                })

        return results

    def _get_node_features(self, node_id: int, node_type: str) -> torch.Tensor:
        """获取节点特征"""
        if self.feature_service:
            features = self.feature_service.get_features([node_id], node_type)
            return torch.tensor(features, dtype=torch.float32)
        else:
            # 使用默认特征
            feature_dim = self.config.get('feature_dim', 256)
            return torch.zeros(1, feature_dim)

    def _get_subgraph(self, node_id: int, node_type: str) -> Dict:
        """获取节点的K跳邻居子图"""
        if self.graph_service:
            return self.graph_service.get_k_hop_neighbors(
                node_id=node_id,
                node_type=node_type,
                num_hops=self.config.get('num_hops', 2),
                max_neighbors=self.config.get('max_neighbors', [15, 10])
            )
        else:
            return {'nodes': {node_type: [node_id]}, 'edges': {}}

    def _build_blocks(self,
                      subgraph: Dict,
                      target_node_id: int,
                      target_node_type: str) -> List[dgl.DGLHeteroGraph]:
        """构建采样blocks"""
        # 简化实现：构建单层block
        nodes = subgraph.get('nodes', {})
        edges = subgraph.get('edges', {})

        # 构建DGL图
        edge_dict = {}
        for etype, edge_list in edges.items():
            if edge_list:
                src = torch.tensor([e[0] for e in edge_list])
                dst = torch.tensor([e[1] for e in edge_list])
                edge_dict[etype] = (src, dst)

        if edge_dict:
            num_nodes_dict = {ntype: len(nids) for ntype, nids in nodes.items()}
            block = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
        else:
            # 创建空图
            block = dgl.heterograph({
                (target_node_type, 'self', target_node_type): (
                    torch.tensor([0]), torch.tensor([0])
                )
            })

        return [block]

    def _node_classification(self,
                             blocks: List[dgl.DGLHeteroGraph],
                             features: torch.Tensor,
                             node_type: str) -> Dict:
        """节点分类推理"""
        # 构建输入
        x_dict = {node_type: features}

        # 模型前向传播
        logits = self.model(blocks, x_dict)

        # 处理输出
        if isinstance(logits, dict):
            logits = logits.get(node_type, logits[list(logits.keys())[0]])

        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        confidence = probs.max().item()

        return {
            'prediction': pred,
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }

    def _link_prediction(self,
                         blocks: List[dgl.DGLHeteroGraph],
                         features: torch.Tensor,
                         batch: Dict) -> Dict:
        """链接预测推理"""
        # 简化实现
        return {
            'link_score': 0.5,
            'link_exists': False
        }

    def postprocess(self, results: List[Dict]) -> List[str]:
        """
        后处理推理结果

        Args:
            results: 推理结果列表

        Returns:
            JSON字符串列表
        """
        return [json.dumps(result) for result in results]

    def handle(self, data: List[Dict], context: Dict) -> List[str]:
        """
        处理推理请求的主入口

        Args:
            data: 请求数据
            context: 上下文

        Returns:
            响应数据
        """
        if not self.initialized:
            self.initialize(context)

        batch = self.preprocess(data)
        results = self.inference(batch)
        return self.postprocess(results)


# TorchServe入口点
_service = HeteroGNNHandler()


def handle(data, context):
    """TorchServe调用的入口函数"""
    return _service.handle(data, context)
