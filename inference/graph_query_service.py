"""
图查询服务
提供K跳邻居查询功能，支持在线推理时的子图获取
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class GraphQueryService:
    """图查询服务"""

    def __init__(self,
                 endpoints: Optional[List[str]] = None,
                 partition_path: Optional[str] = None,
                 num_partitions: int = 32):
        """
        初始化图查询服务

        Args:
            endpoints: 分布式图服务端点列表
            partition_path: 本地分区数据路径
            num_partitions: 分区数量
        """
        self.endpoints = endpoints or []
        self.partition_path = partition_path
        self.num_partitions = num_partitions

        # 本地图数据缓存
        self.local_graph_data = {}

        # 加载本地数据（如果有）
        if partition_path:
            self._load_local_graph()

    def _load_local_graph(self):
        """加载本地图数据"""
        if not self.partition_path or not os.path.exists(self.partition_path):
            return

        try:
            import pyarrow.parquet as pq

            # 加载边数据
            edge_data_path = os.path.join(self.partition_path, 'edge_data')
            if os.path.exists(edge_data_path):
                for etype_dir in os.listdir(edge_data_path):
                    etype_path = os.path.join(edge_data_path, etype_dir)
                    if os.path.isdir(etype_path):
                        table = pq.read_table(etype_path)
                        src = table['src_id'].to_numpy()
                        dst = table['dst_id'].to_numpy()

                        # 构建邻接表
                        self.local_graph_data[etype_dir] = {
                            'src': src,
                            'dst': dst,
                            'adj_list': self._build_adj_list(src, dst)
                        }

            logger.info(f"已加载本地图数据: {len(self.local_graph_data)} 种边类型")

        except Exception as e:
            logger.error(f"加载本地图数据失败: {e}")

    def _build_adj_list(self,
                        src: np.ndarray,
                        dst: np.ndarray) -> Dict[int, List[int]]:
        """构建邻接表"""
        adj_list = {}
        for s, d in zip(src, dst):
            if s not in adj_list:
                adj_list[s] = []
            adj_list[s].append(d)
        return adj_list

    def get_k_hop_neighbors(self,
                            node_id: int,
                            node_type: str,
                            num_hops: int = 2,
                            max_neighbors: List[int] = [15, 10]) -> Dict:
        """
        获取K跳邻居

        Args:
            node_id: 目标节点ID
            node_type: 节点类型
            num_hops: 跳数
            max_neighbors: 每跳的最大邻居数

        Returns:
            子图数据 {'nodes': {...}, 'edges': {...}}
        """
        if self.endpoints:
            return self._query_remote(node_id, node_type, num_hops, max_neighbors)
        else:
            return self._query_local(node_id, node_type, num_hops, max_neighbors)

    def _query_local(self,
                     node_id: int,
                     node_type: str,
                     num_hops: int,
                     max_neighbors: List[int]) -> Dict:
        """本地查询K跳邻居"""
        all_nodes = {node_type: {node_id}}
        all_edges = {}

        current_frontier = {node_type: {node_id}}

        for hop in range(num_hops):
            max_n = max_neighbors[hop] if hop < len(max_neighbors) else max_neighbors[-1]
            next_frontier = {}

            for etype, data in self.local_graph_data.items():
                adj_list = data['adj_list']
                edge_list = []

                # 解析边类型
                parts = etype.split('__')
                if len(parts) == 3:
                    src_type, rel_type, dst_type = parts
                else:
                    continue

                # 只处理当前frontier中节点类型匹配的边
                if src_type not in current_frontier:
                    continue

                for src_node in current_frontier[src_type]:
                    if src_node in adj_list:
                        neighbors = adj_list[src_node]

                        # 采样
                        if len(neighbors) > max_n:
                            sampled = np.random.choice(neighbors, max_n, replace=False)
                        else:
                            sampled = neighbors

                        for dst_node in sampled:
                            edge_list.append((src_node, dst_node))

                            # 添加到下一层frontier
                            if dst_type not in next_frontier:
                                next_frontier[dst_type] = set()
                            next_frontier[dst_type].add(dst_node)

                            # 添加到所有节点
                            if dst_type not in all_nodes:
                                all_nodes[dst_type] = set()
                            all_nodes[dst_type].add(dst_node)

                if edge_list:
                    if etype not in all_edges:
                        all_edges[etype] = []
                    all_edges[etype].extend(edge_list)

            current_frontier = next_frontier

        # 转换set为list
        return {
            'nodes': {ntype: list(nodes) for ntype, nodes in all_nodes.items()},
            'edges': all_edges
        }

    def _query_remote(self,
                      node_id: int,
                      node_type: str,
                      num_hops: int,
                      max_neighbors: List[int]) -> Dict:
        """远程查询K跳邻居"""
        import requests

        # 确定节点所在分区
        partition_id = self._get_partition(node_id)

        # 发送查询请求
        endpoint = self.endpoints[partition_id % len(self.endpoints)]
        url = f"{endpoint}/query/neighbors"

        payload = {
            'node_id': node_id,
            'node_type': node_type,
            'num_hops': num_hops,
            'max_neighbors': max_neighbors
        }

        try:
            response = requests.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"远程查询失败: {e}")
            return {'nodes': {node_type: [node_id]}, 'edges': {}}

    def _get_partition(self, node_id: int) -> int:
        """获取节点所属分区"""
        return abs(hash(node_id)) % self.num_partitions


class GraphQueryServer:
    """图查询服务器 - 用于部署在各分区节点上"""

    def __init__(self,
                 partition_id: int,
                 partition_path: str,
                 host: str = '0.0.0.0',
                 port: int = 8080):
        """
        初始化服务器

        Args:
            partition_id: 分区ID
            partition_path: 分区数据路径
            host: 监听地址
            port: 监听端口
        """
        self.partition_id = partition_id
        self.partition_path = partition_path
        self.host = host
        self.port = port

        # 加载分区数据
        self.graph_service = GraphQueryService(
            partition_path=partition_path
        )

    def start(self):
        """启动服务器"""
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/query/neighbors', methods=['POST'])
        def query_neighbors():
            data = request.json
            result = self.graph_service.get_k_hop_neighbors(
                node_id=data['node_id'],
                node_type=data['node_type'],
                num_hops=data.get('num_hops', 2),
                max_neighbors=data.get('max_neighbors', [15, 10])
            )
            return jsonify(result)

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'partition_id': self.partition_id})

        app.run(host=self.host, port=self.port)


def main():
    """启动图查询服务器"""
    import argparse

    parser = argparse.ArgumentParser(description='图查询服务器')
    parser.add_argument('--partition-id', type=int, required=True)
    parser.add_argument('--partition-path', type=str, required=True)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    server = GraphQueryServer(
        partition_id=args.partition_id,
        partition_path=args.partition_path,
        host=args.host,
        port=args.port
    )
    server.start()


if __name__ == '__main__':
    main()
