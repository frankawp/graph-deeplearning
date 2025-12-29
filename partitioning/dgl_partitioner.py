"""
DGL图分区器
封装DGL的分布式图分区功能
"""

import dgl
from dgl import DGLHeteroGraph
from dgl.distributed import partition_graph
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json


class DGLPartitioner:
    """DGL分布式图分区器"""

    def __init__(self,
                 num_partitions: int = 32,
                 part_method: str = 'random',
                 balance_edges: bool = True,
                 output_path: str = '/data/partitions'):
        """
        初始化分区器

        Args:
            num_partitions: 分区数量
            part_method: 分区方法
                - 'metis': METIS分区（适合中小规模图）
                - 'random': 随机分区（适合超大规模图）
            balance_edges: 是否平衡边数量
            output_path: 分区输出路径
        """
        self.num_partitions = num_partitions
        self.part_method = part_method
        self.balance_edges = balance_edges
        self.output_path = output_path

    def partition(self,
                  graph: DGLHeteroGraph,
                  graph_name: str = 'hetero_graph',
                  node_features: Optional[Dict[str, torch.Tensor]] = None,
                  edge_features: Optional[Dict[Tuple, torch.Tensor]] = None) -> str:
        """
        对图进行分区

        Args:
            graph: DGL异构图
            graph_name: 图名称
            node_features: 节点特征字典 {node_type: features}
            edge_features: 边特征字典 {edge_type: features}

        Returns:
            分区配置文件路径
        """
        print(f"开始图分区...")
        print(f"  - 分区数: {self.num_partitions}")
        print(f"  - 分区方法: {self.part_method}")
        print(f"  - 节点数: {graph.num_nodes()}")
        print(f"  - 边数: {graph.num_edges()}")

        # 添加特征到图
        if node_features:
            for ntype, feat in node_features.items():
                graph.nodes[ntype].data['feat'] = feat

        if edge_features:
            for etype, feat in edge_features.items():
                graph.edges[etype].data['feat'] = feat

        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)

        # 执行分区
        partition_graph(
            graph,
            graph_name=graph_name,
            num_parts=self.num_partitions,
            out_path=self.output_path,
            part_method=self.part_method,
            balance_edges=self.balance_edges,
            return_mapping=False
        )

        part_config = os.path.join(self.output_path, f'{graph_name}.json')
        print(f"分区完成！配置文件: {part_config}")

        return part_config

    def partition_large_graph(self,
                              edge_data: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]],
                              num_nodes_dict: Dict[str, int],
                              graph_name: str = 'hetero_graph',
                              chunk_size: int = 10000000) -> str:
        """
        分块处理超大规模图的分区

        Args:
            edge_data: 边数据 {etype: (src_ids, dst_ids)}
            num_nodes_dict: 节点数量 {ntype: count}
            graph_name: 图名称
            chunk_size: 每次处理的边数量

        Returns:
            分区配置文件路径
        """
        print(f"开始超大规模图分区（分块处理）...")

        # 对于超大规模图，使用随机分区
        # 先分配节点到分区
        node_partitions = {}
        for ntype, num_nodes in num_nodes_dict.items():
            # 使用随机分区
            partitions = np.random.randint(0, self.num_partitions, size=num_nodes)
            node_partitions[ntype] = partitions
            print(f"  - {ntype}: {num_nodes:,} 节点已分配分区")

        # 创建分区目录
        for pid in range(self.num_partitions):
            partition_dir = os.path.join(self.output_path, f'part{pid}')
            os.makedirs(partition_dir, exist_ok=True)

        # 按分区写入边数据
        for etype, (src_ids, dst_ids) in edge_data.items():
            src_type, rel_type, dst_type = etype
            print(f"  - 处理边类型 {etype}: {len(src_ids):,} 边")

            # 获取源节点的分区
            src_parts = node_partitions[src_type][src_ids]

            # 按分区分组边
            for pid in range(self.num_partitions):
                mask = src_parts == pid
                part_src = src_ids[mask]
                part_dst = dst_ids[mask]

                # 保存到分区目录
                partition_dir = os.path.join(self.output_path, f'part{pid}')
                edge_file = os.path.join(
                    partition_dir,
                    f'{src_type}_{rel_type}_{dst_type}_edges.npy'
                )
                np.save(edge_file, np.stack([part_src, part_dst], axis=0))

        # 写入分区元数据
        self._write_partition_metadata(graph_name, num_nodes_dict, node_partitions)

        part_config = os.path.join(self.output_path, f'{graph_name}.json')
        print(f"超大规模图分区完成！配置文件: {part_config}")

        return part_config

    def _write_partition_metadata(self,
                                   graph_name: str,
                                   num_nodes_dict: Dict[str, int],
                                   node_partitions: Dict[str, np.ndarray]):
        """写入分区元数据"""
        # 统计每个分区的节点数
        partition_stats = []
        for pid in range(self.num_partitions):
            stats = {'partition_id': pid, 'num_nodes': {}, 'num_edges': 0}
            for ntype, partitions in node_partitions.items():
                stats['num_nodes'][ntype] = int(np.sum(partitions == pid))
            partition_stats.append(stats)

        # 写入配置文件
        config = {
            'graph_name': graph_name,
            'num_parts': self.num_partitions,
            'part_method': self.part_method,
            'node_types': list(num_nodes_dict.keys()),
            'num_nodes': num_nodes_dict,
            'partition_stats': partition_stats,
            'part_path': self.output_path
        }

        config_path = os.path.join(self.output_path, f'{graph_name}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # 保存节点到分区的映射
        for ntype, partitions in node_partitions.items():
            mapping_path = os.path.join(self.output_path, f'{ntype}_partition_map.npy')
            np.save(mapping_path, partitions)


class HeteroGraphLoader:
    """从CGDF格式加载异构图"""

    def __init__(self, cgdf_path: str):
        """
        初始化加载器

        Args:
            cgdf_path: CGDF数据路径
        """
        self.cgdf_path = cgdf_path
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """加载元数据"""
        import pyarrow.parquet as pq

        metadata_path = os.path.join(self.cgdf_path, 'metadata.json')

        # 尝试直接读取JSON
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)

        # 尝试从Spark写入的格式读取
        temp_path = os.path.join(self.cgdf_path, 'metadata_temp')
        if os.path.exists(temp_path):
            for file in os.listdir(temp_path):
                if file.startswith('part-'):
                    with open(os.path.join(temp_path, file), 'r') as f:
                        return json.loads(f.read())

        raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

    def load_graph(self) -> DGLHeteroGraph:
        """
        加载完整图

        Returns:
            DGL异构图
        """
        import pyarrow.parquet as pq

        # 加载边数据构建图
        edge_data = {}

        edge_types = self.metadata.get('edge_types', [])
        for etype_info in edge_types:
            if isinstance(etype_info, dict):
                etype = etype_info.get('etype')
                src_type = etype_info.get('src_type')
                rel_type = etype_info.get('rel_type')
                dst_type = etype_info.get('dst_type')
            else:
                etype = f"{etype_info[0]}__{etype_info[1]}__{etype_info[2]}"
                src_type, rel_type, dst_type = etype_info

            edge_path = os.path.join(self.cgdf_path, 'edge_data', etype)

            if os.path.exists(edge_path):
                table = pq.read_table(edge_path)
                src_ids = table['src_id'].to_numpy()
                dst_ids = table['dst_id'].to_numpy()

                edge_data[(src_type, rel_type, dst_type)] = (
                    torch.from_numpy(src_ids),
                    torch.from_numpy(dst_ids)
                )

        # 获取节点数量
        num_nodes_dict = self.metadata.get('num_nodes', {})

        # 构建图
        graph = dgl.heterograph(edge_data, num_nodes_dict=num_nodes_dict)

        # 加载节点特征
        for ntype in self.metadata.get('node_types', []):
            node_path = os.path.join(self.cgdf_path, 'node_data', ntype)

            if os.path.exists(node_path):
                table = pq.read_table(node_path)

                if 'feat' in table.column_names:
                    # 将特征转换为Tensor
                    feat_list = table['feat'].to_pylist()
                    feat = torch.tensor(feat_list, dtype=torch.float32)
                    graph.nodes[ntype].data['feat'] = feat

                if 'label' in table.column_names:
                    labels = torch.from_numpy(table['label'].to_numpy())
                    graph.nodes[ntype].data['label'] = labels

        return graph
