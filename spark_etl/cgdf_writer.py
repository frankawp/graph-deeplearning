"""
CGDF (Chunked Graph Data Format) 写入器
将处理后的图数据写入DGL可识别的CGDF格式
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Tuple, Optional
import json
import os


class CGDFWriter:
    """CGDF格式写入器"""

    def __init__(self,
                 spark: SparkSession,
                 output_path: str,
                 num_chunks: int = 32):
        """
        初始化CGDF写入器

        Args:
            spark: SparkSession实例
            output_path: 输出路径
            num_chunks: 分块数量（通常等于分区数）
        """
        self.spark = spark
        self.output_path = output_path
        self.num_chunks = num_chunks

        # 统计信息
        self.stats = {
            'node_types': {},
            'edge_types': {},
            'total_nodes': 0,
            'total_edges': 0
        }

    def write_nodes(self,
                    node_type: str,
                    nodes_df: DataFrame,
                    id_col: str = 'global_node_id',
                    feature_col: str = 'feature_vec',
                    label_col: str = 'label') -> int:
        """
        写入节点数据

        Args:
            node_type: 节点类型
            nodes_df: 节点DataFrame
            id_col: 节点ID列名
            feature_col: 特征列名
            label_col: 标签列名

        Returns:
            写入的节点数量
        """
        # 选择需要的列
        columns = [id_col]
        if feature_col in nodes_df.columns:
            columns.append(feature_col)
        if label_col in nodes_df.columns:
            columns.append(label_col)

        df = nodes_df.select(*columns)

        # 重命名列以符合CGDF规范
        df = df.withColumnRenamed(id_col, 'node_id')
        if feature_col in nodes_df.columns:
            df = df.withColumnRenamed(feature_col, 'feat')
        if label_col in nodes_df.columns:
            df = df.withColumnRenamed(label_col, 'label')

        # 写入
        output_dir = os.path.join(self.output_path, 'node_data', node_type)
        df.repartition(self.num_chunks).write.mode('overwrite').parquet(output_dir)

        count = df.count()
        self.stats['node_types'][node_type] = count
        self.stats['total_nodes'] += count

        return count

    def write_edges(self,
                    edge_type: Tuple[str, str, str],
                    edges_df: DataFrame,
                    src_col: str = 'src',
                    dst_col: str = 'dst',
                    weight_col: Optional[str] = 'edge_weight') -> int:
        """
        写入边数据

        Args:
            edge_type: (源类型, 边类型, 目标类型)
            edges_df: 边DataFrame
            src_col: 源节点列名
            dst_col: 目标节点列名
            weight_col: 边权重列名（可选）

        Returns:
            写入的边数量
        """
        src_type, rel_type, dst_type = edge_type

        # 选择需要的列
        columns = [src_col, dst_col]
        if weight_col and weight_col in edges_df.columns:
            columns.append(weight_col)

        df = edges_df.select(*columns)

        # 重命名列以符合CGDF规范
        df = df.withColumnRenamed(src_col, 'src_id')
        df = df.withColumnRenamed(dst_col, 'dst_id')
        if weight_col and weight_col in edges_df.columns:
            df = df.withColumnRenamed(weight_col, 'weight')

        # 写入
        type_str = f"{src_type}__{rel_type}__{dst_type}"
        output_dir = os.path.join(self.output_path, 'edge_data', type_str)
        df.repartition(self.num_chunks).write.mode('overwrite').parquet(output_dir)

        count = df.count()
        self.stats['edge_types'][type_str] = count
        self.stats['total_edges'] += count

        return count

    def write_train_mask(self,
                         node_type: str,
                         mask_df: DataFrame,
                         mask_type: str = 'train') -> int:
        """
        写入训练/验证/测试mask

        Args:
            node_type: 节点类型
            mask_df: 包含mask的DataFrame
            mask_type: 'train', 'val', 或 'test'

        Returns:
            mask中True的数量
        """
        mask_col = f'{mask_type}_mask'

        # 筛选mask为True的节点
        df = mask_df.filter(F.col(mask_col) == True).select('node_id')

        # 写入
        output_dir = os.path.join(
            self.output_path, 'node_data', node_type, f'{mask_type}_idx'
        )
        df.repartition(1).write.mode('overwrite').parquet(output_dir)

        return df.count()

    def write_metadata(self,
                       node_types: List[str],
                       edge_types: List[Tuple[str, str, str]],
                       feature_dims: Dict[str, int],
                       num_classes: Optional[int] = None):
        """
        写入CGDF元数据文件

        Args:
            node_types: 节点类型列表
            edge_types: 边类型列表
            feature_dims: 各节点类型的特征维度
            num_classes: 分类任务的类别数（可选）
        """
        metadata = {
            'graph_name': 'hetero_graph',
            'node_types': node_types,
            'edge_types': [
                {
                    'src_type': e[0],
                    'rel_type': e[1],
                    'dst_type': e[2],
                    'etype': f"{e[0]}__{e[1]}__{e[2]}"
                }
                for e in edge_types
            ],
            'num_nodes': self.stats['node_types'],
            'num_edges': self.stats['edge_types'],
            'total_nodes': self.stats['total_nodes'],
            'total_edges': self.stats['total_edges'],
            'feature_dims': feature_dims,
            'num_chunks': self.num_chunks
        }

        if num_classes:
            metadata['num_classes'] = num_classes

        # 写入JSON
        metadata_path = os.path.join(self.output_path, 'metadata.json')

        # 使用Spark DataFrame写入避免驱动程序限制
        metadata_json = json.dumps(metadata, indent=2)
        self.spark.sparkContext.parallelize([metadata_json]).coalesce(1).saveAsTextFile(
            os.path.join(self.output_path, 'metadata_temp')
        )

        print(f"元数据已写入: {metadata_path}")

    def get_stats(self) -> Dict:
        """获取写入统计信息"""
        return self.stats


class CGDFPartitionWriter:
    """CGDF分区写入器 - 用于超大规模图的分布式写入"""

    def __init__(self,
                 spark: SparkSession,
                 output_path: str,
                 num_partitions: int = 32):
        """
        初始化分区写入器

        Args:
            spark: SparkSession实例
            output_path: 输出路径
            num_partitions: 分区数量
        """
        self.spark = spark
        self.output_path = output_path
        self.num_partitions = num_partitions

    def partition_and_write(self,
                            nodes_df: DataFrame,
                            edges_df: DataFrame,
                            node_type: str,
                            edge_type: Tuple[str, str, str],
                            partition_col: str = 'partition_id'):
        """
        对图数据进行分区并写入

        Args:
            nodes_df: 节点DataFrame
            edges_df: 边DataFrame
            node_type: 节点类型
            edge_type: 边类型
            partition_col: 分区列名
        """
        # 为节点分配分区
        nodes_partitioned = nodes_df.withColumn(
            partition_col,
            F.abs(F.hash(F.col('node_id'))) % self.num_partitions
        )

        # 边跟随源节点的分区
        src_partitions = nodes_partitioned.select(
            F.col('node_id').alias('src'),
            F.col(partition_col)
        )

        edges_partitioned = edges_df.join(src_partitions, on='src')

        # 分区写入
        for pid in range(self.num_partitions):
            partition_path = os.path.join(self.output_path, f'part_{pid}')

            # 写入该分区的节点
            partition_nodes = nodes_partitioned.filter(
                F.col(partition_col) == pid
            ).drop(partition_col)

            partition_nodes.write.mode('overwrite').parquet(
                os.path.join(partition_path, 'node_data', node_type)
            )

            # 写入该分区的边
            partition_edges = edges_partitioned.filter(
                F.col(partition_col) == pid
            ).drop(partition_col)

            edge_str = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            partition_edges.write.mode('overwrite').parquet(
                os.path.join(partition_path, 'edge_data', edge_str)
            )

        print(f"已将数据分区写入 {self.num_partitions} 个分区")
