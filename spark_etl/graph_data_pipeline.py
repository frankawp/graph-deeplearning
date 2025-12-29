"""
图数据ETL管道
从Hive提取异构图数据，转换为CGDF格式供DGL分布式训练使用
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, LongType, FloatType,
    ArrayType, StringType, IntegerType
)
from typing import Dict, List, Tuple, Optional
import json
import os


class GraphDataPipeline:
    """异构图数据ETL管道"""

    def __init__(self, spark: SparkSession, config: Dict):
        """
        初始化ETL管道

        Args:
            spark: SparkSession实例
            config: 配置字典，包含:
                - database: Hive数据库名
                - node_types: 节点类型列表
                - edge_types: 边类型列表 [(src_type, edge_name, dst_type), ...]
                - output_path: 输出路径
                - num_partitions: 分区数量（默认32）
        """
        self.spark = spark
        self.config = config
        self.database = config.get('database', 'graph_db')
        self.node_types = config.get('node_types', [])
        self.edge_types = config.get('edge_types', [])
        self.output_path = config.get('output_path', '/data/cgdf')
        self.num_partitions = config.get('num_partitions', 32)

        # 节点ID映射表
        self.node_id_mappings: Dict[str, DataFrame] = {}

    def run(self, dt: str) -> Dict:
        """
        运行完整的ETL流程

        Args:
            dt: 数据分区日期

        Returns:
            包含统计信息的字典
        """
        print(f"开始ETL流程，数据日期: {dt}")

        # Step 1: 提取节点数据
        print("Step 1: 提取节点数据...")
        node_dfs = self._extract_all_nodes(dt)

        # Step 2: 构建全局节点ID映射
        print("Step 2: 构建全局节点ID映射...")
        self._build_global_node_mapping(node_dfs)

        # Step 3: 提取边数据
        print("Step 3: 提取边数据...")
        edge_dfs = self._extract_all_edges(dt)

        # Step 4: 转换边的节点ID为全局ID
        print("Step 4: 转换边的节点ID...")
        edge_dfs = self._transform_edge_ids(edge_dfs)

        # Step 5: 导出为CGDF格式
        print("Step 5: 导出CGDF格式...")
        stats = self._export_cgdf(node_dfs, edge_dfs)

        print("ETL流程完成！")
        return stats

    def _extract_all_nodes(self, dt: str) -> Dict[str, DataFrame]:
        """提取所有类型的节点数据"""
        node_dfs = {}

        for node_type in self.node_types:
            table_name = f"{self.database}.nodes_{node_type}"
            df = self._extract_nodes(table_name, node_type, dt)
            node_dfs[node_type] = df
            count = df.count()
            print(f"  - {node_type}: {count:,} 节点")

        return node_dfs

    def _extract_nodes(self, table_name: str, node_type: str, dt: str) -> DataFrame:
        """
        从Hive提取单个类型的节点数据

        Args:
            table_name: Hive表名
            node_type: 节点类型
            dt: 数据分区日期
        """
        query = f"""
            SELECT
                node_id,
                feature_vec,
                label
            FROM {table_name}
            WHERE dt = '{dt}'
        """

        df = self.spark.sql(query)

        # 添加节点类型列
        df = df.withColumn('node_type', F.lit(node_type))

        return df

    def _extract_all_edges(self, dt: str) -> Dict[Tuple[str, str, str], DataFrame]:
        """提取所有类型的边数据"""
        edge_dfs = {}

        for edge_type in self.edge_types:
            src_type, edge_name, dst_type = edge_type
            table_name = f"{self.database}.edges_{edge_name}"
            df = self._extract_edges(table_name, edge_type, dt)
            edge_dfs[edge_type] = df
            count = df.count()
            print(f"  - {edge_type}: {count:,} 边")

        return edge_dfs

    def _extract_edges(self, table_name: str,
                       edge_type: Tuple[str, str, str],
                       dt: str) -> DataFrame:
        """
        从Hive提取单个类型的边数据

        Args:
            table_name: Hive表名
            edge_type: (源节点类型, 边类型, 目标节点类型)
            dt: 数据分区日期
        """
        src_type, edge_name, dst_type = edge_type

        query = f"""
            SELECT
                src_node_id,
                dst_node_id,
                edge_weight
            FROM {table_name}
            WHERE dt = '{dt}'
              AND src_type = '{src_type}'
              AND dst_type = '{dst_type}'
        """

        df = self.spark.sql(query)

        return df

    def _build_global_node_mapping(self, node_dfs: Dict[str, DataFrame]):
        """
        构建全局节点ID映射
        将异构图中不同类型的节点映射到统一的全局ID空间
        """
        offset = 0

        for node_type in self.node_types:
            df = node_dfs[node_type]

            # 为每个节点分配全局ID
            window = Window.orderBy('node_id')
            mapping_df = df.select('node_id').distinct().withColumn(
                'global_node_id',
                F.row_number().over(window) + F.lit(offset) - 1
            )

            # 缓存映射表
            mapping_df = mapping_df.cache()
            self.node_id_mappings[node_type] = mapping_df

            count = mapping_df.count()
            print(f"  - {node_type}: 全局ID范围 [{offset}, {offset + count - 1}]")
            offset += count

    def _transform_edge_ids(self,
                            edge_dfs: Dict[Tuple[str, str, str], DataFrame]
                           ) -> Dict[Tuple[str, str, str], DataFrame]:
        """将边的原始节点ID转换为全局节点ID"""
        transformed = {}

        for edge_type, df in edge_dfs.items():
            src_type, edge_name, dst_type = edge_type

            # 获取源节点和目标节点的ID映射
            src_mapping = self.node_id_mappings[src_type]
            dst_mapping = self.node_id_mappings[dst_type]

            # 转换源节点ID
            df = df.join(
                src_mapping.withColumnRenamed('node_id', 'src_node_id')
                          .withColumnRenamed('global_node_id', 'src_global_id'),
                on='src_node_id'
            )

            # 转换目标节点ID
            df = df.join(
                dst_mapping.withColumnRenamed('node_id', 'dst_node_id')
                          .withColumnRenamed('global_node_id', 'dst_global_id'),
                on='dst_node_id'
            )

            # 只保留需要的列
            df = df.select(
                F.col('src_global_id').alias('src'),
                F.col('dst_global_id').alias('dst'),
                'edge_weight'
            )

            transformed[edge_type] = df

        return transformed

    def _export_cgdf(self,
                     node_dfs: Dict[str, DataFrame],
                     edge_dfs: Dict[Tuple[str, str, str], DataFrame]) -> Dict:
        """
        导出为CGDF (Chunked Graph Data Format) 格式

        CGDF格式结构:
        output_path/
        ├── node_data/
        │   ├── {node_type}/
        │   │   ├── part-0.parquet
        │   │   └── ...
        ├── edge_data/
        │   ├── {src_type}_{edge_name}_{dst_type}/
        │   │   ├── part-0.parquet
        │   │   └── ...
        └── metadata.json
        """
        stats = {
            'nodes': {},
            'edges': {},
            'total_nodes': 0,
            'total_edges': 0
        }

        # 导出节点数据
        node_data_path = os.path.join(self.output_path, 'node_data')
        for node_type, df in node_dfs.items():
            # 添加全局ID
            mapping = self.node_id_mappings[node_type]
            df = df.join(mapping, on='node_id')

            # 导出
            output = os.path.join(node_data_path, node_type)
            df.select(
                'global_node_id',
                'feature_vec',
                'label'
            ).repartition(self.num_partitions).write.mode('overwrite').parquet(output)

            count = df.count()
            stats['nodes'][node_type] = count
            stats['total_nodes'] += count
            print(f"  - 导出节点 {node_type}: {count:,}")

        # 导出边数据
        edge_data_path = os.path.join(self.output_path, 'edge_data')
        for edge_type, df in edge_dfs.items():
            src_type, edge_name, dst_type = edge_type
            type_str = f"{src_type}_{edge_name}_{dst_type}"

            output = os.path.join(edge_data_path, type_str)
            df.repartition(self.num_partitions).write.mode('overwrite').parquet(output)

            count = df.count()
            stats['edges'][type_str] = count
            stats['total_edges'] += count
            print(f"  - 导出边 {type_str}: {count:,}")

        # 导出元数据
        self._export_metadata(stats)

        return stats

    def _export_metadata(self, stats: Dict):
        """导出元数据文件"""
        metadata = {
            'node_types': self.node_types,
            'edge_types': [
                {'src_type': e[0], 'edge_name': e[1], 'dst_type': e[2]}
                for e in self.edge_types
            ],
            'num_partitions': self.num_partitions,
            'stats': stats,
            'node_id_ranges': {}
        }

        # 添加节点ID范围信息
        offset = 0
        for node_type in self.node_types:
            count = stats['nodes'][node_type]
            metadata['node_id_ranges'][node_type] = {
                'start': offset,
                'end': offset + count - 1,
                'count': count
            }
            offset += count

        # 写入JSON文件
        metadata_path = os.path.join(self.output_path, 'metadata.json')

        # 使用Spark写入（避免驱动程序写入问题）
        metadata_df = self.spark.createDataFrame(
            [(json.dumps(metadata),)],
            ['metadata']
        )
        metadata_df.coalesce(1).write.mode('overwrite').text(
            os.path.join(self.output_path, 'metadata_temp')
        )

        print(f"  - 元数据已导出")


def main():
    """ETL主入口"""
    # 创建SparkSession
    spark = SparkSession.builder \
        .appName("GraphDataPipeline") \
        .enableHiveSupport() \
        .getOrCreate()

    # 配置
    config = {
        'database': 'graph_db',
        'node_types': ['user', 'item'],
        'edge_types': [
            ('user', 'click', 'item'),
            ('user', 'purchase', 'item'),
            ('item', 'similar', 'item'),
        ],
        'output_path': '/data/cgdf',
        'num_partitions': 32
    }

    # 运行ETL
    pipeline = GraphDataPipeline(spark, config)
    stats = pipeline.run(dt='2024-01-01')

    print("\n=== ETL统计 ===")
    print(f"总节点数: {stats['total_nodes']:,}")
    print(f"总边数: {stats['total_edges']:,}")

    spark.stop()


if __name__ == '__main__':
    main()
