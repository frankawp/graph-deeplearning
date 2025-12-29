"""
异构图构建器
将ETL处理后的数据构建为DGL异构图对象
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class HeteroGraphBuilder:
    """异构图构建器"""

    def __init__(self, spark: SparkSession):
        """
        初始化构建器

        Args:
            spark: SparkSession实例
        """
        self.spark = spark

    def build_edge_index_from_df(self,
                                  edges_df: DataFrame,
                                  src_col: str = 'src',
                                  dst_col: str = 'dst') -> DataFrame:
        """
        从DataFrame构建边索引

        Args:
            edges_df: 边数据DataFrame
            src_col: 源节点列名
            dst_col: 目标节点列名

        Returns:
            包含src, dst列的DataFrame
        """
        return edges_df.select(
            F.col(src_col).alias('src'),
            F.col(dst_col).alias('dst')
        )

    def add_reverse_edges(self,
                          edges_df: DataFrame,
                          edge_type: Tuple[str, str, str]) -> Tuple[DataFrame, Tuple[str, str, str]]:
        """
        为边添加反向边

        Args:
            edges_df: 边数据DataFrame
            edge_type: (src_type, edge_name, dst_type)

        Returns:
            (反向边DataFrame, 反向边类型)
        """
        src_type, edge_name, dst_type = edge_type

        # 创建反向边
        reverse_df = edges_df.select(
            F.col('dst').alias('src'),
            F.col('src').alias('dst'),
            F.col('edge_weight')
        )

        # 反向边类型
        reverse_type = (dst_type, f'rev_{edge_name}', src_type)

        return reverse_df, reverse_type

    def compute_node_degrees(self,
                             edges_df: DataFrame,
                             direction: str = 'out') -> DataFrame:
        """
        计算节点度数

        Args:
            edges_df: 边数据DataFrame
            direction: 'out'=出度, 'in'=入度, 'both'=总度数

        Returns:
            包含node_id, degree列的DataFrame
        """
        if direction == 'out':
            degree_df = edges_df.groupBy('src').agg(
                F.count('*').alias('degree')
            ).withColumnRenamed('src', 'node_id')
        elif direction == 'in':
            degree_df = edges_df.groupBy('dst').agg(
                F.count('*').alias('degree')
            ).withColumnRenamed('dst', 'node_id')
        else:  # both
            out_degree = edges_df.groupBy('src').agg(
                F.count('*').alias('out_degree')
            ).withColumnRenamed('src', 'node_id')

            in_degree = edges_df.groupBy('dst').agg(
                F.count('*').alias('in_degree')
            ).withColumnRenamed('dst', 'node_id')

            degree_df = out_degree.join(in_degree, on='node_id', how='outer').fillna(0)
            degree_df = degree_df.withColumn(
                'degree',
                F.col('out_degree') + F.col('in_degree')
            )

        return degree_df

    def identify_super_nodes(self,
                             edges_df: DataFrame,
                             threshold: int = 10000) -> DataFrame:
        """
        识别超级节点（度数超过阈值的节点）

        Args:
            edges_df: 边数据DataFrame
            threshold: 度数阈值

        Returns:
            超级节点DataFrame
        """
        degree_df = self.compute_node_degrees(edges_df, direction='both')

        super_nodes = degree_df.filter(F.col('degree') > threshold)

        return super_nodes

    def sample_edges(self,
                     edges_df: DataFrame,
                     sample_ratio: float = 0.1,
                     seed: int = 42) -> DataFrame:
        """
        对边进行采样

        Args:
            edges_df: 边数据DataFrame
            sample_ratio: 采样比例
            seed: 随机种子

        Returns:
            采样后的边DataFrame
        """
        return edges_df.sample(fraction=sample_ratio, seed=seed)

    def split_train_val_test(self,
                             nodes_df: DataFrame,
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             seed: int = 42) -> DataFrame:
        """
        划分训练/验证/测试集

        Args:
            nodes_df: 节点DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子

        Returns:
            添加了train_mask, val_mask, test_mask列的DataFrame
        """
        # 添加随机数列
        df = nodes_df.withColumn('rand', F.rand(seed))

        # 划分
        train_threshold = train_ratio
        val_threshold = train_ratio + val_ratio

        df = df.withColumn(
            'train_mask',
            F.when(F.col('rand') < train_threshold, True).otherwise(False)
        ).withColumn(
            'val_mask',
            F.when(
                (F.col('rand') >= train_threshold) & (F.col('rand') < val_threshold),
                True
            ).otherwise(False)
        ).withColumn(
            'test_mask',
            F.when(F.col('rand') >= val_threshold, True).otherwise(False)
        )

        # 删除随机数列
        df = df.drop('rand')

        return df

    def normalize_features(self,
                           nodes_df: DataFrame,
                           feature_col: str = 'feature_vec') -> DataFrame:
        """
        对节点特征进行L2归一化

        Args:
            nodes_df: 节点DataFrame
            feature_col: 特征列名

        Returns:
            特征归一化后的DataFrame
        """
        from pyspark.ml.feature import Normalizer
        from pyspark.ml.linalg import Vectors, VectorUDT

        # 将Array转换为Vector
        array_to_vector = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
        df = nodes_df.withColumn('features_vec', array_to_vector(F.col(feature_col)))

        # 归一化
        normalizer = Normalizer(inputCol='features_vec', outputCol='normalized_features', p=2.0)
        df = normalizer.transform(df)

        # 转回Array
        vector_to_array = F.udf(lambda v: v.toArray().tolist(), F.ArrayType(F.FloatType()))
        df = df.withColumn(feature_col, vector_to_array(F.col('normalized_features')))

        # 删除临时列
        df = df.drop('features_vec', 'normalized_features')

        return df
