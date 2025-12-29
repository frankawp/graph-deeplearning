"""
特征同步服务
将离线训练的节点特征/Embedding同步到在线Redis缓存
"""

import json
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureSyncService:
    """特征同步服务"""

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None,
                 key_prefix: str = 'gnn:feat',
                 ttl: int = 86400):
        """
        初始化特征同步服务

        Args:
            redis_host: Redis主机
            redis_port: Redis端口
            redis_db: Redis数据库号
            redis_password: Redis密码
            key_prefix: 键前缀
            ttl: 过期时间（秒）
        """
        import redis

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False  # 返回bytes
        )
        self.key_prefix = key_prefix
        self.ttl = ttl

        # 测试连接
        try:
            self.redis_client.ping()
            logger.info(f"Redis连接成功: {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def _make_key(self, node_type: str, node_id: int) -> str:
        """构建Redis键"""
        return f"{self.key_prefix}:{node_type}:{node_id}"

    def set_feature(self,
                    node_id: int,
                    node_type: str,
                    feature: Union[List[float], np.ndarray]) -> bool:
        """
        设置单个节点的特征

        Args:
            node_id: 节点ID
            node_type: 节点类型
            feature: 特征向量

        Returns:
            是否成功
        """
        key = self._make_key(node_type, node_id)

        if isinstance(feature, np.ndarray):
            feature = feature.tolist()

        value = json.dumps(feature)

        try:
            self.redis_client.setex(key, self.ttl, value)
            return True
        except Exception as e:
            logger.error(f"设置特征失败 {key}: {e}")
            return False

    def get_feature(self,
                    node_id: int,
                    node_type: str) -> Optional[List[float]]:
        """
        获取单个节点的特征

        Args:
            node_id: 节点ID
            node_type: 节点类型

        Returns:
            特征向量，不存在则返回None
        """
        key = self._make_key(node_type, node_id)

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取特征失败 {key}: {e}")
            return None

    def set_features_batch(self,
                           node_ids: List[int],
                           node_type: str,
                           features: Union[List[List[float]], np.ndarray]) -> int:
        """
        批量设置节点特征

        Args:
            node_ids: 节点ID列表
            node_type: 节点类型
            features: 特征矩阵

        Returns:
            成功设置的数量
        """
        if isinstance(features, np.ndarray):
            features = features.tolist()

        pipe = self.redis_client.pipeline()
        count = 0

        for node_id, feature in zip(node_ids, features):
            key = self._make_key(node_type, node_id)
            value = json.dumps(feature)
            pipe.setex(key, self.ttl, value)
            count += 1

            # 每1000个执行一次
            if count % 1000 == 0:
                pipe.execute()
                pipe = self.redis_client.pipeline()

        # 执行剩余的
        if count % 1000 != 0:
            pipe.execute()

        return count

    def get_features(self,
                     node_ids: List[int],
                     node_type: str,
                     default_dim: int = 256) -> np.ndarray:
        """
        批量获取节点特征

        Args:
            node_ids: 节点ID列表
            node_type: 节点类型
            default_dim: 默认特征维度（用于缺失特征）

        Returns:
            特征矩阵
        """
        keys = [self._make_key(node_type, nid) for nid in node_ids]

        try:
            values = self.redis_client.mget(keys)

            features = []
            for value in values:
                if value:
                    features.append(json.loads(value))
                else:
                    # 使用零向量作为默认特征
                    features.append([0.0] * default_dim)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"批量获取特征失败: {e}")
            return np.zeros((len(node_ids), default_dim), dtype=np.float32)

    def delete_feature(self, node_id: int, node_type: str) -> bool:
        """删除单个节点的特征"""
        key = self._make_key(node_type, node_id)
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"删除特征失败 {key}: {e}")
            return False

    def sync_from_parquet(self,
                          parquet_path: str,
                          node_type: str,
                          id_col: str = 'node_id',
                          feature_col: str = 'feat',
                          batch_size: int = 10000) -> int:
        """
        从Parquet文件同步特征到Redis

        Args:
            parquet_path: Parquet文件路径
            node_type: 节点类型
            id_col: ID列名
            feature_col: 特征列名
            batch_size: 批大小

        Returns:
            同步的节点数量
        """
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        total_count = 0

        for i in range(0, len(table), batch_size):
            batch = table.slice(i, min(batch_size, len(table) - i))

            node_ids = batch[id_col].to_pylist()
            features = batch[feature_col].to_pylist()

            count = self.set_features_batch(node_ids, node_type, features)
            total_count += count

            logger.info(f"已同步 {total_count} 个节点特征")

        return total_count

    def sync_from_hive(self,
                       spark,
                       table_name: str,
                       node_type: str,
                       id_col: str = 'node_id',
                       feature_col: str = 'embedding_vec',
                       batch_size: int = 10000) -> int:
        """
        从Hive表同步特征到Redis

        Args:
            spark: SparkSession
            table_name: Hive表名
            node_type: 节点类型
            id_col: ID列名
            feature_col: 特征列名
            batch_size: 批大小

        Returns:
            同步的节点数量
        """
        df = spark.sql(f"""
            SELECT {id_col}, {feature_col}
            FROM {table_name}
            WHERE node_type = '{node_type}'
        """)

        # 使用foreach批量写入Redis
        total_count = 0

        def sync_partition(partition):
            count = 0
            batch_ids = []
            batch_features = []

            for row in partition:
                batch_ids.append(row[id_col])
                batch_features.append(row[feature_col])

                if len(batch_ids) >= batch_size:
                    self.set_features_batch(batch_ids, node_type, batch_features)
                    count += len(batch_ids)
                    batch_ids = []
                    batch_features = []

            # 处理剩余
            if batch_ids:
                self.set_features_batch(batch_ids, node_type, batch_features)
                count += len(batch_ids)

            return [count]

        counts = df.rdd.mapPartitions(sync_partition).collect()
        total_count = sum(counts)

        logger.info(f"从Hive同步完成: {total_count} 个节点特征")
        return total_count

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            info = self.redis_client.info('memory')
            keys_count = self.redis_client.dbsize()

            return {
                'total_keys': keys_count,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'peak_memory': info.get('used_memory_peak_human', 'N/A')
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


def main():
    """特征同步主入口"""
    import argparse

    parser = argparse.ArgumentParser(description='特征同步服务')
    parser.add_argument('--redis-host', type=str, default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--parquet-path', type=str, required=True)
    parser.add_argument('--node-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=10000)

    args = parser.parse_args()

    service = FeatureSyncService(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )

    count = service.sync_from_parquet(
        parquet_path=args.parquet_path,
        node_type=args.node_type,
        batch_size=args.batch_size
    )

    print(f"同步完成: {count} 个节点")
    print(f"统计信息: {service.get_stats()}")


if __name__ == '__main__':
    main()
