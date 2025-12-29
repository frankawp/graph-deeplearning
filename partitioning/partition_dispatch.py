"""
分区分发器
将分区数据分发到各个集群节点
"""

import os
import subprocess
from typing import Dict, List, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class PartitionDispatcher:
    """分区数据分发器"""

    def __init__(self,
                 partition_path: str,
                 ip_config_path: str,
                 remote_path: str = '/data/graph_partitions',
                 ssh_user: str = 'root',
                 ssh_port: int = 22):
        """
        初始化分发器

        Args:
            partition_path: 本地分区数据路径
            ip_config_path: IP配置文件路径
            remote_path: 远程节点上的数据路径
            ssh_user: SSH用户名
            ssh_port: SSH端口
        """
        self.partition_path = partition_path
        self.remote_path = remote_path
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port

        # 读取节点列表
        self.nodes = self._load_node_list(ip_config_path)
        print(f"已加载 {len(self.nodes)} 个节点")

    def _load_node_list(self, ip_config_path: str) -> List[str]:
        """加载节点列表"""
        nodes = []
        with open(ip_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    nodes.append(line)
        return nodes

    def dispatch(self,
                 num_partitions: int,
                 partitions_per_node: int = 1,
                 max_workers: int = 8) -> Dict[str, List[int]]:
        """
        分发分区数据到各节点

        Args:
            num_partitions: 总分区数
            partitions_per_node: 每个节点承载的分区数
            max_workers: 并行传输的最大线程数

        Returns:
            节点到分区的映射 {node: [partition_ids]}
        """
        # 计算分区分配
        node_partitions = self._assign_partitions(num_partitions, partitions_per_node)

        print(f"开始分发 {num_partitions} 个分区到 {len(self.nodes)} 个节点...")

        # 并行分发
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for node, partition_ids in node_partitions.items():
                future = executor.submit(
                    self._dispatch_to_node,
                    node,
                    partition_ids
                )
                futures[future] = node

            for future in as_completed(futures):
                node = futures[future]
                try:
                    success = future.result()
                    results[node] = {
                        'partitions': node_partitions[node],
                        'success': success
                    }
                    status = "成功" if success else "失败"
                    print(f"  - {node}: {status}")
                except Exception as e:
                    results[node] = {
                        'partitions': node_partitions[node],
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  - {node}: 失败 - {e}")

        # 保存分配记录
        self._save_dispatch_record(node_partitions)

        return node_partitions

    def _assign_partitions(self,
                           num_partitions: int,
                           partitions_per_node: int) -> Dict[str, List[int]]:
        """分配分区到节点"""
        node_partitions = {}

        partition_id = 0
        for i, node in enumerate(self.nodes):
            node_partitions[node] = []
            for _ in range(partitions_per_node):
                if partition_id < num_partitions:
                    node_partitions[node].append(partition_id)
                    partition_id += 1

        return node_partitions

    def _dispatch_to_node(self, node: str, partition_ids: List[int]) -> bool:
        """
        将分区数据发送到指定节点

        Args:
            node: 节点主机名或IP
            partition_ids: 要发送的分区ID列表

        Returns:
            是否成功
        """
        try:
            # 创建远程目录
            mkdir_cmd = f"ssh -p {self.ssh_port} {self.ssh_user}@{node} 'mkdir -p {self.remote_path}'"
            subprocess.run(mkdir_cmd, shell=True, check=True, capture_output=True)

            # 传输分区数据
            for pid in partition_ids:
                local_partition = os.path.join(self.partition_path, f'part{pid}')
                if os.path.exists(local_partition):
                    # 使用rsync传输
                    rsync_cmd = (
                        f"rsync -avz -e 'ssh -p {self.ssh_port}' "
                        f"{local_partition}/ "
                        f"{self.ssh_user}@{node}:{self.remote_path}/part{pid}/"
                    )
                    subprocess.run(rsync_cmd, shell=True, check=True, capture_output=True)

            # 传输分区配置文件
            config_files = [f for f in os.listdir(self.partition_path)
                           if f.endswith('.json') or f.endswith('.npy')]
            for config_file in config_files:
                local_file = os.path.join(self.partition_path, config_file)
                scp_cmd = (
                    f"scp -P {self.ssh_port} {local_file} "
                    f"{self.ssh_user}@{node}:{self.remote_path}/"
                )
                subprocess.run(scp_cmd, shell=True, check=True, capture_output=True)

            return True

        except subprocess.CalledProcessError as e:
            print(f"分发到 {node} 失败: {e}")
            return False

    def _save_dispatch_record(self, node_partitions: Dict[str, List[int]]):
        """保存分发记录"""
        record = {
            'partition_path': self.partition_path,
            'remote_path': self.remote_path,
            'node_partitions': node_partitions
        }

        record_path = os.path.join(self.partition_path, 'dispatch_record.json')
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)

        print(f"分发记录已保存: {record_path}")

    def verify_dispatch(self) -> Dict[str, bool]:
        """
        验证分发是否成功

        Returns:
            节点验证结果 {node: success}
        """
        results = {}

        for node in self.nodes:
            try:
                # 检查远程目录是否存在
                check_cmd = (
                    f"ssh -p {self.ssh_port} {self.ssh_user}@{node} "
                    f"'test -d {self.remote_path} && echo exists'"
                )
                result = subprocess.run(
                    check_cmd, shell=True, capture_output=True, text=True
                )
                results[node] = 'exists' in result.stdout
            except Exception:
                results[node] = False

        return results


def main():
    """分发主入口"""
    import argparse

    parser = argparse.ArgumentParser(description='分发图分区数据到集群')
    parser.add_argument('--partition-path', required=True, help='本地分区数据路径')
    parser.add_argument('--ip-config', required=True, help='IP配置文件路径')
    parser.add_argument('--remote-path', default='/data/graph_partitions', help='远程数据路径')
    parser.add_argument('--num-partitions', type=int, default=32, help='分区数量')
    parser.add_argument('--ssh-user', default='root', help='SSH用户名')

    args = parser.parse_args()

    dispatcher = PartitionDispatcher(
        partition_path=args.partition_path,
        ip_config_path=args.ip_config,
        remote_path=args.remote_path,
        ssh_user=args.ssh_user
    )

    node_partitions = dispatcher.dispatch(num_partitions=args.num_partitions)

    print("\n分区分配结果:")
    for node, partitions in node_partitions.items():
        print(f"  {node}: 分区 {partitions}")


if __name__ == '__main__':
    main()
