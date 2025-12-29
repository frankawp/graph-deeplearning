#!/bin/bash
# 图分区脚本
# 使用DGL对图进行分区并分发到各节点

set -e

# 配置
PROJECT_HOME=${PROJECT_HOME:-"/data/graph_learning"}
CGDF_PATH="$PROJECT_HOME/data/cgdf"
PARTITION_PATH="$PROJECT_HOME/data/partitions"
IP_CONFIG="$PROJECT_HOME/config/ip_config.txt"
NUM_PARTITIONS=32
GRAPH_NAME="hetero_graph"

# Python环境
PYTHON=${PYTHON:-"$PROJECT_HOME/.venv/bin/python"}

echo "=== 图分区 ==="
echo "CGDF路径: $CGDF_PATH"
echo "分区输出: $PARTITION_PATH"
echo "分区数量: $NUM_PARTITIONS"

# 创建输出目录
mkdir -p $PARTITION_PATH

# 执行分区
$PYTHON << EOF
import sys
sys.path.insert(0, '$PROJECT_HOME')

from partitioning.dgl_partitioner import DGLPartitioner, HeteroGraphLoader

# 加载CGDF数据
print("加载CGDF数据...")
loader = HeteroGraphLoader('$CGDF_PATH')
graph = loader.load_graph()
print(f"图信息: {graph.num_nodes()} 节点, {graph.num_edges()} 边")

# 分区
print("执行分区...")
partitioner = DGLPartitioner(
    num_partitions=$NUM_PARTITIONS,
    part_method='random',  # 超大规模图使用random
    output_path='$PARTITION_PATH'
)

part_config = partitioner.partition(
    graph,
    graph_name='$GRAPH_NAME'
)

print(f"分区完成: {part_config}")
EOF

echo ""
echo "=== 分发分区数据 ==="

# 分发到各节点
$PYTHON "$PROJECT_HOME/partitioning/partition_dispatch.py" \
    --partition-path "$PARTITION_PATH" \
    --ip-config "$IP_CONFIG" \
    --remote-path "/data/graph_partitions" \
    --num-partitions $NUM_PARTITIONS

echo "=== 分区完成 ==="
