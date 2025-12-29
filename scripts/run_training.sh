#!/bin/bash
# 分布式训练启动脚本

set -e

# 配置
PROJECT_HOME=${PROJECT_HOME:-"/data/graph_learning"}
PARTITION_PATH="/data/graph_partitions"
IP_CONFIG="$PROJECT_HOME/config/ip_config.txt"
TRAIN_CONFIG="$PROJECT_HOME/config/training_config.yaml"
GRAPH_NAME="hetero_graph"

# Python环境
PYTHON=${PYTHON:-"$PROJECT_HOME/.venv/bin/python"}

# DistDGL进程配置
NUM_TRAINERS=1
NUM_SAMPLERS=1
NUM_SERVERS=1

echo "=== DistDGL分布式训练 ==="
echo "IP配置: $IP_CONFIG"
echo "分区路径: $PARTITION_PATH"
echo "训练配置: $TRAIN_CONFIG"

# 检查配置文件
if [ ! -f "$IP_CONFIG" ]; then
    echo "错误: IP配置文件不存在"
    exit 1
fi

if [ ! -f "$PARTITION_PATH/$GRAPH_NAME.json" ]; then
    echo "错误: 分区配置文件不存在"
    exit 1
fi

# 统计节点数
NUM_NODES=$(grep -v '^#' "$IP_CONFIG" | grep -v '^$' | wc -l)
echo "集群节点数: $NUM_NODES"

# 使用DGL分布式启动
$PYTHON -m dgl.distributed.launch \
    --workspace "$PROJECT_HOME" \
    --num_trainers $NUM_TRAINERS \
    --num_samplers $NUM_SAMPLERS \
    --num_servers $NUM_SERVERS \
    --part_config "$PARTITION_PATH/$GRAPH_NAME.json" \
    --ip_config "$IP_CONFIG" \
    "$PROJECT_HOME/distributed/train_dist.py" \
    --config "$TRAIN_CONFIG" \
    --graph-name "$GRAPH_NAME" \
    --part-config "$PARTITION_PATH/$GRAPH_NAME.json" \
    --ip-config "$IP_CONFIG"

echo "=== 训练完成 ==="
