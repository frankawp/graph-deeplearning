#!/bin/bash
# DistDGL分布式训练启动脚本
# 在32节点CPU集群上启动分布式训练

set -e

# 配置
WORKSPACE="/data/graph_learning"
IP_CONFIG="$WORKSPACE/config/ip_config.txt"
PART_CONFIG="$WORKSPACE/partitions/hetero_graph.json"
TRAIN_CONFIG="$WORKSPACE/config/training_config.yaml"
GRAPH_NAME="hetero_graph"

# 每个节点的进程配置
NUM_TRAINERS=1
NUM_SAMPLERS=1
NUM_SERVERS=1

# Python环境
PYTHON_ENV="$WORKSPACE/.venv/bin/python"

# 训练脚本
TRAIN_SCRIPT="$WORKSPACE/distributed/train_dist.py"

echo "=== DistDGL分布式训练启动 ==="
echo "工作目录: $WORKSPACE"
echo "IP配置: $IP_CONFIG"
echo "分区配置: $PART_CONFIG"

# 检查配置文件
if [ ! -f "$IP_CONFIG" ]; then
    echo "错误: IP配置文件不存在: $IP_CONFIG"
    exit 1
fi

if [ ! -f "$PART_CONFIG" ]; then
    echo "错误: 分区配置文件不存在: $PART_CONFIG"
    exit 1
fi

# 统计节点数
NUM_NODES=$(grep -v '^#' "$IP_CONFIG" | grep -v '^$' | wc -l)
echo "集群节点数: $NUM_NODES"

# 使用DGL的分布式启动工具
echo ""
echo "启动分布式训练..."

$PYTHON_ENV -m dgl.distributed.launch \
    --workspace "$WORKSPACE" \
    --num_trainers $NUM_TRAINERS \
    --num_samplers $NUM_SAMPLERS \
    --num_servers $NUM_SERVERS \
    --part_config "$PART_CONFIG" \
    --ip_config "$IP_CONFIG" \
    "$TRAIN_SCRIPT" \
    --config "$TRAIN_CONFIG" \
    --graph-name "$GRAPH_NAME" \
    --part-config "$PART_CONFIG" \
    --ip-config "$IP_CONFIG"

echo ""
echo "=== 训练完成 ==="
