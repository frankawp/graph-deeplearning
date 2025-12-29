#!/bin/bash
# 在线推理服务部署脚本

set -e

# 配置
PROJECT_HOME=${PROJECT_HOME:-"/data/graph_learning"}
MODEL_PATH="$PROJECT_HOME/checkpoints/best_model.pt"
MODEL_STORE="$PROJECT_HOME/inference/model-store"
CONFIG_PATH="$PROJECT_HOME/config/training_config.yaml"

# TorchServe配置
MODEL_NAME="hetero_gnn"
INFERENCE_PORT=8080
MANAGEMENT_PORT=8081

echo "=== 部署推理服务 ==="

# 创建model-store目录
mkdir -p $MODEL_STORE

# 导出模型为TorchScript
echo "导出模型..."
python << EOF
import torch
import yaml
import sys
sys.path.insert(0, '$PROJECT_HOME')

from models.hetero_sage import HeteroSAGEForNodeClassification

# 加载配置
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

# 创建模型（使用示例参数）
in_feats = {'user': 256, 'item': 256}
etypes = [('user', 'click', 'item'), ('user', 'purchase', 'item')]

model = HeteroSAGEForNodeClassification(
    in_feats=in_feats,
    hidden_feats=config['hidden_channels'],
    num_classes=config['num_classes'],
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    target_ntype=config['target_node_type'],
    etypes=etypes
)

# 加载权重
checkpoint = torch.load('$MODEL_PATH', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 导出为TorchScript
# 注意：异构图模型的TorchScript导出较复杂，这里保存PyTorch格式
torch.save(model, '$MODEL_STORE/model.pt')

# 保存配置
import json
with open('$MODEL_STORE/config.json', 'w') as f:
    json.dump({
        'hidden_channels': config['hidden_channels'],
        'num_classes': config['num_classes'],
        'target_node_type': config['target_node_type'],
        'redis_host': config.get('redis', {}).get('host', 'localhost'),
        'redis_port': config.get('redis', {}).get('port', 6379),
        'feature_dim': 256,
        'num_hops': 2,
        'max_neighbors': [15, 10]
    }, f)

print("模型导出完成")
EOF

# 打包模型
echo "打包模型..."
torch-model-archiver \
    --model-name $MODEL_NAME \
    --version 1.0 \
    --serialized-file "$MODEL_STORE/model.pt" \
    --handler "$PROJECT_HOME/inference/model_handler.py" \
    --extra-files "$MODEL_STORE/config.json" \
    --export-path "$MODEL_STORE"

# 启动TorchServe
echo "启动TorchServe..."
torchserve --stop || true  # 停止已有实例

torchserve --start \
    --model-store "$MODEL_STORE" \
    --models "$MODEL_NAME=$MODEL_NAME.mar" \
    --ts-config "$PROJECT_HOME/inference/ts_config.properties" \
    --foreground &

echo ""
echo "=== 推理服务已启动 ==="
echo "推理端口: $INFERENCE_PORT"
echo "管理端口: $MANAGEMENT_PORT"
echo ""
echo "测试请求:"
echo "curl -X POST http://localhost:$INFERENCE_PORT/predictions/$MODEL_NAME -H 'Content-Type: application/json' -d '{\"node_id\": 12345, \"node_type\": \"user\", \"task_type\": \"node_classify\"}'"
