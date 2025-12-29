# 分布式图学习框架

基于DGL (Deep Graph Library) 的分布式异构图神经网络训练框架。

## 环境要求

- 32台200GB内存物理机（无GPU）
- Hadoop集群 + Hive
- Python 3.8+
- DGL 2.0+
- PyTorch 2.0+

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux
# 或 .venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装DGL CPU版本
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### 2. 数据准备

#### 2.1 Hive表结构

```sql
-- 节点表
CREATE TABLE graph_db.nodes_user (
    node_id BIGINT,
    feature_vec ARRAY<FLOAT>,
    label INT
) PARTITIONED BY (dt STRING) STORED AS PARQUET;

-- 边表
CREATE TABLE graph_db.edges_click (
    src_node_id BIGINT,
    dst_node_id BIGINT,
    src_type STRING,
    dst_type STRING,
    edge_weight FLOAT
) PARTITIONED BY (dt STRING) STORED AS PARQUET;
```

#### 2.2 运行ETL

```bash
./scripts/run_etl.sh 2024-01-01
```

### 3. 图分区

```bash
./scripts/run_partition.sh
```

### 4. 分布式训练

```bash
./scripts/run_training.sh
```

### 5. 部署推理服务

```bash
./scripts/deploy_inference.sh
```

## 项目结构

```
graph_learning_framework/
├── config/                     # 配置文件
│   ├── cluster_config.yaml     # 集群配置
│   ├── ip_config.txt           # 节点IP列表
│   └── training_config.yaml    # 训练配置
├── spark_etl/                  # PySpark数据处理
│   ├── graph_data_pipeline.py  # Hive → CGDF管道
│   ├── hetero_graph_builder.py # 异构图构建
│   └── cgdf_writer.py          # CGDF格式写入
├── partitioning/               # 图分区
│   ├── dgl_partitioner.py      # DGL分区封装
│   └── partition_dispatch.py   # 分区分发
├── models/                     # 模型定义
│   ├── hetero_sage.py          # 异构GraphSAGE
│   ├── hetero_gat.py           # 异构GAT
│   └── rgcn.py                 # R-GCN
├── distributed/                # 分布式训练
│   ├── train_dist.py           # DistDGL训练脚本
│   ├── launch_cluster.sh       # 集群启动脚本
│   └── utils.py                # 工具函数
├── inference/                  # 在线推理
│   ├── model_handler.py        # TorchServe Handler
│   ├── graph_query_service.py  # 图查询服务
│   └── feature_sync.py         # 特征同步
└── scripts/                    # 运行脚本
```

## 核心组件

### DistDGL架构

- **Server nodes**: 存储图分区和特征
- **Sampler nodes**: 执行分布式邻居采样
- **Trainer nodes**: 执行模型训练

### 支持的模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| HeteroGraphSAGE | 异构图GraphSAGE | 通用节点分类 |
| HeteroGAT | 异构图注意力网络 | 需要注意力机制 |
| RGCN | 关系图卷积网络 | 多种边类型 |

### 支持的任务

- 节点分类
- 链接预测
- 图分类

## 配置说明

### cluster_config.yaml

```yaml
cluster:
  total_nodes: 32
  memory_per_node_gb: 200

roles:
  server_nodes: 8
  sampler_nodes: 8
  trainer_nodes: 16
```

### training_config.yaml

```yaml
model_type: "sage"
hidden_channels: 256
num_layers: 3
batch_size: 1024
learning_rate: 0.001
fanouts: [15, 10, 5]
```

## 性能估算

- 10亿节点、500维特征 ≈ 2TB总数据量
- 32台机器分摊 ≈ 62.5GB/机器
- 加上图结构 ≈ 100-120GB/机器
- 剩余80GB可用于训练

## 参考资料

- [DGL官方文档](https://docs.dgl.ai/)
- [DistDGL分布式训练](https://docs.dgl.ai/en/latest/guide/distributed.html)
- [异构图教程](https://docs.dgl.ai/en/latest/guide/training-heterogeneous.html)
