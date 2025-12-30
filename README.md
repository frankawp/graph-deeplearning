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

## 分布式架构详解

### 系统架构全景图

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              分布式图学习系统架构                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                            离线数据处理阶段                                        │   │
│  │                                                                                  │   │
│  │   ┌─────────┐      ┌─────────────────┐      ┌─────────────────┐                  │   │
│  │   │  Hive   │ ───► │  Spark ETL      │ ───► │  CGDF 格式      │                  │   │
│  │   │  数据仓库 │      │ graph_data_     │      │  node_data/     │                  │   │
│  │   └─────────┘      │ pipeline.py     │      │  edge_data/     │                  │   │
│  │                    └─────────────────┘      │  metadata.json  │                  │   │
│  │                                             └────────┬────────┘                  │   │
│  │                                                      │                           │   │
│  │                                                      ▼                           │   │
│  │                                             ┌─────────────────┐                  │   │
│  │                                             │  DGL Partitioner │                  │   │
│  │                                             │  32个分区         │                  │   │
│  │                                             └────────┬────────┘                  │   │
│  │                                                      │                           │   │
│  └──────────────────────────────────────────────────────┼───────────────────────────┘   │
│                                                         │                               │
│                                                         ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         分布式训练集群 (32节点 × 200GB)                            │   │
│  │                                                                                  │   │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                         node-01 (Master)                                │    │   │
│  │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │    │   │
│  │   │   │ Server进程   │  │ Sampler进程 │  │ Trainer进程  │                     │    │   │
│  │   │   │ (图分区存储)  │  │ (邻居采样)   │  │ (模型训练)   │                     │    │   │
│  │   │   └─────────────┘  └─────────────┘  └─────────────┘                     │    │   │
│  │   └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    ...                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                         node-32                                         │    │   │
│  │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │    │   │
│  │   │   │ Server进程   │  │ Sampler进程 │  │ Trainer进程  │                     │    │   │
│  │   │   └─────────────┘  └─────────────┘  └─────────────┘                     │    │   │
│  │   └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                               │
│                                         ▼ 模型导出                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              在线推理服务                                          │   │
│  │                                                                                  │   │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                     │   │
│  │   │  TorchServe  │ ◄── │    Redis     │ ◄── │ GraphQuery   │                     │   │
│  │   │  (模型推理)   │     │  (特征缓存)   │     │  Server      │                     │   │
│  │   │  Port: 8080  │     │  Port: 6379  │     │  Port: 8080  │                     │   │
│  │   └──────────────┘     └──────────────┘     └──────────────┘                     │   │
│  │                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 集群角色分配 (32节点)

| 角色 | 节点数 | 内存分配 | 职责 |
|------|--------|----------|------|
| Server | 8 | 150GB | 存储图分区 + 特征服务 |
| Sampler | 8 | 50GB | 分布式邻居采样 |
| Trainer | 16 | 50GB | 模型参数训练 |

### 各节点启动内容

#### 训练阶段 (node-01 ~ node-32)

通过 `dgl.distributed.launch` 自动启动，每个节点上会启动 3 个进程：

| 进程 | 端口 | 功能 |
|------|------|------|
| Server | 30000 (RPC) | 存储本节点的图分区，响应远程特征/边查询 |
| Sampler | - | 从 Server 拉取邻居，执行分布式采样 |
| Trainer | - | 接收采样的 mini-batch，执行前向/反向传播 |

```bash
# 在 Master 节点执行，自动分发到所有节点
python -m dgl.distributed.launch \
    --workspace "$PROJECT_HOME" \
    --num_trainers 1 \
    --num_samplers 1 \
    --num_servers 1 \
    --part_config "/data/graph_partitions/hetero_graph.json" \
    --ip_config "config/ip_config.txt" \
    distributed/train_dist.py ...
```

#### 推理阶段

**TorchServe 推理服务：**

```bash
torchserve --start \
    --model-store "/data/graph_learning/inference/model-store" \
    --models "hetero_gnn=hetero_gnn.mar" \
    --ts-config "inference/ts_config.properties"
```

| 端口 | 服务 |
|------|------|
| 8080 | 推理请求 API |
| 8081 | 管理 API |
| 8082 | Metrics |

**Redis 特征缓存：**

```bash
redis-server --port 6379 --maxmemory 100gb

# 同步特征到 Redis
python inference/feature_sync.py \
    --redis-host redis-master \
    --parquet-path /data/cgdf/node_data/user \
    --node-type user
```

**图查询服务：**

```bash
# 每个分区节点启动一个
python inference/graph_query_service.py \
    --partition-id 0 \
    --partition-path /data/graph_partitions/part0 \
    --port 8080
```

### 完整部署流程

```
Step 1: 离线 ETL (Spark 集群)
────────────────────────────────────
spark-submit spark_etl/graph_data_pipeline.py
                    │
                    ▼
Step 2: 图分区 (单节点 / HDFS)
────────────────────────────────────
python partitioning/dgl_partitioner.py
                    │
                    ▼
Step 3: 分发分区数据到各节点
────────────────────────────────────
rsync -avz /data/graph_partitions/ node-{01..32}:/data/graph_partitions/
                    │
                    ▼
Step 4: 启动分布式训练 (Master 节点)
────────────────────────────────────
bash scripts/run_training.sh
                    │
                    ▼
Step 5: 部署在线推理
────────────────────────────────────
# 5.1 启动 Redis
redis-server

# 5.2 同步特征到 Redis
python inference/feature_sync.py ...

# 5.3 启动图查询服务 (多节点)
python inference/graph_query_service.py ...

# 5.4 启动 TorchServe
bash scripts/deploy_inference.sh
```

### 网络通信

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段通信                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Trainer ◄────────────────────────────────► Trainer            │
│      │         梯度同步 (AllReduce)              │              │
│      │         Backend: Gloo                     │              │
│      │         Port: 29500                       │              │
│      │                                           │              │
│      ▼                                           ▼              │
│   Sampler ◄────────────────────────────────► Sampler            │
│      │         采样请求/响应                      │              │
│      │         RPC Port: 30000                   │              │
│      │                                           │              │
│      ▼                                           ▼              │
│   Server ◄─────────────────────────────────► Server             │
│            图分区数据交换                                        │
│            RPC Port: 30000                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        推理阶段通信                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Client ──► TorchServe:8080 ──► Redis:6379                     │
│                    │                                            │
│                    └──► GraphQueryServer:8080                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 单节点服务清单

| 阶段 | 节点 | 启动命令 | 端口 |
|------|------|----------|------|
| 训练 | 所有节点 | `dgl.distributed.launch` 自动启动 | 29500, 30000 |
| 推理 | 推理节点 | `torchserve --start` | 8080, 8081 |
| 推理 | Redis节点 | `redis-server` | 6379 |
| 推理 | 图查询节点 | `graph_query_service.py` | 8080 |

## 数据结构

### 节点表 (`nodes_{node_type}`)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `node_id` | BIGINT | 节点原始ID（业务ID） |
| `feature_vec` | ARRAY<FLOAT> | 节点特征向量 |
| `label` | INT | 节点标签（用于分类任务） |
| `dt` | STRING | 分区字段（日期） |

### 边表 (`edges_{edge_name}`)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `src_node_id` | BIGINT | 源节点ID |
| `dst_node_id` | BIGINT | 目标节点ID |
| `edge_weight` | FLOAT | 边权重 |
| `src_type` | STRING | 源节点类型 |
| `dst_type` | STRING | 目标节点类型 |
| `dt` | STRING | 分区字段（日期） |

### CGDF 输出格式

```
output_path/
├── node_data/
│   ├── user/              # 节点类型目录
│   │   └── part-*.parquet # global_node_id, feature_vec, label
│   └── item/
│       └── part-*.parquet
├── edge_data/
│   ├── user_click_item/   # src_type_edge_name_dst_type
│   │   └── part-*.parquet # src, dst, edge_weight
│   └── ...
└── metadata.json          # 元数据（节点类型、边类型、ID范围等）
```

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
