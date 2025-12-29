#!/bin/bash
# PySpark ETL执行脚本
# 从Hive提取数据并转换为CGDF格式

set -e

# 配置
SPARK_HOME=${SPARK_HOME:-"/opt/spark"}
PROJECT_HOME=${PROJECT_HOME:-"/data/graph_learning"}
PYTHON_SCRIPT="$PROJECT_HOME/spark_etl/graph_data_pipeline.py"

# Spark配置
DRIVER_MEMORY="16g"
EXECUTOR_MEMORY="32g"
EXECUTOR_CORES=8
NUM_EXECUTORS=16

# 数据日期
DT=${1:-$(date +%Y-%m-%d)}

echo "=== PySpark ETL ==="
echo "数据日期: $DT"
echo "项目目录: $PROJECT_HOME"

# 检查Spark
if [ ! -d "$SPARK_HOME" ]; then
    echo "错误: SPARK_HOME不存在: $SPARK_HOME"
    exit 1
fi

# 提交Spark任务
$SPARK_HOME/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --driver-memory $DRIVER_MEMORY \
    --executor-memory $EXECUTOR_MEMORY \
    --executor-cores $EXECUTOR_CORES \
    --num-executors $NUM_EXECUTORS \
    --conf spark.sql.shuffle.partitions=200 \
    --conf spark.default.parallelism=200 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --py-files "$PROJECT_HOME/spark_etl/__init__.py,$PROJECT_HOME/spark_etl/hetero_graph_builder.py,$PROJECT_HOME/spark_etl/cgdf_writer.py" \
    $PYTHON_SCRIPT \
    --dt $DT

echo "=== ETL完成 ==="
