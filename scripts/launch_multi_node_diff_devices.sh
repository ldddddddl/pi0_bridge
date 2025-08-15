#!/bin/bash

# 用法:
# ./launch_hetero_multi_node.sh <主节点IP> <主节点端口> <本机node_rank> <本机GPU_IDs> <所有进程总数world_size> <本机rank起始值>
# 例: ./launch_hetero_multi_node.sh
#     ./launch_hetero_multi_node.sh 10.1.0.12 29500 1 2,3,4,5,6,7,8,9 12 4

: "
手动设置
export MASTER_ADDR=10.10.1.175
export MASTER_PORT=29500
export WORLD_SIZE=2
export NODE_RANK=0
"
# 清空 29500 端口
: "
for pid in $(lsof -ti:29500); do
    echo "正在终止进程: $pid"
    kill -9 $pid
done
"

# /etc/hosts 如果通信不上, 在 /etc/hosts 中添加 ip username

# ====== 你可以在这里写默认参数 ======
DEFAULT_MASTER_ADDR="10.10.1.175"
DEFAULT_MASTER_PORT=29500
DEFAULT_NODE_RANK=0
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_WORLD_SIZE=16
DEFAULT_RANK_START=0
# ===================================

# 判断是否带参数，带了就用参数，否则用默认
MASTER_ADDR=${1:-$DEFAULT_MASTER_ADDR}
MASTER_PORT=${2:-$DEFAULT_MASTER_PORT}
NODE_RANK=${3:-$DEFAULT_NODE_RANK}
GPU_IDS=${4:-$DEFAULT_GPU_IDS}
WORLD_SIZE=${5:-$DEFAULT_WORLD_SIZE}
RANK_START=${6:-$DEFAULT_RANK_START}

IFS=',' read -ra GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARR[@]}

echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "本机 node_rank: $NODE_RANK"
echo "本机 GPU IDs: $GPU_IDS"
echo "本机 rank 起始值: $RANK_START"
echo "总进程数 world_size: $WORLD_SIZE"
echo "本机 GPU 数量: $NUM_GPUS"

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export NODE_RANK

# 检查端口是否被占用
if lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "警告: 端口 $MASTER_PORT 已被占用，可能会导致连接问题"
    echo "建议先清理端口: lsof -ti:$MASTER_PORT | xargs kill -9"
fi

# 创建日志目录
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 存储进程ID的数组
declare -a pids

for i in "${!GPU_ID_ARR[@]}"; do
    export CUDA_VISIBLE_DEVICES=${GPU_ID_ARR[$i]}
    RANK=$((RANK_START + i))
    LOCAL_RANK=$i
    echo "启动进程: global_rank=$RANK, local_rank=$LOCAL_RANK, node_rank=$NODE_RANK, 使用物理GPU=${GPU_ID_ARR[$i]}"
    python scripts/run_distributed.py \
        --mode manual \
        --rank $RANK \
        --world-size $WORLD_SIZE \
        --local-rank $LOCAL_RANK \
        --node-rank $NODE_RANK \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT &
    pids[$RANK]=$!
    echo "进程 $RANK 已启动，PID: ${pids[$RANK]}, 日志: $LOG_FILE"
    
    # 等待一小段时间再启动下一个进程，避免同时启动导致的资源竞争
    sleep 2
done

echo "所有进程已启动，等待完成..."
for pid in ${pids[@]}; do
    wait $pid
done
echo "所有训练进程已完成"