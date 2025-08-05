#!/bin/bash

# 用法:
# ./launch_hetero_multi_node.sh <主节点IP> <主节点端口> <本机node_rank> <本机GPU_IDs> <所有进程总数world_size> <本机rank起始值>
# 例: ./launch_hetero_multi_node.sh 10.1.0.12 29500 0 0,1,3 12 0
#     ./launch_hetero_multi_node.sh 10.1.0.12 29500 1 2,3,4,5,6,7,8,9 12 4

set -e

MASTER_ADDR=${1:-"localhost"}
MASTER_PORT=${2:-29500}
NODE_RANK=${3:-0}
GPU_IDS=${4:-"0,1"}
WORLD_SIZE=${5:-1}
RANK_START=${6:-0}

IFS=',' read -ra GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARR[@]}

echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "本机 node_rank: $NODE_RANK"
echo "本机 GPU IDs: $GPU_IDS"
echo "本机 rank 起始值: $RANK_START"
echo "总进程数 world_size: $WORLD_SIZE"
echo "本机 GPU 数量: $NUM_GPUS"

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
        --master-port $MASTER_PORT \
        -- \
        pi0_bridge_traj \
        --exp-name "hetero_multi_node" \
        --overwrite \
        --data.repo-id "/home/ubuntu/vla/pi0_bridge/datasets/converted_dataset/dataset0729" &
    pids[$RANK]=$!
done

echo "所有进程已启动，等待完成..."
for pid in ${pids[@]}; do
    wait $pid
done
echo "所有训练进程已完成"