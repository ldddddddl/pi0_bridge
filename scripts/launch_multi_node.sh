#!/bin/bash

# 多机多卡训练启动脚本
# 使用方法: ./launch_multi_node.sh <主节点IP> <节点数量> <每节点GPU数量>

set -e

MASTER_ADDR=${1:-"localhost"}
NUM_NODES=${2:-2}
GPUS_PER_NODE=${3:-4}

echo "启动多机多卡训练:"
echo "  主节点地址: $MASTER_ADDR"
echo "  节点数量: $NUM_NODES"
echo "  每节点GPU数量: $GPUS_PER_NODE"

# 检查参数
if [ -z "$MASTER_ADDR" ]; then
    echo "错误: 请提供主节点IP地址"
    echo "使用方法: $0 <主节点IP> [节点数量] [每节点GPU数量]"
    exit 1
fi

# 设置环境变量
export MASTER_PORT=29500
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "总进程数: $WORLD_SIZE"

# 启动所有节点的进程
for node_rank in $(seq 0 $((NUM_NODES-1))); do
    for local_rank in $(seq 0 $((GPUS_PER_NODE-1))); do
        rank=$((node_rank * GPUS_PER_NODE + local_rank))
        
        echo "启动进程: Rank=$rank, Node=$node_rank, Local=$local_rank"
        
        # 在后台启动进程
        python scripts/run_distributed.py \
            --mode manual \
            --rank $rank \
            --world-size $WORLD_SIZE \
            --local-rank $local_rank \
            --node-rank $node_rank \
            --master-addr $MASTER_ADDR \
            --master-port 29500 \
            -- \
            pi0_bridge_traj \
            --exp-name "multi_node_training" \
            --overwrite \
            --data.repo-id "/home/ubuntu/vla/pi0_bridge/datasets/converted_dataset/dataset0729" &
        
        # 记录进程ID
        pids[$rank]=$!
    done
done

echo "所有进程已启动，等待完成..."

# 等待所有进程完成
for pid in ${pids[@]}; do
    wait $pid
done

echo "所有训练进程已完成" 