#!/bin/bash

# 多机多卡训练快速启动示例
# 包含各种常见场景的启动命令

echo "多机多卡训练快速启动示例"
echo "=========================="

# 设置默认参数
DEFAULT_MASTER_ADDR="localhost"
DEFAULT_NUM_NODES=2
DEFAULT_GPUS_PER_NODE=4
DEFAULT_EXP_NAME="distributed_training"

# 颜色输出函数
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

# 示例1: 单机多卡
example_single_node() {
    print_green "\n示例1: 单机多卡训练 (4个GPU)"
    print_yellow "命令:"
    echo "python scripts/run_distributed.py \\"
    echo "    --mode manual \\"
    echo "    --world-size 4 \\"
    echo "    --local-rank 0 \\"
    echo "    --node-rank 0 \\"
    echo "    --master-addr localhost \\"
    echo "    --master-port 29500 \\"
    echo "    -- \\"
    echo "    pi0_bridge_traj \\"
    echo "    --exp-name \"single_node_training\" \\"
    echo "    --overwrite"
    
    echo ""
    print_yellow "或者使用bash脚本:"
    echo "./scripts/launch_multi_node.sh localhost 1 4"
}

# 示例2: 多机多卡
example_multi_node() {
    print_green "\n示例2: 多机多卡训练 (2个节点，每个节点4个GPU)"
    print_yellow "在主节点上运行:"
    echo "./scripts/launch_multi_node.sh 192.168.1.100 2 4"
    
    echo ""
    print_yellow "或者手动启动:"
    echo "# 节点1 (主节点):"
    echo "export MASTER_ADDR=192.168.1.100"
    echo "export MASTER_PORT=29500"
    echo "export WORLD_SIZE=8"
    echo ""
    echo "for i in {0..3}; do"
    echo "    python scripts/run_distributed.py \\"
    echo "        --mode manual \\"
    echo "        --rank \$i \\"
    echo "        --world-size 8 \\"
    echo "        --local-rank \$i \\"
    echo "        --node-rank 0 \\"
    echo "        --master-addr 192.168.1.100 \\"
    echo "        --master-port 29500 \\"
    echo "        -- \\"
    echo "        pi0_bridge_traj \\"
    echo "        --exp-name \"multi_node_training\" \\"
    echo "        --overwrite &"
    echo "done"
}

# 示例3: SLURM集群
example_slurm() {
    print_green "\n示例3: SLURM集群训练"
    print_yellow "1. 创建SLURM脚本:"
    echo "python scripts/run_distributed.py --mode create_slurm \\"
    echo "    --job-name \"distributed_training\" \\"
    echo "    --num-nodes 2 \\"
    echo "    --gpus-per-node 4 \\"
    echo "    --time-limit \"24:00:00\" \\"
    echo "    --partition gpu \\"
    echo "    --output-dir ./slurm_scripts"
    
    echo ""
    print_yellow "2. 提交作业:"
    echo "sbatch slurm_scripts/run_slurm.sh"
}

# 示例4: 环境测试
example_test() {
    print_green "\n示例4: 分布式环境测试"
    print_yellow "测试单机环境:"
    echo "python scripts/test_distributed.py"
    
    echo ""
    print_yellow "测试多机环境:"
    echo "# 在主节点上:"
    echo "export MASTER_ADDR=192.168.1.100"
    echo "export MASTER_PORT=29500"
    echo "export WORLD_SIZE=8"
    echo "python scripts/test_distributed.py"
    
    echo ""
    echo "# 在其他节点上:"
    echo "export MASTER_ADDR=192.168.1.100"
    echo "export MASTER_PORT=29500"
    echo "export WORLD_SIZE=8"
    echo "python scripts/test_distributed.py"
}

# 示例5: 环境变量设置
example_env_setup() {
    print_green "\n示例5: 环境变量设置"
    print_yellow "在 ~/.bashrc 中添加:"
    echo "export JAX_PLATFORM_NAME=gpu"
    echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
    echo "export CUDA_VISIBLE_DEVICES=0,1,2,3"
}

# 示例6: 故障排除
example_troubleshooting() {
    print_green "\n示例6: 常见问题排查"
    print_yellow "1. 检查网络连通性:"
    echo "ping <其他节点IP>"
    echo "telnet <其他节点IP> 29500"
    
    echo ""
    print_yellow "2. 检查端口占用:"
    echo "netstat -tulpn | grep 29500"
    
    echo ""
    print_yellow "3. 检查GPU状态:"
    echo "nvidia-smi"
    
    echo ""
    print_yellow "4. 检查环境变量:"
    echo "env | grep -E \"(RANK|WORLD_SIZE|MASTER|CUDA)\""
}

# 显示所有示例
show_all_examples() {
    echo "选择要查看的示例:"
    echo "1) 单机多卡训练"
    echo "2) 多机多卡训练"
    echo "3) SLURM集群训练"
    echo "4) 分布式环境测试"
    echo "5) 环境变量设置"
    echo "6) 故障排除"
    echo "7) 显示所有示例"
    echo "0) 退出"
    
    read -p "请输入选择 (0-7): " choice
    
    case $choice in
        1) example_single_node ;;
        2) example_multi_node ;;
        3) example_slurm ;;
        4) example_test ;;
        5) example_env_setup ;;
        6) example_troubleshooting ;;
        7) 
            example_single_node
            example_multi_node
            example_slurm
            example_test
            example_env_setup
            example_troubleshooting
            ;;
        0) exit 0 ;;
        *) echo "无效选择" ;;
    esac
}

# 主菜单
main_menu() {
    while true; do
        echo ""
        print_green "多机多卡训练快速启动示例"
        echo "================================"
        echo "1) 查看启动示例"
        echo "2) 运行环境测试"
        echo "3) 创建SLURM脚本"
        echo "4) 查看详细文档"
        echo "0) 退出"
        
        read -p "请选择操作 (0-4): " choice
        
        case $choice in
            1) show_all_examples ;;
            2) 
                print_yellow "运行分布式环境测试..."
                python scripts/test_distributed.py
                ;;
            3)
                print_yellow "创建SLURM脚本..."
                python scripts/run_distributed.py --mode create_slurm \
                    --job-name "distributed_training" \
                    --num-nodes 2 \
                    --gpus-per-node 4 \
                    --time-limit "24:00:00" \
                    --partition gpu \
                    --output-dir ./slurm_scripts
                ;;
            4)
                print_yellow "查看详细文档..."
                if [ -f "scripts/README_distributed_training.md" ]; then
                    cat scripts/README_distributed_training.md
                else
                    echo "文档文件不存在"
                fi
                ;;
            0) exit 0 ;;
            *) echo "无效选择" ;;
        esac
    done
}

# 如果直接运行脚本，显示主菜单
if [ "${BASH_SOURCE:-$0}" = "$0" ]; then
    main_menu
fi 