# 多机多卡训练使用指南

本指南介绍如何使用修改后的训练脚本进行多机多卡分布式训练。

## 主要修改

### 1. 分布式环境初始化
- 添加了 `init_distributed_environment()` 函数来自动检测和设置分布式训练环境
- 支持SLURM集群和手动启动两种方式
- 自动设置环境变量：`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`

### 2. 日志和监控优化
- 只在主进程（rank=0）上初始化wandb和打印日志
- 进度条只在主进程显示
- 检查点保存只在主进程进行

### 3. 数据加载优化
- 数据加载器自动根据进程数调整batch size
- 支持分布式数据采样

## 使用方法

### 方法1: 使用启动脚本（推荐）

#### 单机多卡
```bash
# 在单机上启动4个GPU进程
python scripts/run_distributed.py \
    --mode manual \
    --world-size 4 \
    --gpus-per-node 4 \
    -- \
    pi0_bridge_traj \
    --exp-name "single_node_training" \
    --overwrite
```

#### 多机多卡
```bash
# 方法1: 使用bash脚本
chmod +x scripts/launch_multi_node.sh
./scripts/launch_multi_node.sh <主节点IP> <节点数量> <每节点GPU数量>

# 示例: 2个节点，每个节点4个GPU
./scripts/launch_multi_node.sh 192.168.1.100 2 4
```

#### SLURM集群
```bash
# 1. 创建SLURM脚本
python scripts/run_distributed.py --mode create_slurm \
    --job-name "distributed_training" \
    --num-nodes 2 \
    --gpus-per-node 4 \
    --time-limit "24:00:00" \
    --partition gpu \
    --output-dir ./slurm_scripts

# 2. 提交作业
sbatch slurm_scripts/run_slurm.sh
```

### 方法2: 手动启动

#### 节点1（主节点）
```bash
# 设置环境变量
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8  # 总进程数

# 启动4个进程
for i in {0..3}; do
    python scripts/run_distributed.py \
        --mode manual \
        --rank $i \
        --world-size 8 \
        --local-rank $i \
        --node-rank 0 \
        --master-addr 192.168.1.100 \
        --master-port 29500 \
        -- \
        pi0_bridge_traj \
        --exp-name "multi_node_training" \
        --overwrite &
done
```

#### 节点2
```bash
# 设置环境变量
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8

# 启动4个进程
for i in {0..3}; do
    python scripts/run_distributed.py \
        --mode manual \
        --rank $((i+4)) \
        --world-size 8 \
        --local-rank $i \
        --node-rank 1 \
        --master-addr 192.168.1.100 \
        --master-port 29500 \
        -- \
        pi0_bridge_traj \
        --exp-name "multi_node_training" \
        --overwrite &
done
```

## 环境变量说明

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `RANK` | 当前进程的全局rank | 0, 1, 2, 3... |
| `WORLD_SIZE` | 总进程数 | 8 |
| `LOCAL_RANK` | 当前节点内的rank | 0, 1, 2, 3 |
| `NODE_RANK` | 节点rank | 0, 1 |
| `MASTER_ADDR` | 主节点IP地址 | 192.168.1.100 |
| `MASTER_PORT` | 主节点端口 | 29500 |

## 配置建议

### 1. 网络配置
- 确保所有节点之间网络连通
- 建议使用高速网络（如InfiniBand）
- 检查防火墙设置，确保端口29500开放

### 2. 环境配置
```bash
# 设置JAX环境变量
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 设置CUDA环境
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3. 训练参数调整
- `batch_size`: 总batch size，会自动分配到各个进程
- `num_workers`: 建议设置为每节点CPU核心数的1/4
- `fsdp_devices`: 设置为所有可用的GPU设备

## 故障排除

### 1. 连接问题
```bash
# 测试节点间连通性
ping <其他节点IP>
telnet <其他节点IP> 29500
```

### 2. 端口冲突
```bash
# 检查端口占用
netstat -tulpn | grep 29500
# 如果端口被占用，修改MASTER_PORT环境变量
```

### 3. 内存问题
```bash
# 调整内存分配
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### 4. 进程同步问题
- 确保所有进程同时启动
- 检查网络延迟
- 增加启动超时时间

## 性能优化建议

1. **数据加载优化**
   - 使用SSD存储数据集
   - 增加数据加载的num_workers
   - 使用数据预取

2. **网络优化**
   - 使用高速网络连接
   - 减少网络通信频率
   - 使用梯度累积减少通信开销

3. **内存优化**
   - 合理设置batch_size
   - 使用梯度检查点
   - 启用混合精度训练

## 监控和调试

### 1. 日志查看
```bash
# 查看主进程日志
tail -f logs/training.log

# 查看所有进程日志
find logs/ -name "*.log" -exec tail -f {} \;
```

### 2. 性能监控
```bash
# 监控GPU使用率
nvidia-smi -l 1

# 监控网络使用
iftop -i <网络接口>
```

### 3. 调试技巧
- 先在小规模上测试
- 使用单进程模式调试
- 检查环境变量设置
- 查看错误日志定位问题 