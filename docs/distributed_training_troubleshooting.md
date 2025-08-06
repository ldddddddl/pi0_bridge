# 分布式训练故障排除指南

## 问题描述

在多机多卡训练中，主节点经常少一个进程，导致训练无法正常启动。从节点日志显示进程2和进程3正常运行，但主节点缺少进程0或进程1。

## 问题原因分析

1. **进程启动时序问题**：主节点和从节点的进程启动时间不同步
2. **JAX分布式初始化超时**：等待所有进程就绪的时间不够
3. **网络连接问题**：节点间通信不稳定
4. **环境变量设置不一致**：不同节点的环境变量配置有差异

## 解决方案

### 1. 使用修复版本的启动脚本

使用 `scripts/launch_multi_node_diff_devices_fixed.sh` 替代原来的启动脚本：

```bash
# 主节点 (node_rank=0)
./scripts/launch_multi_node_diff_devices_fixed.sh 10.10.1.16 29500 0 0,1 4 0

# 从节点 (node_rank=1)  
./scripts/launch_multi_node_diff_devices_fixed.sh 10.10.1.16 29500 1 0,1 4 2
```

### 2. 修复版本的主要改进

- **进程同步机制**：使用文件系统同步所有节点准备状态
- **增加重试次数**：JAX分布式初始化重试次数从3次增加到5次
- **指数退避重试**：重试间隔从5秒增加到10秒，并采用指数退避策略
- **更好的错误诊断**：增加详细的环境变量和错误信息输出

### 3. 使用诊断工具

在启动训练前，可以使用诊断工具检查环境：

```bash
# 设置环境变量
export MASTER_ADDR=10.10.1.16
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=4
export LOCAL_RANK=0
export NODE_RANK=0

# 运行诊断
python scripts/debug_distributed.py
```

### 4. 手动检查步骤

#### 4.1 检查网络连接

```bash
# 检查主节点端口是否可访问
telnet 10.10.1.16 29500

# 或者使用nc
nc -zv 10.10.1.16 29500
```

#### 4.2 检查环境变量

```bash
# 检查关键环境变量
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
```

#### 4.3 检查GPU状态

```bash
# 检查GPU使用情况
nvidia-smi

# 检查CUDA设备
python -c "import torch; print(torch.cuda.device_count())"
```

### 5. 常见问题及解决方法

#### 5.1 端口被占用

```bash
# 检查端口使用情况
netstat -tuln | grep 29500

# 如果端口被占用，可以更换端口
export MASTER_PORT=29501
```

#### 5.2 防火墙问题

```bash
# 检查防火墙设置
sudo ufw status

# 如果需要，开放端口
sudo ufw allow 29500
```

#### 5.3 进程启动顺序问题

确保所有节点同时启动，或者使用修复版本脚本中的同步机制。

#### 5.4 JAX初始化超时

- 增加重试次数和间隔
- 检查网络延迟
- 确保所有节点的时间同步

### 6. 调试技巧

#### 6.1 启用详细日志

```bash
# 设置JAX调试环境变量
export JAX_DEBUG_NANS=True
export JAX_DEBUG_INFS=True
export JAX_LOG_COMPILES=True
```

#### 6.2 单步调试

```bash
# 在训练脚本中添加断点
python -m pdb scripts/train.py
```

#### 6.3 检查进程状态

```bash
# 查看相关进程
ps aux | grep python

# 查看进程树
pstree -p
```

### 7. 最佳实践

1. **统一环境**：确保所有节点使用相同的Python环境和依赖版本
2. **时间同步**：使用NTP同步所有节点的时间
3. **网络优化**：确保节点间网络延迟低且稳定
4. **资源监控**：监控GPU内存和网络带宽使用情况
5. **日志管理**：为每个进程设置独立的日志文件

### 8. 配置文件示例

#### 8.1 主节点配置

```bash
#!/bin/bash
export MASTER_ADDR=10.10.1.16
export MASTER_PORT=29500
export WORLD_SIZE=4
export NODE_RANK=0

# 启动主节点进程
./scripts/launch_multi_node_diff_devices_fixed.sh 10.10.1.16 29500 0 0,1 4 0
```

#### 8.2 从节点配置

```bash
#!/bin/bash
export MASTER_ADDR=10.10.1.16
export MASTER_PORT=29500
export WORLD_SIZE=4
export NODE_RANK=1

# 启动从节点进程
./scripts/launch_multi_node_diff_devices_fixed.sh 10.10.1.16 29500 1 0,1 4 2
```

### 9. 监控和日志

#### 9.1 实时监控

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控网络连接
watch -n 1 netstat -tuln | grep 29500
```

#### 9.2 日志分析

```bash
# 查看训练日志
tail -f logs/training.log

# 查看错误日志
grep -i error logs/*.log
```

通过以上方法，应该能够解决主节点少一个进程的问题。如果问题仍然存在，请使用诊断工具收集更多信息，并根据具体错误信息进行针对性解决。 