# 多机多卡训练重构总结

## 概述

本次重构将原有的单机训练脚本 `scripts/train.py` 改造为支持多机多卡分布式训练，同时提供了完整的启动脚本、测试工具和文档。

## 主要修改内容

### 1. 核心训练脚本修改 (`scripts/train.py`)

#### 新增功能：
- **分布式环境初始化**: `init_distributed_environment()` 函数
- **智能环境检测**: 支持SLURM集群和手动启动两种方式
- **进程感知日志**: 只在主进程显示进度条和wandb日志
- **分布式数据加载**: 自动根据进程数调整batch size
- **检查点管理**: 只在主进程保存检查点

#### 关键修改点：
```python
# 分布式环境初始化
dist_info = init_distributed_environment()
init_logging(dist_info)

# 进程感知的日志和监控
if dist_info["rank"] == 0:
    wandb.log(reduced_info, step=step)
    pbar.write(f"Step {step}: {info_str}")

# 数据加载器优化
local_batch_size=batch_size // jax.process_count()
```

### 2. 启动脚本 (`scripts/run_distributed.py`)

#### 功能特性：
- **多模式支持**: SLURM、手动启动、脚本创建
- **环境变量管理**: 自动设置分布式训练所需的环境变量
- **参数化配置**: 支持自定义rank、world_size等参数
- **SLURM脚本生成**: 自动生成SLURM提交脚本

#### 使用示例：
```bash
# 单机多卡
python scripts/run_distributed.py --mode manual --world-size 4

# 多机多卡
python scripts/run_distributed.py --mode manual --rank 0 --world-size 8

# 创建SLURM脚本
python scripts/run_distributed.py --mode create_slurm --num-nodes 2
```

### 3. 快速启动脚本 (`scripts/launch_multi_node.sh`)

#### 功能特性：
- **一键启动**: 简化多机多卡训练启动流程
- **自动进程管理**: 自动启动所有节点的进程
- **错误处理**: 包含基本的错误检查和进程同步

#### 使用示例：
```bash
# 2个节点，每个节点4个GPU
./scripts/launch_multi_node.sh 192.168.1.100 2 4
```

### 4. 环境测试脚本 (`scripts/test_distributed.py`)

#### 测试内容：
- **环境变量检查**: 验证分布式环境变量设置
- **设备信息测试**: 检查JAX设备可用性
- **网络连通性**: 测试节点间网络连接
- **基本通信**: 验证进程间通信功能

#### 使用示例：
```bash
python scripts/test_distributed.py
```

### 5. 快速启动示例 (`scripts/quick_start_examples.sh`)

#### 功能特性：
- **交互式菜单**: 提供友好的用户界面
- **多种场景**: 覆盖单机、多机、SLURM等场景
- **故障排除**: 包含常见问题的解决方案

## 支持的环境

### 1. SLURM集群
- 自动检测SLURM环境变量
- 支持多节点作业提交
- 自动设置主节点地址和端口

### 2. 手动启动
- 支持自定义rank和world_size
- 灵活的环境变量配置
- 适用于各种集群环境

### 3. 单机多卡
- 简化的启动流程
- 自动GPU设备分配
- 适合开发和测试

## 环境变量说明

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `RANK` | 当前进程的全局rank | 0, 1, 2, 3... |
| `WORLD_SIZE` | 总进程数 | 8 |
| `LOCAL_RANK` | 当前节点内的rank | 0, 1, 2, 3 |
| `NODE_RANK` | 节点rank | 0, 1 |
| `MASTER_ADDR` | 主节点IP地址 | 192.168.1.100 |
| `MASTER_PORT` | 主节点端口 | 29500 |

## 使用流程

### 1. 环境准备
```bash
# 设置JAX环境变量
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

### 2. 环境测试
```bash
# 运行分布式环境测试
python scripts/test_distributed.py
```

### 3. 启动训练
```bash
# 方法1: 使用启动脚本
./scripts/launch_multi_node.sh <主节点IP> <节点数> <每节点GPU数>

# 方法2: 使用Python脚本
python scripts/run_distributed.py --mode manual --world-size 8

# 方法3: SLURM集群
sbatch slurm_scripts/run_slurm.sh
```

## 性能优化

### 1. 数据加载优化
- 自动根据进程数调整batch size
- 支持分布式数据采样
- 优化数据预取和缓存

### 2. 通信优化
- 使用JAX内置的分布式通信
- 减少不必要的网络通信
- 支持梯度累积减少通信开销

### 3. 内存优化
- 合理设置内存分配比例
- 支持梯度检查点
- 优化模型参数分布

## 故障排除

### 常见问题及解决方案

1. **网络连接问题**
   ```bash
   # 检查网络连通性
   ping <其他节点IP>
   telnet <其他节点IP> 29500
   ```

2. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 29500
   # 修改MASTER_PORT环境变量
   ```

3. **内存不足**
   ```bash
   # 调整内存分配
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
   ```

4. **进程同步问题**
   - 确保所有进程同时启动
   - 检查网络延迟
   - 增加启动超时时间

## 监控和调试

### 1. 日志管理
- 主进程日志：包含完整的训练信息
- 从进程日志：包含错误和警告信息
- 分布式日志：包含通信相关信息

### 2. 性能监控
```bash
# GPU使用率监控
nvidia-smi -l 1

# 网络使用监控
iftop -i <网络接口>

# 进程状态监控
ps aux | grep python
```

### 3. 调试技巧
- 先在小规模上测试
- 使用单进程模式调试
- 检查环境变量设置
- 查看错误日志定位问题

## 文件结构

```
scripts/
├── train.py                          # 主训练脚本（已修改）
├── run_distributed.py                # 分布式启动脚本（新增）
├── launch_multi_node.sh             # 快速启动脚本（新增）
├── test_distributed.py              # 环境测试脚本（新增）
├── quick_start_examples.sh          # 启动示例脚本（新增）
├── README_distributed_training.md   # 详细使用文档（新增）
└── DISTRIBUTED_TRAINING_SUMMARY.md # 重构总结文档（本文件）
```

## 兼容性说明

### 向后兼容
- 原有的单机训练功能完全保留
- 环境变量自动检测，无需手动设置
- 支持原有的所有训练参数

### 扩展性
- 支持任意数量的节点和GPU
- 可扩展支持其他集群管理系统
- 模块化设计，易于维护和扩展

## 总结

本次重构成功将单机训练脚本改造为支持多机多卡分布式训练，主要特点：

1. **自动化程度高**: 自动检测环境，减少手动配置
2. **兼容性好**: 保持原有功能，支持多种启动方式
3. **工具完善**: 提供完整的测试、启动和监控工具
4. **文档详细**: 包含详细的使用说明和故障排除指南
5. **易于使用**: 提供多种简化的启动方式

通过这些修改，用户可以轻松地在单机、多机或SLURM集群环境中进行分布式训练，大大提高了训练效率和资源利用率。 