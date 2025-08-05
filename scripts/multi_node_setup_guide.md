 # 多机训练环境搭建指南

## 🖥️ 环境一致性要求

### 必须一致的环境组件

#### 1. Python环境
```bash
# 检查Python版本
python --version
# 所有节点必须输出相同版本，如: Python 3.9.18

# 检查pip版本
pip --version
```

#### 2. JAX和CUDA环境
```bash
# 检查JAX版本
python -c "import jax; print(jax.__version__)"

# 检查CUDA版本
nvidia-smi
# 查看CUDA Version字段

# 检查JAX是否能检测到GPU
python -c "import jax; print(jax.devices())"
```

#### 3. 依赖包版本
```bash
# 导出当前环境的包列表
pip freeze > requirements.txt

# 在其他节点上安装相同版本
pip install -r requirements.txt
```

## 🛠️ 环境搭建步骤

### 步骤1: 主节点环境准备
```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装JAX (根据您的CUDA版本)
# CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 12.1
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 验证环境
python -c "import jax; print('JAX版本:', jax.__version__); print('可用设备:', jax.devices())"
```

### 步骤2: 同步环境到其他节点

#### 方法1: 使用requirements.txt
```bash
# 在主节点上导出环境
pip freeze > requirements.txt

# 将requirements.txt复制到其他节点
scp requirements.txt user@node2:/path/to/project/
scp requirements.txt user@node3:/path/to/project/

# 在其他节点上安装
ssh user@node2
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 方法2: 使用conda环境
```bash
# 在主节点上导出conda环境
conda env export > environment.yml

# 在其他节点上创建相同环境
conda env create -f environment.yml
```

#### 方法3: 使用Docker
```dockerfile
# Dockerfile示例
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 安装Python和依赖
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 复制项目代码
COPY . /app
WORKDIR /app

# 安装项目依赖
RUN pip3 install -r requirements.txt
```

### 步骤3: 代码同步

#### 方法1: Git同步
```bash
# 确保所有节点使用相同的代码版本
git pull origin main
git checkout <specific-commit-hash>
```

#### 方法2: 手动同步
```bash
# 使用rsync同步代码
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    /path/to/project/ user@node2:/path/to/project/
```

## 🔧 环境验证脚本

### 创建环境检查脚本
```python
#!/usr/bin/env python3
# scripts/check_environment.py

import sys
import subprocess
import platform

def check_python_version():
    print(f"Python版本: {sys.version}")
    return sys.version

def check_jax():
    try:
        import jax
        print(f"JAX版本: {jax.__version__}")
        print(f"JAX设备: {jax.devices()}")
        return jax.__version__
    except ImportError:
        print("JAX未安装")
        return None

def check_cuda():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA信息:")
            print(result.stdout)
            return True
        else:
            print("nvidia-smi命令失败")
            return False
    except FileNotFoundError:
        print("nvidia-smi未找到")
        return False

def check_system_info():
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"主机名: {platform.node()}")

if __name__ == "__main__":
    print("=== 环境检查 ===")
    check_system_info()
    check_python_version()
    check_jax()
    check_cuda()
```

## ⚠️ 常见问题和解决方案

### 1. 版本不一致问题
```bash
# 问题: JAX版本不同
# 解决: 在所有节点上重新安装相同版本
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. CUDA版本问题
```bash
# 问题: CUDA版本不匹配
# 解决: 确保所有节点使用相同的CUDA版本
# 检查CUDA版本
nvcc --version
nvidia-smi
```

### 3. 网络连接问题
```bash
# 测试节点间连通性
ping <其他节点IP>
telnet <其他节点IP> 29500

# 检查防火墙设置
sudo ufw status
sudo ufw allow 29500
```

### 4. 权限问题
```bash
# 确保用户有足够权限
sudo usermod -a -G docker $USER
sudo chown -R $USER:$USER /path/to/project
```

## 📋 环境检查清单

### 部署前检查
- [ ] 所有节点的Python版本一致
- [ ] 所有节点的JAX版本一致
- [ ] 所有节点的CUDA版本一致
- [ ] 所有节点的代码版本一致
- [ ] 网络连通性正常
- [ ] 防火墙设置正确
- [ ] 用户权限配置正确

### 运行时检查
- [ ] 所有进程能正常启动
- [ ] 进程间通信正常
- [ ] GPU资源分配正确
- [ ] 数据加载正常
- [ ] 训练进度同步

## 🚀 自动化部署脚本

### 创建部署脚本
```bash
#!/bin/bash
# scripts/deploy_to_nodes.sh

NODES=("node1" "node2" "node3")
PROJECT_PATH="/path/to/project"

for node in "${NODES[@]}"; do
    echo "部署到节点: $node"
    
    # 同步代码
    rsync -avz --exclude='.git' --exclude='__pycache__' \
        $PROJECT_PATH/ user@$node:$PROJECT_PATH/
    
    # 同步环境
    scp requirements.txt user@$node:$PROJECT_PATH/
    
    # 在远程节点上安装依赖
    ssh user@$node "cd $PROJECT_PATH && \
        source .venv/bin/activate && \
        pip install -r requirements.txt"
    
    echo "节点 $node 部署完成"
done
```

## 📊 环境监控

### 监控脚本
```python
# scripts/monitor_nodes.py
import subprocess
import time

def monitor_gpu_usage(nodes):
    while True:
        for node in nodes:
            try:
                result = subprocess.run(
                    f'ssh user@{node} nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits',
                    shell=True, capture_output=True, text=True
                )
                print(f"{node}: {result.stdout.strip()}")
            except Exception as e:
                print(f"监控 {node} 失败: {e}")
        time.sleep(10)
```
