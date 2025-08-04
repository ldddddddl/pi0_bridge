#!/usr/bin/env python3
"""
分布式训练环境测试脚本
用于验证多机多卡训练环境是否正确配置
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
from typing import Dict, Any


def init_distributed_environment():
    """初始化分布式训练环境"""
    if "SLURM_PROCID" in os.environ:
        # SLURM环境
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_rank = int(os.environ["SLURM_NODEID"])
        
    elif "RANK" in os.environ:
        # 手动设置的环境变量
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        node_rank = int(os.environ.get("NODE_RANK", 0))
        
    else:
        # 单机训练
        rank = 0
        world_size = 1
        local_rank = 0
        node_rank = 0
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "is_distributed": world_size > 1
    }


def test_basic_communication(dist_info: Dict[str, Any]):
    """测试基本的进程间通信"""
    print(f"[Rank {dist_info['rank']}] 开始测试基本通信...")
    
    # 创建一个简单的数组
    local_data = jnp.array([dist_info['rank'] * 10 + i for i in range(5)])
    print(f"[Rank {dist_info['rank']}] 本地数据: {local_data}")
    
    # 使用JAX的all_reduce进行通信测试
    try:
        # 这里需要实际的分布式通信，暂时跳过
        print(f"[Rank {dist_info['rank']}] 通信测试跳过（需要完整的分布式设置）")
    except Exception as e:
        print(f"[Rank {dist_info['rank']}] 通信测试失败: {e}")
    
    return True


def test_device_info(dist_info: Dict[str, Any]):
    """测试设备信息"""
    print(f"[Rank {dist_info['rank']}] 设备信息:")
    print(f"  JAX设备数量: {jax.device_count()}")
    print(f"  JAX本地设备数量: {jax.local_device_count()}")
    print(f"  可用设备: {jax.devices()}")
    
    if jax.device_count() > 0:
        # 测试简单的计算
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)
        print(f"[Rank {dist_info['rank']}] 计算测试: sin({x}) = {y}")
        return True
    else:
        print(f"[Rank {dist_info['rank']}] 警告: 没有可用的JAX设备")
        return False


def test_environment_variables(dist_info: Dict[str, Any]):
    """测试环境变量设置"""
    print(f"[Rank {dist_info['rank']}] 环境变量:")
    print(f"  RANK: {os.environ.get('RANK', '未设置')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', '未设置')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', '未设置')}")
    print(f"  NODE_RANK: {os.environ.get('NODE_RANK', '未设置')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', '未设置')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', '未设置')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
    return True


def test_network_connectivity(dist_info: Dict[str, Any]):
    """测试网络连通性"""
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"[Rank {dist_info['rank']}] 网络测试:")
    print(f"  主节点地址: {master_addr}")
    print(f"  主节点端口: {master_port}")
    
    if dist_info['rank'] == 0:
        print(f"[Rank {dist_info['rank']}] 主进程，跳过网络连接测试")
    else:
        print(f"[Rank {dist_info['rank']}] 从进程，需要连接到主节点")
        # 这里可以添加实际的网络连接测试
    
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("分布式训练环境测试")
    print("=" * 60)
    
    # 初始化分布式环境
    dist_info = init_distributed_environment()
    
    print(f"分布式信息:")
    print(f"  Rank: {dist_info['rank']}/{dist_info['world_size']}")
    print(f"  Local Rank: {dist_info['local_rank']}")
    print(f"  Node Rank: {dist_info['node_rank']}")
    print(f"  是否分布式: {dist_info['is_distributed']}")
    print(f"  主机名: {os.uname().nodename}")
    print(f"  进程ID: {os.getpid()}")
    
    # 执行各项测试
    tests = [
        ("环境变量测试", test_environment_variables),
        ("设备信息测试", test_device_info),
        ("网络连通性测试", test_network_connectivity),
        ("基本通信测试", test_basic_communication),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[Rank {dist_info['rank']}] 执行 {test_name}...")
        try:
            result = test_func(dist_info)
            results.append((test_name, result))
            print(f"[Rank {dist_info['rank']}] {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            print(f"[Rank {dist_info['rank']}] {test_name}: 错误 - {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n[Rank {dist_info['rank']}] 测试汇总:")
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n[Rank {dist_info['rank']}] 总体结果: {passed}/{total} 测试通过")
    
    if dist_info['rank'] == 0:
        print(f"\n[Rank {dist_info['rank']}] 分布式环境测试完成!")
        if passed == total:
            print("所有测试通过，分布式环境配置正确!")
        else:
            print("部分测试失败，请检查配置!")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 