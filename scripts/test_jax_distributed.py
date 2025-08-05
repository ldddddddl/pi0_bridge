#!/usr/bin/env python3
"""
测试JAX分布式环境配置
"""

import os
import jax
import jax.numpy as jnp
from train import init_distributed_environment

os.environ["MASTER_ADDR"] = "10.1.0.6"
os.environ["MASTER_PORT"] = "29500"
os.environ["WORLD_SIZE"] = "12"
os.environ["NODE_RANK"] = "1"
os.environ["RANK"] = "4"
os.environ["LOCAL_RANK"] = "0"

def test_jax_distributed():
    """测试JAX分布式环境"""
    print("=" * 60)
    print("JAX分布式环境测试")
    print("=" * 60)
    
    # 打印环境变量
    print("环境变量:")
    for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        print(f"  {key}: {os.environ.get(key, '未设置')}")
    
    # 初始化分布式环境
    print("\n初始化分布式环境...")
    dist_info = init_distributed_environment()
    
    print(f"分布式信息: {dist_info}")
    print(f"JAX process_count: {jax.process_count()}")
    print(f"JAX process_index: {jax.process_index()}")
    print(f"JAX device_count: {jax.device_count()}")
    print(f"JAX local_device_count: {jax.local_device_count()}")
    
    # 测试基本计算
    print("\n测试基本计算...")
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sin(x)
    print(f"sin({x}) = {y}")
    
    # 如果是分布式环境，测试通信
    if dist_info['is_distributed']:
        print("\n测试分布式通信...")
        try:
            # 创建一个简单的数组
            local_data = jnp.array([dist_info['rank'] * 10 + i for i in range(5)])
            print(f"Rank {dist_info['rank']} 本地数据: {local_data}")
            
            # 使用all_reduce测试通信
            global_data = jax.lax.all_reduce(local_data, 'sum')
            print(f"Rank {dist_info['rank']} 全局求和结果: {global_data}")
            
            print("分布式通信测试成功!")
        except Exception as e:
            print(f"分布式通信测试失败: {e}")
    else:
        print("单机模式，跳过分布式通信测试")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_jax_distributed() 