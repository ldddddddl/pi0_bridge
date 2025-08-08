import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

print(f"JAX 版本: {jax.__version__}")
print(f"设备列表: {jax.devices()}")
print(f"后端平台: {xla_bridge.get_backend().platform}")
print(f"CUDA 路径: {xla_bridge.get_backend().platform_version}")

# 运行简单计算测试
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
z = x * y
print(f"计算结果: {z}")
