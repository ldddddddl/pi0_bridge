import jax
from jax._src import xla_bridge

# 检查 JAX 是否尝试加载 CUDA 插件
print("Backend:", xla_bridge.get_backend().platform)

# 检查可能的插件名称（不同版本可能不同）
print("Possible plugin names:", dir(xla_bridge))
