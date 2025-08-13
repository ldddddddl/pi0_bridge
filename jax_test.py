import jax, jax.lib.xla_bridge as xb
print("设备列表:", jax.devices())
print("device_count:", jax.device_count())
print("local_device_count:", jax.local_device_count())
print("XLA 后端:", xb.get_backend().platform)
print("jax 版本:", jax.__version__)



import torch

print("torch 版本:", torch.__version__)
print("torch 设备列表:", torch.cuda.device_count())
print("torch 设备名称:", torch.cuda.get_device_name(0))
print("torch 设备类型:", torch.cuda.get_device_capability(0))