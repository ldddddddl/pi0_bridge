import jax, jax.lib.xla_bridge as xb
print("设备列表:", jax.devices())
print("device_count:", jax.device_count())
print("local_device_count:", jax.local_device_count())
print("XLA 后端:", xb.get_backend().platform)
print("jax 版本:", jax.__version__)
