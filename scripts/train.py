import dataclasses
import datetime
import functools
import logging
import platform
import os
import sys
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# 顶层定义Subset类，便于PyTorch DataLoader多进程pickle
class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]

    def __len__(self):
        return len(self.indices)


def init_distributed_environment():
    """初始化分布式训练环境"""
    # 设置分布式训练环境变量
    if "SLURM_PROCID" in os.environ:
        # SLURM环境
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_rank = int(os.environ["SLURM_NODEID"])
        
        # 设置JAX分布式环境
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        
        # 设置分布式训练端口
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
            
    elif "RANK" in os.environ:
        # 手动设置的环境变量
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        node_rank = int(os.environ.get("NODE_RANK", 0))
        
        # 设置JAX分布式环境变量
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        
        # 设置JAX分布式训练环境变量
        os.environ["JAX_COORDINATOR_ADDRESS"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["JAX_COORDINATOR_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
    else:
        # 单机训练
        rank = 0
        world_size = 1
        local_rank = 0
        node_rank = 0
    
    # 设置CUDA设备
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    
    # 如果是分布式训练，初始化JAX分布式环境
    if world_size > 1:
        try:
            # 设置JAX分布式环境变量
            os.environ["JAX_COORDINATOR_ADDRESS"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["JAX_COORDINATOR_PORT"] = os.environ.get("MASTER_PORT", "29500")
            
            # 添加超时机制和重试逻辑
            import time
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    # 初始化JAX分布式环境 - 这会阻塞直到所有进程都启动
                    jax.distributed.initialize(
                        coordinator_address=f"{os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '29500')}",
                        num_processes=world_size,
                        process_id=rank,
                        local_device_ids=[local_rank]
                    )
                    print(f"JAX分布式初始化成功: rank={rank}/{world_size}, process_count={jax.process_count()}, process_index={jax.process_index()}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"JAX分布式初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        print(f"JAX分布式初始化最终失败: {e}")
                        raise
                        
        except Exception as e:
            print(f"JAX分布式初始化失败: {e}")
            raise
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "is_distributed": world_size > 1
    }


def init_logging(dist_info):
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt=f"%(asctime)s.%(msecs)03d [%(levelname)s] [R{dist_info['rank']}/W{dist_info['world_size']}] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True, dist_info=None):
    # 只在主进程上初始化wandb
    if dist_info and dist_info["rank"] != 0:
        wandb.init(mode="disabled")
        return
        
    ct = datetime.datetime.now()
    strf_time = ct.strftime("%Y-%m-%d-%H-%M-%S")

    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name+strf_time,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def partial_load(model_params, loaded_params):
    """
    递归地只加载权重文件中有的参数，缺失的参数保持模型初始化的值。
    """
    if isinstance(model_params, dict) and isinstance(loaded_params, dict):
        out = {}
        for k in model_params:
            if k in loaded_params:
                out[k] = partial_load(model_params[k], loaded_params[k])
            else:
                out[k] = model_params[k]
        return out
    return loaded_params if loaded_params is not None else model_params


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and partially merges the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # 不再严格检查结构，只做部分加载
    merged_params = partial_load(params_shape, loaded_params)
    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(merged_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        total_loss = jnp.mean(chunked_loss)
        # 用 stop_gradient + float() 保证副产物为Python标量，避免nnx.value_and_grad报错
        return total_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=False)(
        model, train_rng, observation, actions
    )
    # mse_loss_in, mse_loss_out = model.get_mse_loss()

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        def ema_update(old, new):
            if isinstance(old, jnp.ndarray) and isinstance(new, jnp.ndarray):
                # 只对浮点类型做EMA
                if jnp.issubdtype(old.dtype, jnp.floating) and jnp.issubdtype(new.dtype, jnp.floating):
                    return state.ema_decay * old + (1 - state.ema_decay) * new
                return new
            return new
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                ema_update,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "train_total_loss": loss,
        # "train_mse_loss_in": mse_loss_in,
        # "train_mse_loss_out": mse_loss_out,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def valid_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        total_loss = jnp.mean(chunked_loss)
        # 用 stop_gradient + float() 保证副产物为Python标量，避免nnx.value_and_grad报错
        return total_loss

    # mse_loss_in, mse_loss_out = model.get_mse_loss()
    valid_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss = loss_fn(model, valid_rng, observation, actions)
    # 这里只返回loss，后续可扩展更多指标
    info = {"valid_total_loss": loss,
            # "valid_mse_loss_in": mse_loss_in,
            # "valid_mse_loss_out": mse_loss_out
            }
    return info


def main(config: _config.TrainConfig):
    print("[Debug] 进入 main() 函数")
    # 初始化分布式环境
    print("[Debug] 开始分布式环境初始化")
    dist_info = init_distributed_environment()
    print(f"[Debug] 分布式环境初始化完成: rank={dist_info['rank']}, world_size={dist_info['world_size']}, local_rank={dist_info['local_rank']}, node_rank={dist_info['node_rank']}, is_distributed={dist_info['is_distributed']}")
    init_logging(dist_info)
    
    logging.info(f"Running on: {platform.node()} (Rank {dist_info['rank']}/{dist_info['world_size']})")
    logging.info(f"JAX process_count: {jax.process_count()}, process_index: {jax.process_index()}")
    logging.info(f"JAX device_count: {jax.device_count()}, local_device_count: {jax.local_device_count()}")

    print(f"[Debug][Rank {dist_info['rank']}] 开始数据加载配置")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    
    # 改进Mesh创建逻辑，确保与分布式设备数量匹配
    if dist_info["is_distributed"]:
        # 在分布式环境下，使用所有可用设备
        fsdp_devices = jax.device_count()
        print(f"[Debug][Rank {dist_info['rank']}] 分布式环境，使用 {fsdp_devices} 个设备")
    else:
        # 单机环境下，使用配置的设备数量
        fsdp_devices = config.fsdp_devices
        print(f"[Debug][Rank {dist_info['rank']}] 单机环境，使用 {fsdp_devices} 个设备")
    
    mesh = sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(f"[Debug][Rank {dist_info['rank']}] 开始检查点管理和wandb初始化")
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled, dist_info=dist_info)
    print(f"[Debug][Rank {dist_info['rank']}] 检查点和wandb初始化完成")
    # breakpoint()
    valid_data_size = config.valid_data_size  # 验证集比例
    valid_interval = config.valid_interval  # 每多少步验证一次

    print(f"[Debug][Rank {dist_info['rank']}] 开始加载完整数据集")
    full_data_config = config.data.create(config.assets_dirs, config.model)
    full_dataset = _data_loader.create_torch_dataset(full_data_config, config.model.action_horizon, config.model)
    print(f"[Debug][Rank {dist_info['rank']}] 完整数据集加载完成，样本数: {len(full_dataset)}")
    full_len = len(full_dataset)
    valid_len = int(full_len * valid_data_size)
    train_len = full_len - valid_len
    indices = np.arange(full_len)
    np.random.seed(config.seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_len]
    valid_indices = indices[train_len:]

    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices) if valid_len > 0 else train_dataset

    # 用DataLoaderImpl包装，保证输出(batch_observation, batch_actions)
    def make_loader(dataset, shuffle, sharding, batch_size, num_workers, seed):
        # 先transform
        dataset = _data_loader.transform_dataset(dataset, full_data_config)
        torch_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=batch_size // jax.process_count(),
            sharding=sharding,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
        )
        # DataLoaderImpl包装，返回(Observation, Actions)
        return _data_loader.DataLoaderImpl(full_data_config, torch_loader)

    print(f"[Debug][Rank {dist_info['rank']}] 开始创建train_loader和valid_loader")
    # 改进worker数量配置
    if dist_info["is_distributed"]:
        # 在分布式环境下使用少量worker，避免资源竞争
        num_workers = min(2, config.num_workers) if config.num_workers > 0 else 0
        print(f"[Debug][Rank {dist_info['rank']}] 分布式环境，使用 {num_workers} 个worker")
    else:
        # 单机环境下使用配置的worker数量
        num_workers = config.num_workers
        print(f"[Debug][Rank {dist_info['rank']}] 单机环境，使用 {num_workers} 个worker")
    
    train_loader = make_loader(train_dataset, True, data_sharding, config.batch_size, num_workers, config.seed)
    valid_loader = make_loader(valid_dataset, False, data_sharding, config.batch_size, 0, config.seed + 1)
    print(f"[Debug][Rank {dist_info['rank']}] train_loader和valid_loader创建完成")

    train_data_iter = iter(train_loader)
    valid_data_iter = iter(valid_loader)
    print(f"[Debug][Rank {dist_info['rank']}] 开始获取第一个batch")
    batch = next(train_data_iter)
    print(f"[Debug][Rank {dist_info['rank']}] 第一个batch获取完成")

    logging.info(f"Train/Valid split: train={train_len}, valid={valid_len}")
    logging.info(f"Initialized train loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check (only on main process)
    if dist_info["rank"] == 0:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    print(f"[Debug][Rank {dist_info['rank']}] 开始模型初始化和权重加载")
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    print(f"[Debug][Rank {dist_info['rank']}] 模型初始化完成，等待进程就绪")
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    print(f"[Debug][Rank {dist_info['rank']}] 模型初始化和权重加载完成")

    # 添加分布式训练状态检查
    if dist_info["is_distributed"]:
        print(f"[Debug][Rank {dist_info['rank']}] 检查分布式训练状态...")
        # 确保所有进程都到达这个点
        jax.block_until_ready(jax.device_get(jnp.array([dist_info['rank']])))
        print(f"[Debug][Rank {dist_info['rank']}] 分布式训练状态检查完成")
        
        # 验证设备分配
        expected_devices = jax.device_count()
        actual_devices = len(jax.devices())
        if expected_devices != actual_devices:
            print(f"[Warning][Rank {dist_info['rank']}] 设备数量不匹配: 期望 {expected_devices}, 实际 {actual_devices}")
        
        # 验证进程信息
        print(f"[Debug][Rank {dist_info['rank']}] 进程信息: process_count={jax.process_count()}, process_index={jax.process_index()}")

    if resuming:
        print(f"[Debug][Rank {dist_info['rank']}] 恢复训练状态")
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader)
        print(f"[Debug][Rank {dist_info['rank']}] 训练状态恢复完成")

    print(f"[Debug][Rank {dist_info['rank']}] 开始训练主循环")
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=dist_info["rank"] != 0,  # 只在主进程显示进度条
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            if dist_info["rank"] == 0:  # 只在主进程打印日志
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
        # 每valid_interval步进行一次验证
        if step % valid_interval == 0 and valid_len > 0:
            try:
                valid_batch = next(valid_data_iter)
            except StopIteration:
                valid_data_iter = iter(valid_loader)
                valid_batch = next(valid_data_iter)
            valid_info = valid_step(config, train_rng, train_state, valid_batch)
            valid_info = jax.device_get(valid_info)
            if dist_info["rank"] == 0:  # 只在主进程打印验证日志
                pbar.write(f"Step {step}: valid_loss={valid_info['valid_total_loss']:.4f}")
                wandb.log(valid_info, step=step)
        try:
            batch = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_loader)
            batch = next(train_data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # 只在主进程保存检查点
            if dist_info["rank"] == 0:
                _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    if dist_info["rank"] == 0:
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    import os
    import sys

    # 在 VSCode 里直接 Run 时，先设置好环境变量
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    # 设置使用的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # 使用第一张GPU，可以改为"0,1,2"来使用多张卡

    # 然后把 sys.argv "伪造" 成你在终端里敲的那条命令
    sys.argv = [
        sys.argv[0],  # 脚本名
        "pi0_bridge_traj",  # 第一个位置参数
        "--exp-name",
        "pi0_bridge_traj",
        "--overwrite",
        "--data.repo-id",
        "/home/ubuntu/vla/pi0_bridge/datasets/converted_dataset/dataset0729",
    ]

    main(_config.cli())
