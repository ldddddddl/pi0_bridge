"""查看 _CONFIGS 获取可用配置列表。"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 解决直接使用 nnx.filterlib.Filter 的 tyro 问题
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """决定用于设置数据管道的资源（如标准化统计数据）的位置。

    这些资源将被复制到检查点中的 `assets/asset_id` 目录中。

    这可以用于从不同的检查点（例如基础模型检查点）或其他集中位置加载资源。
    例如，要在微调期间从基础模型检查点加载 Trossen 机器人的标准化统计数据，使用：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # 资源目录。如果未提供，将使用配置的 assets_dirs。这对于从不同的检查点（例如基础模型检查点）
    # 或其他集中位置加载资源很有用。
    assets_dir: str | None = None

    # 资源 ID。如果未提供，将使用仓库 ID。这允许用户引用描述不同机器人平台的资源。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot 仓库 ID。如果为 None，将创建假数据。
    repo_id: str | None = None
    root: str | None = None
    # 资源目录中包含数据资源的目录。
    asset_id: str | None = None
    # 包含预计算的标准化统计数据。如果为 None，将不执行标准化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 用于将数据集特定格式的输入调整为数据转换期望的格式。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据转换，通常包括机器人特定的转换。将在数据标准化之前应用。
    # 参见 `model.Observation` 和 `model.Actions` 获取标准化数据。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型特定的转换。将在数据标准化之后应用。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 如果为 true，将使用分位数标准化。否则，将使用正态 z-分数标准化。
    use_quantile_norm: bool = False

    # 数据加载器将用于生成动作序列的键名。序列长度由模型配置中的 `action_horizon` 字段定义。
    # 如果您的 LeRobot 数据集使用不同的键来表示动作，您应该调整这个。
    action_sequence_keys: Sequence[str] = ("action",)

    # 如果为 true，将从 LeRobot 数据集的 task 字段加载提示。
    prompt_from_task: bool = False

    # 仅用于 RLDS 数据加载器（即，目前仅用于 DROID）。
    rlds_data_dir: str | None = None
    # DROID 数据集的动作空间。
    action_space: droid_rlds_dataset.DroidActionSpace | None = None



class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建一个转换组。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准 pi0 模型创建模型转换。"""

    # 如果提供，将决定模型将使用的默认提示。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot 仓库 ID
    repo_id: str = tyro.MISSING
    # 决定如何加载资源
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 将被工厂更新的基础配置
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。"""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"从 {data_assets_dir} 加载标准化统计数据")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"在 {data_assets_dir} 中未找到标准化统计数据，跳过。")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # 数据转换的工厂
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型转换的工厂
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # 如果为 true，关节维度将在传递给模型之前转换为相对于当前状态的增量。
    # 夹持器维度将保持绝对值。
    use_delta_joint_actions: bool = True
    # 如果提供，当"prompt"键不存在时将被注入到输入数据中。
    default_prompt: str | None = None
    # 如果为 true，这将把关节和夹持器值从标准 Aloha 空间转换为用于训练基础模型的 pi 内部运行时空间。
    # 使用标准 Aloha 数据的人应该将此设置为 true。
    adapt_to_pi: bool = True

    # 重新打包转换。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "top_image"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 将用于从数据集读取动作序列的动作键。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    此配置用于配置在数据管道的各个部分应用的转换。
    对于您自己的数据集，您可以复制此类并根据下面的注释修改转换以匹配您的数据集。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 重新打包转换仅应用于来自数据集的数据，而不是在推理期间应用。
        # 我们可以使用它使数据集的输入尽可能接近来自推理环境的输入（例如匹配键）。
        # 下面，我们将数据集中的键（在数据转换脚本中定义）与我们在推理管道中使用的键（在 libero 的推理脚本中定义）匹配。
        # 对于您自己的数据集，首先确定您的环境传递给策略服务器的键，
        # 然后修改下面的映射，使您的数据集的键与这些目标键匹配。
        # 重新打包转换在这里只是重新映射键名。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_image",
                        # "observation/wrist_image": "front_image",
                        # "observation/right_image": "right_image",
                        "observation/state": "state",
                        # "observation/gripper_pose": "gripper_pose",
                        "actions": "action",
                        "prompt": "lang_goal",
                    }
                )
            ]
        )

        # 数据转换应用于来自数据集的数据和推理期间。
        # 下面，我们定义了进入模型的数据转换（"inputs"）和来自模型的数据转换（"outputs"）（后者仅在推理期间使用）。
        # 我们在 `libero_policy.py` 中定义了这些转换。您可以在那里查看详细注释，了解如何修改转换以匹配您的数据集。
        # 一旦您创建了自己的转换，您可以用自己的转换替换下面的转换。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 另一个数据转换：pi0 模型在增量动作上训练（相对于每个动作块中的第一个状态）。
        # 如果您的数据有"绝对"动作（例如目标关节角度），您可以取消注释以下行以将动作转换为增量动作。
        # 唯一的例外是夹持器动作，它们始终是绝对的。
        # 在下面的示例中，我们将对前 6 个动作（关节）应用增量转换，并保持第 7 个动作（夹持器）不变，即绝对。
        # 在 Libero 中，数据集中的原始动作已经是增量动作，所以我们不需要应用单独的增量转换（这就是为什么它被注释掉）。
        # 根据您的数据集是使用"绝对"还是"增量"动作来选择是否应用此转换。

        # TODO(karl): 一旦我们更新了 Libero 检查点以不使用增量动作转换，就注释掉这个
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # 模型转换包括对提示和动作目标进行标记化等操作
        # 对于您自己的数据集，您不需要在这里更改任何内容。
        model_transforms = ModelTransformFactory()(model_config)

        # 我们返回所有用于训练和推理的数据转换。这里不需要更改任何内容。
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    用于在 RLDS 数据格式上训练 DROID 的配置（用于在更大的数据集上进行高效训练）。
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation.image",
                        "observation/wrist_image_left": "observation.wrist_image",
                        "observation/joint_position": "observation.joint_position",
                        "observation/gripper_position": "observation.gripper_position",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 数据加载器返回绝对关节位置动作 -- 转换为增量用于训练。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "必须为 RLDS 数据加载器设置 rlds 数据目录。"

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置的名称。必须唯一。将用于引用此配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称。将用于命名元数据和检查点目录。
    exp_name: str = tyro.MISSING

    # 定义模型配置。某些属性（action_dim、action_horizon 和 max_token_len）由所有模型共享
    # -- 参见 BaseModelConfig。特定的模型实现（例如，Pi0Config）继承自 BaseModelConfig，
    # 可能定义其他属性。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # 权重加载器可以在模型初始化后从磁盘加载权重（可能是部分的）
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定应该冻结哪些权重。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 确定要训练的数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资源的基础目录（例如，标准化统计数据）
    assets_base_dir: str = "./assets"
    # 检查点的基础目录
    checkpoint_base_dir: str = "./checkpoints"

    # 训练期间用于随机生成器的随机种子
    seed: int = 42
    # 全局批量大小
    batch_size: int = 32
    # 用于数据加载器的工作进程数。增加这个数字将加快数据加载速度，
    # 但会增加内存和 CPU 使用率。
    num_workers: int = 2
    # 要运行的训练步骤（批次）数
    num_train_steps: int = 10_000

    # 记录训练指标的频率（以步骤为单位）
    log_interval: int = 100
    # 保存检查点的频率（以步骤为单位）
    save_interval: int = 5000
    # 如果设置，匹配 step % keep_period == 0 的现有检查点将不会被删除。
    keep_period: int | None = 5000

    # 如果为 true，如果检查点目录存在，它将被覆盖。
    overwrite: bool = True
    # 如果为 true，训练将从最后一个检查点恢复
    resume: bool = False

    # 如果为 true，将启用 wandb 日志记录
    wandb_enabled: bool = True

    # 用于传递元数据到策略服务器
    policy_metadata: dict[str, Any] | None = None

    # 如果值大于 1，将启用 FSDP 并在指定数量的设备上分片；总体设备内存将减少，
    # 但训练可能会变慢。
    # 例如，如果总设备数为 4，fsdp 设备数为 2；那么模型将被分片到 2 个设备，
    # 并在 2 组设备之间运行数据并行。
    fsdp_devices: int = 2

    # 末端位置维度
    end_pos_dim: int = 8
    # 输出格式
    output_format: str = "end_pos"
    # 验证集比例
    valid_data_size: float = 0.2
    # 验证间隔
    valid_interval: int = 100        
    # device id 
    device: str = "1, 2, 3"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取此配置的资源目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        import datetime
        ct = datetime.datetime.now()
        t_formated = ct.strftime("%Y-%m-%d-%H-%M")
        """获取此配置的检查点目录。"""
        if not self.exp_name:
            raise ValueError("必须设置 --exp_name")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name / t_formated).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取可训练参数过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("不能同时恢复和覆盖。")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Aloha 推理配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # DROID 推理配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            repo_id="lerobot/droid_100",
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Libero 微调配置。
    #
    # 这些训练配置定义了在您自己的数据集上微调基础模型的超参数。
    # 它们用于定义关键元素，例如您要训练的数据集、您使用的基础检查点，
    # 以及其他超参数，例如要运行多少训练步骤或使用什么学习率。
    # 对于您自己的数据集，您可以复制此类并根据下面的注释修改数据集名称和数据转换。
    TrainConfig(
        # 更改名称以反映您的模型和数据集。
        name="pi0_libero",
        # 定义模型配置 -- 在此示例中，我们使用 pi0 作为模型架构
        # 并执行*完整*微调。在下面的示例中，我们展示如何修改此配置
        # 以执行*低内存*（LORA）微调，并使用 pi0-FAST 作为替代架构。
        model=pi0.Pi0Config(),
        # 定义您要训练的数据集。在此示例中，我们使用 Libero 数据集。
        # 对于您自己的数据集，您可以更改 repo_id 以指向您的数据集。
        # 同时修改 DataConfig 以使用您在上面为您的数据集创建的新配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 此标志决定是否从 LeRobot 数据集的 `task` 字段加载提示（即任务说明）。
                # 如果设置为 True，提示将在输入字典中显示为 `prompt` 字段。建议设置为 True。
                prompt_from_task=True,
            ),
        ),
        # 定义要加载哪个预训练检查点来初始化模型。
        # 这应该与您上面选择的模型配置匹配 -- 在这种情况下，我们使用 pi0 基础模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 在下面，您可以定义其他超参数，例如学习率、训练步骤等。
        # 查看基础 TrainConfig 类获取可用超参数的完整列表。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_bridge_single",
        model=pi0.Pi0Config(action_dim=32, end_pos_dim=8, action_horizon=1, max_token_len=180),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/zk/vla/pi0_bridge/datasets/converted_dataset/202507013",
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="trossen",
            # ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "top_image",
                            },
                            "state": "state",
                            "actions": "action",
                            "prompt": "lang_goal",
                            
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=20_000,
    ),
    
    #
    # Bridge 微调配置。
    #
    # 这些训练配置定义了在您自己的数据集上微调基础模型的超参数。
    # 它们用于定义关键元素，例如您要训练的数据集、您使用的基础检查点，
    # 以及其他超参数，例如要运行多少训练步骤或使用什么学习率。
    # 对于您自己的数据集，您可以复制此类并根据下面的注释修改数据集名称和数据转换。
    TrainConfig(
        # 更改名称以反映您的模型和数据集。
        name="pi0_bridge",
        # 定义模型配置 -- 在此示例中，我们使用 pi0 作为模型架构
        # 并执行*完整*微调。在下面的示例中，我们展示如何修改此配置
        # 以执行*低内存*（LORA）微调，并使用 pi0-FAST 作为替代架构。
        # max_token_len 一个好的经验法则是单臂机器人使用约 180，双臂机器人使用约 250。
        model=pi0.Pi0Config(action_dim=32, end_pos_dim=8, action_horizon=1, max_token_len=180),
        # 定义您要训练的数据集。在此示例中，我们使用 Libero 数据集。
        # 对于您自己的数据集，您可以更改 repo_id 以指向您的数据集。
        # 同时修改 DataConfig 以使用您在上面为您的数据集创建的新配置。
        data=LeRobotLiberoDataConfig(
            repo_id="/home/lpy/vla/pi0_bridge/datasets/converted_dataset/202507013",
            # repo_id='/datasets/converted_dataset/202507013',
            base_config=DataConfig(
                # 此标志决定是否从 LeRobot 数据集的 `task` 字段加载提示（即任务说明）。
                # 如果设置为 True，提示将在输入字典中显示为 `prompt` 字段。建议设置为 True。
                prompt_from_task=True,
                root='/home/lpy/vla/pi0_bridge/datasets/converted_dataset/202507013',
                repo_id='/home/lpy/vla/pi0_bridge/datasets/converted_dataset/202507013',
            ),
        ),
        # 定义要加载哪个预训练检查点来初始化模型。
        # 这应该与您上面选择的模型配置匹配 -- 在这种情况下，我们使用 pi0 基础模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        # 在下面，您可以定义其他超参数，例如学习率、训练步骤等。
        # 查看基础 TrainConfig 类获取可用超参数的完整列表。
        # num_train_steps=30_000,
        # num_train_steps=200,
        # batch_size=1,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 这是一个加载 pi0 模型进行 LoRA 微调的示例。
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # 冻结过滤器定义了在训练期间应该冻结哪些参数。
        # 我们在模型配置中有一个便利函数，它返回 LoRA 微调的默认冻结过滤器。
        # 只需确保它与您上面选择的模型配置匹配。
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 这是一个加载 pi0-FAST 模型进行完整微调的示例。
        # 修改 action_dim 和 action_horizon 以匹配您的数据集（action horizon 等于
        # 所需动作块的长度）。
        # max_token_len 是模型可以处理的最大（非图像）标记数。
        # 这包括标记化的提示、身体感知状态和（FAST 标记化的）动作标记。
        # 选择这个值太小可能会截断序列末尾的标记（代码会发出警告），
        # 而选择太大会浪费内存（因为我们将每个批次元素填充到 max_token_len）。
        # 一个好的经验法则是单臂机器人使用约 180，双臂机器人使用约 250。
        # 通常，我们首先选择一个较小的值，如果在训练期间看到许多警告被发出，
        # 考虑增加这个值。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        # 注意这里我们加载 pi0-FAST 基础模型检查点。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 这是一个加载 pi0-FAST 模型进行 LoRA 微调的示例。
        # 关于设置 action_dim、action_horizon 和 max_token_len，请参考上面的注释。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 再次确保在提取 LoRA 微调的冻结过滤器时，冻结过滤器与模型配置匹配
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    #
    # Aloha 微调配置。
    #
    # 这是一个用于演示如何在自定义 LeRobot 数据集上训练的测试配置。
    # 有关如何转换和训练您自己的 Aloha 数据集的说明，请参见 examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # DROID 微调配置。
    #
    TrainConfig(
        name="pi0_fast_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid_100",
            # 将此设置为您的 DROID RLDS 数据集的路径（`droid` 目录的父目录）
            rlds_data_dir="~/.cache/huggingface/lerobot/lerobot/droid_100",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k 步应该足够，在 8x H100s 上大约需要 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 重要：RLDS DataLoader 需要 num_workers=0，内部处理多进程
    ),
    #
    # ALOHA Sim 配置。此配置用于演示如何在简单的模拟环境中进行训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=1,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=1,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("配置名称必须唯一。")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """通过名称获取配置。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" 您是否想要 '{closest[0]}'？ " if closest else ""
        raise ValueError(f"未找到配置 '{config_name}'。{closest_str}")

    return _CONFIGS_DICT[config_name]
