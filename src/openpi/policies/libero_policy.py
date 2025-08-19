import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """为Libero策略创建一个随机输入示例。"""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    此类用于将模型输入转换为预期格式。它用于训练和推理。

    对于您自己的数据集，您可以复制此类并根据下面的注释修改键，以将数据集的正确元素传输到模型中。
    """

    # 模型的动作维度。将用于为pi0模型（非pi0-FAST）填充状态和动作。
    # 对于您自己的数据集，请勿更改此值。
    action_dim: int

    # 确定将使用哪个模型。
    # 对于您自己的数据集，请勿更改此值。
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 我们只对pi0模型进行填充掩码，而不是pi0-FAST。对于您自己的数据集，请勿更改此值。
        mask_padding = self.model_type == _model.ModelType.PI0

        # 我们将本体感受输入填充到模型的动作维度。
        # 对于pi0-FAST，我们不填充状态。对于Libero，我们不需要区分，
        # 因为pi0-FAST的action_dim = 7，小于state_dim = 8，所以跳过填充。
        # 为您的数据集保留此设置，但如果您的数据集将本体感受输入存储在不同的键中
        # 而不是"observation/state"，您应该更改下面的键。
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # 可能需要将图像解析为uint8 (H,W,C)，因为LeRobot自动存储为float32 (C,H,W)，
        # 策略推理时会跳过此步骤。
        # 为您的数据集保留此设置，但如果您的数据集将图像存储在不同的键中
        # 而不是"observation/image"或"observation/wrist_image"，
        # 您应该更改下面的键。
        # Pi0模型目前支持三种图像输入：一个第三人称视角，
        # 和两个手腕视角（左和右）。如果您的数据集没有特定类型的图像，
        # 例如手腕图像，您可以在此处注释掉它，并用零数组替换它，就像我们对
        # 右手腕图像所做的那样。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建输入字典。请勿更改下面字典中的键。
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 用适当形状的零数组填充任何不存在的图像。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 如果``mask_padding``为True，则用False掩码任何不存在的图像。
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # 将动作填充到模型动作维度。为您的数据集保留此设置。
        # 动作仅在训练期间可用。
        if "actions" in data:
            # 我们正在填充到模型动作维度。
            # 对于pi0-FAST，这是无操作（因为action_dim = 7）。
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # 将提示（即语言指令）传递给模型。
        # 为您的数据集保留此设置（但如果指令不是存储在"prompt"中，请修改键；
        # 输出字典始终需要具有键"prompt"）。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    此类用于将模型输出转换回数据集特定格式。它仅用于推理。

    对于您自己的数据集，您可以复制此类并根据下面的注释修改动作维度。
    """

    def __call__(self, data: dict) -> dict:
        # 只返回前N个动作——由于我们上面将动作填充到模型动作维度，
        # 我们现在需要在返回字典中解析出正确数量的动作。
        # 对于pi0_bridge，我们返回前8个动作（因为模型是8维的）。
        # 对于您自己的数据集，用数据集的动作维度替换`8`。
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    """
    此类用于将模型输出转换回数据集特定格式。它仅用于推理。

    对于pi0_bridge，我们返回前8个动作（因为模型是8维的）。
    """

    def __call__(self, data: dict) -> dict:
        # 只返回前N个动作——由于我们上面将动作填充到模型动作维度，
        # 我们现在需要在返回字典中解析出正确数量的动作。
        # 对于pi0_bridge，我们返回前8个动作（因为模型是8维的）。
        # 对于您自己的数据集，用数据集的动作维度替换`8`。
        return {"actions": np.asarray(data["actions"][:, :7])}

@dataclasses.dataclass(frozen=True)
class BridgeDualArmOutputs(transforms.DataTransformFn):
    """
    此类用于将模型输出转换回数据集特定格式。它仅用于推理。

    对于pi0_bridge_traj，我们返回前14个动作（因为模型是14维的）。
    """

    def __call__(self, data: dict) -> dict:
        # 只返回前N个动作——由于我们上面将动作填充到模型动作维度，
        # 我们现在需要在返回字典中解析出正确数量的动作。
        # 对于pi0_bridge，我们返回前8个动作（因为模型是8维的）。
        # 对于您自己的数据集，用数据集的动作维度替换`8`。
        return {"actions": np.asarray(data["actions"][:, :14])}