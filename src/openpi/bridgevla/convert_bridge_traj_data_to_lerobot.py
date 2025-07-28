"""
将数据集转换为 LeRobot 格式的最小示例脚本。

本例使用 bridge 数据集（存储在 RLDS 中），但可以很容易地修改为任何其他自定义格式的数据。

用法：
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /path/to/your/data

如果你想将数据集推送到 Hugging Face Hub，可以使用以下命令：
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

注意：运行本脚本需要安装 tensorflow_datasets：
`uv pip install tensorflow tensorflow_datasets`

转换后的数据集将被保存到 $LEROBOT_HOME 目录下。
运行本转换脚本大约需要 30 分钟。
"""

import os
from pathlib import Path
import pickle
import shutil

from einops import rearrange

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
import tyro

# from openpi.bridgevla.libs.point_renderer_main.point_renderer. \
#     rvt_renderer import RVTBoxRenderer as BoxRenderer
from openpi.bridgevla.mvt import config as mvt_config

# from openpi.bridgevla.utils.rvt_utils import move_pc_in_bound
# from openpi.bridgevla.mvt.utils import place_pc_in_cube

os.environ["http_proxy"] = "http://localhost:10808"
os.environ["https_proxy"] = "http://localhost:10808"


# 使用用户主目录下的路径
HOME = str(Path.home())
os.environ["LEROBOT_HOME"] = os.path.join(HOME, "pi0_bridge/datasets")
# REPO_NAME = "lddddl/dobot_formate_0611"  # 数据集名称
REPO_NAME = None
OUTPUT_PATH = os.path.join(HOME, "vla/pi0_bridge/datasets/converted_dataset")  # 输出目录
# DATASET_PATH = os.path.join(HOME, '/home/BridgeVLA/data/202507013')
SCENE_BOUNDS_REAL = [
    -1.1,
    -0.6,
    -0.2,
    0.2,
    0.5,
    0.6,
]
PLACE_WITH_MEAN = True
PCD2RGB = False
HORIZON = "1"

def read_episode_data(episode_path):
    """
    读取新的数据格式中的episode数据
    """
    episode_path = os.path.join(episode_path, "pkl")
    pkl_files = [f for f in os.listdir(episode_path) if f.endswith(".pkl")]
    if not pkl_files:
        raise ValueError(f"No pkl files found in {episode_path}")

    pkl_file = os.path.join(episode_path, pkl_files[0])
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    return data

def process_image_from_file(image_path):
    """从文件路径处理图像数据到正确的格式。
    Args:
        image_path: str, 图像文件路径
    Returns:
        numpy array, shape (C, 256, 256)
    """
    # 读取图像
    img = Image.open(image_path)
    img = np.array(img)

    # 确保只有RGB通道
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    elif len(img.shape) == 3 and img.shape[2] == 1:  # 灰度图
        img = np.stack([img[:, :, 0]] * 3, axis=2)

    # 转换为连续数组
    img = np.ascontiguousarray(img)
    # 确保数据类型是uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    # HWC -> CHW
    img = rearrange(img, "h w c -> c h w")
    # 下采样到256x256
    return downsample_nn(img, out_h=256, out_w=256)

def process_depth_from_file(depth_path):
    """从文件路径处理深度数据到正确的格式。
    Args:
        depth_path: str, 深度图像文件路径
    Returns:
        numpy array, shape (1, 256, 256)
    """
    # 读取深度图像
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img)

    # 如果是RGB格式的深度图，转换为单通道
    if len(depth.shape) == 3:
        if depth.shape[2] == 3:
            # 假设深度信息在第一个通道
            depth = depth[:, :, 0]
        elif depth.shape[2] == 4:
            depth = depth[:, :, 0]

    # 确保是2D数组
    if len(depth.shape) != 2:
        raise ValueError(f"Unexpected depth image shape: {depth.shape}")

    # 转换为连续数组
    depth = np.ascontiguousarray(depth)
    # 归一化到[0, 1]范围
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    # 添加通道维度
    depth = depth[None, :, :]  # (1, H, W)
    # 下采样到256x256
    depth = downsample_nn(depth, out_h=256, out_w=256)
    return depth.astype(np.float32)


def convert_pose_to_gripper_state(pose_data):
    """
    将pose数据转换为gripper状态
    Args:
        pose_data: dict, 包含end_pose, joint, degree等字段
    Returns:
        dict: 包含position, orientation, claw_status等字段
    """
    # end_pose格式: (x, y, z, rx, ry, rz)
    position = list(pose_data["end_pose"][:3])  # 前3个是位置
    orientation = list(pose_data["end_pose"][3:])  # 后3个是欧拉角

    # 根据degree判断夹爪状态，90度通常是关闭状态
    claw_status = 1 if pose_data["degree"] > 89 else 0

    return {
        "position": position,
        "orientation": orientation,
        "claw_status": claw_status,
        "arm_flag": 0,  # 默认值
        "timestamp": pose_data["timestamp"]
    }

def downsample_nn(img, out_h=224, out_w=224):
    """下采样图像到指定大小。
    Args:
        img: numpy array, shape (C, H, W)
        out_h: 输出高度
        out_w: 输出宽度
    Returns:
        numpy array, shape (C, out_h, out_w)
    """
    assert len(img.shape) == 3, "img must be a 3D array"
    c, h, w = img.shape
    row_idx = (np.linspace(0, h - 1, out_h)).astype(np.int64)
    col_idx = (np.linspace(0, w - 1, out_w)).astype(np.int64)
    # 使用meshgrid创建索引网格
    row_idx, col_idx = np.meshgrid(row_idx, col_idx, indexing="ij")
    return img[:, row_idx, col_idx]

def get_gripper_pos(gripper_pose, step):

    gripper_pose_xyz = np.array(gripper_pose[step]["position"]) / 1000  # mm -> m
    gripper_pose_euler = gripper_pose[step]["orientation"]
    gripper_pose_quat = R.from_euler("xyz", gripper_pose_euler, degrees=True).as_quat()
    gripper_pose_full = np.concatenate(
        (gripper_pose_xyz, gripper_pose_quat, [gripper_pose[step]["claw_status"]]), axis=0
    ).astype(np.float32)

    return gripper_pose_full  # noqa: RET504

def main(data_dir: str, device: str, horizon: int = 1, *, push_to_hub: bool = False):

    image_size = 256
    image_channel = 3
    dyn_cam_info = None
    mvt_cfg = mvt_config.get_cfg_defaults()
    dataset_name = data_dir.split("/")[-1]
    # 清理输出目录中已存在的数据集
    output_path = os.path.join(OUTPUT_PATH, REPO_NAME if REPO_NAME is not None else dataset_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # 确保输出目录的父目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建 LeRobot 数据集，定义要存储的特征
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        fps=25,
        features={
            "top_image": {
                "dtype": "image",
                "shape": (image_channel, image_size, image_size),
                "names": ["channel", "height", "width"],
            },
            # "depth": {
            #     "dtype": "float32",
            #     "shape": (1, image_size, image_size),
            #     "names": ["channel", "height", "width"],
            # },
            "state": {
                "dtype": "float32",
                "shape": (8 * horizon,),  # 修改为horizon倍的长度
                "names": ["state"],
            },
            "lang_goal": {
                "dtype": "string",
                "shape": (1,),
                "names": ["instruction"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8 * horizon,),  # 修改为horizon倍的长度
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    # renderer = BoxRenderer(
    #         device=device,
    #         img_size=(image_size, image_size),
    #         three_views=mvt_cfg.rend_three_views,
    #         with_depth=mvt_cfg.add_depth,
    #     )

    # 遍历数据集目录
    for task in os.listdir(data_dir):
        task_path = os.path.join(data_dir, task)
        if not os.path.isdir(task_path):
            continue

        print(f"Processing task: {task}")

        # 读取episode数据
        episode_data = read_episode_data(task_path)
        task_name = episode_data["task_name"]
        episode_list = episode_data["episode_data"]

        # 获取图像和深度图路径
        rgb_dir = os.path.join(task_path, "image", "rgb", "overhead_cam_rgb")
        depth_dir = os.path.join(task_path, "image", "depth", "overhead_cam_depth")

        # 获取所有图像文件
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

        print(f"Found {len(rgb_files)} RGB images and {len(depth_files)} depth images")
        print(f"Episode has {len(episode_list)} steps")

        # 确保图像数量和episode步数匹配
        min_steps = min(len(rgb_files), len(depth_files), len(episode_list))

        # 转换episode数据为gripper_pose格式
        gripper_pose = []
        for step_data in episode_list:
            gripper_pose.append(convert_pose_to_gripper_state(step_data))

        # ## test
        # endpos_list = []
        # jointpos_list = []
        # gripper_list = []
        # for ep in gripper_pose:
        #     endpos_list.append(ep['position'])
        #     jointpos_list.append(ep['orientation'])
        #     gripper_list.append(ep['claw_status'])
        # import matplotlib.pyplot as plt
        # x = range(len(endpos_list))
        # # 绘制三条曲线
        # plt.plot(x, endpos_list, linestyle='-', label='endpos')
        # plt.plot(x, jointpos_list, linestyle='--', label='joint')
        # plt.plot(x, gripper_list, linestyle='-.', label='gripper')

        # # 添加标题和轴标签
        # plt.title('三列表数据曲线示例')
        # plt.xlabel('X 轴')
        # plt.ylabel('Y 轴')

        # # 显示网格和图例
        # plt.grid(True)
        # plt.legend()

        # # 保存图像到文件
        # plt.savefig('three_series_plot.png', dpi=300, bbox_inches='tight')  # 保存为 PNG，分辨率 300 DPI

        # 根据horizon参数处理数据
        # 计算可以创建多少个完整的horizon序列
        assert horizon > 0
        max_frames = (min_steps - 1) // horizon  # 减1是因为最后一步没有action

        print(f"Creating {max_frames} frames with horizon={horizon}")

        for frame_idx in range(max_frames):
            # 计算当前frame对应的图像索引
            image_idx = frame_idx * horizon

            # 读取并处理RGB图像
            rgb_file = os.path.join(rgb_dir, rgb_files[image_idx])
            rgb = process_image_from_file(rgb_file)

            # 读取并处理深度图像
            depth_file = os.path.join(depth_dir, depth_files[image_idx])
            depth = process_depth_from_file(depth_file)

            # 收集当前horizon范围内的所有state和action
            states = []
            actions = []

            start_step = frame_idx * horizon
            end_step = start_step + horizon

            for step in range(start_step, end_step):
                # 处理gripper pose
                state = get_gripper_pos(gripper_pose=gripper_pose, step=step)
                gripper_pose_full = get_gripper_pos(gripper_pose=gripper_pose, step=step+1)

                states.append(state)
                actions.append(gripper_pose_full)

            # 将states和actions展平为一维数组
            states_flat = np.concatenate(states)
            actions_flat = np.concatenate(actions)

            # 添加帧到数据集
            dataset.add_frame(
                {
                    "top_image": rgb,
                    # "depth": depth,
                    "state": states_flat,
                    "lang_goal": task_name.strip(),
                    "action": actions_flat,
                    "task": task,
                }
            )

        dataset.save_episode()

    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["bridge", "dobot", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    import sys

    use_cuda_id = 0
    assert torch.cuda.is_available()
    device = f"cuda:{use_cuda_id}"

    sys.argv = [
        sys.argv[0],  # 脚本名
        # "--data_dir", '/home/BridgeVLA/data/202507013',  # 数据目录
        "--data_dir", "/home/lpy/vla/pi0_bridge/datasets/pi0_0728",  # 数据目录

        "--device", device,
        "--horizon", HORIZON,  # 添加horizon参数
        # "--push_to_hub",
    ]
    tyro.cli(main)
