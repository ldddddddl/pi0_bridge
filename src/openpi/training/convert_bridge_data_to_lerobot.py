"""
将数据集转换为 LeRobot 格式的最小示例脚本。

本例使用 bridge 数据集（存储在 RLDS 中），但可以很容易地修改为任何其他自定义格式的数据。

用法：
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /path/to/your/data

如果你想将数据集推送到 Hugging Face Hub，可以使用以下命令：
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

注意：运行本脚本需要安装 tensorflow_datasets：
`uv pip install tensorflow tensorflow_datasets`

你可以从 https://huggingface.co/datasets/openvla/modified_bridge_rlds 下载原始 bridge 数据集。
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
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import tyro

os.environ["http_proxy"] = "http://localhost:10808"
os.environ["https_proxy"] = "http://localhost:10808"


# 使用用户主目录下的路径
HOME = str(Path.home())
os.environ["LEROBOT_HOME"] = os.path.join(HOME, "pi0_bridge/datasets")
REPO_NAME = "lddddl/dobot_formate_0611"  # 数据集名称
OUTPUT_PATH = os.path.join(HOME, "pi0_bridge/datasets/converted_dataset")  # 输出目录
DATASET_PATH = os.path.join(HOME, "pi0_bridge/datasets/dobot_formate_0611")


def read_action_file(action_path):
    """
    文件内容类似如下
    Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz) Claw Status
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    """
    with open(action_path, "rb") as f:
        data_str = pickle.load(f)

    # Split the string into lines
    lines = data_str.strip().split("\n")

    # Initialize the result list
    result = []

    # Process each line of data
    for i, line in enumerate(lines):
        if i == 0:  # Skip header
            continue

        # Split the line into components
        parts = line.strip().split()

        # Extract timestamp
        timestamp = parts[0]

        # Extract position (X,Y,Z)
        position = [float(x) for x in parts[1:4]]

        # Extract orientation (Rx,Ry,Rz)
        orientation = [float(x) for x in parts[4:7]]

        if len(parts) == 9:
            claw_status = int(parts[7])
            arm_flag = int(parts[8])
        elif len(parts) == 8:
            claw_status = int(parts[7])
            arm_flag = 0
        else:
            if i == 1 or i == 2 or i == 5:
                claw_status = 0
            else:
                claw_status = 1
            arm_flag = 0

        # Create dictionary for this entry
        entry = {
            "timestamp": timestamp,
            "position": position,
            "orientation": orientation,
            "claw_status": claw_status,
            "arm_flag": arm_flag,
        }

        result.append(entry)

    return result


def convert_pcd_to_base(extrinsic_path, pcd, type="3rd"):
    with open(extrinsic_path, "rb") as f:
        data = pickle.load(f)
        transform = np.array(data)

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)
    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd = (transform @ pcd.T).T[:, :3]
    pcd = pcd.reshape(h, w, 3)
    return pcd


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


def process_image(img_data):
    """处理图像数据到正确的格式。
    Args:
        img_data: numpy array, shape (H, W, C)
    Returns:
        numpy array, shape (C, 256, 256)
    """
    # 确保只有RGB通道
    img = img_data[:, :, :3]
    # 转换为连续数组
    img = np.ascontiguousarray(img)
    # 确保数据类型是uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    # HWC -> CHW
    img = rearrange(img, "h w c -> c h w")
    # 下采样到256x256
    img = downsample_nn(img, out_h=256, out_w=256)
    return img


def process_pcd(pcd_data, extrinsic_path):
    """处理点云数据到正确的格式。
    Args:
        pcd_data: numpy array, shape (H, W, C)
        extrinsic_path: str, 外参矩阵路径
    Returns:
        numpy array, shape (C, 256, 256)
    """
    # 只保留XYZ通道
    pcd = pcd_data[:, :, :3]
    # 转换坐标系
    pcd = convert_pcd_to_base(pcd=pcd, extrinsic_path=extrinsic_path)
    # HWC -> CHW
    pcd = rearrange(pcd, "h w c -> c h w")
    # 下采样到256x256
    pcd = downsample_nn(pcd, out_h=256, out_w=256)
    # 归一化到[0, 1]范围
    pcd = (pcd - pcd.min()) / (pcd.max() - pcd.min())
    return pcd.astype(np.float32)


def get_gripper_pos(gripper_pose, step):
    
    gripper_pose_xyz = np.array(gripper_pose[step]["position"]) / 1000  # mm -> m
    gripper_pose_euler = gripper_pose[step]["orientation"]
    gripper_pose_quat = R.from_euler("xyz", gripper_pose_euler, degrees=True).as_quat()
    gripper_pose_full = np.concatenate(
        (gripper_pose_xyz, gripper_pose_quat, [gripper_pose[step]["claw_status"]]), axis=0
    ).astype(np.float32)

    return gripper_pose_full

def main(data_dir: str, *, push_to_hub: bool = False):

    # 清理输出目录中已存在的数据集
    output_path = os.path.join(OUTPUT_PATH, REPO_NAME)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # 确保输出目录的父目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建 LeRobot 数据集，定义要存储的特征
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        fps=10,
        features={
            "top_image": {
                "dtype": "image",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
            },
            "front_image": {
                "dtype": "image",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
            },
            "right_image": {
                "dtype": "image",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
            },
            "pcd": {
                "dtype": "float32",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
            },
            # "gripper_pose": {
            #     "dtype": "float32",
            #     "shape": (8,),  # position(3) + quaternion(4) + claw_status(1)
            #     "names": ["pose"],
            # },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # current_gripper_state(1) + time(1)
                "names": ["state"],
            },
            "lang_goal": {
                "dtype": "string",
                "shape": (1,),
                "names": ["instruction"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历数据集目录
    for task in os.listdir(data_dir):
        task_path = os.path.join(data_dir, task)
        if not os.path.isdir(task_path):
            continue

        for episode_num in tqdm(os.listdir(task_path)):
            episode_path = os.path.join(task_path, episode_num)

            action_path = os.path.join(episode_path, "pose.pkl")
            rgb_3rd = os.path.join(episode_path, "zed_rgb")
            pcd_3rd = os.path.join(episode_path, "zed_pcd")
            gripper_pose = read_action_file(action_path)

            num_steps = sum(1 for file_name in os.listdir(rgb_3rd) if file_name.endswith(".pkl"))

            # 读取指令
            with open(os.path.join(episode_path, "instruction.pkl"), "rb") as f:
                instruction = pickle.load(f)

            for step in range(num_steps - 1):
                # 处理gripper pose
                state = get_gripper_pos(gripper_pose=gripper_pose, step=step)
                gripper_pose_full = get_gripper_pos(gripper_pose=gripper_pose, step=step+1)

                # 处理low_dim_state
                current_gripper_state = gripper_pose[step]["claw_status"]
                time = (1.0 - (step / float(num_steps - 1))) * 2.0 - 1.0
                low_dim_state = np.array([current_gripper_state, time], dtype=np.float32)

                # 读取并处理RGB图像
                with open(os.path.join(rgb_3rd, f"{step}.pkl"), "rb") as f:
                    rgb = process_image(pickle.load(f))

                # 读取并处理点云数据
                with open(os.path.join(pcd_3rd, f"{step}.pkl"), "rb") as f:
                    pcd = process_pcd(pickle.load(f), extrinsic_path=os.path.join(episode_path, "extrinsic_matrix.pkl"))

                # 添加帧到数据集
                dataset.add_frame(
                    {
                        "top_image": rgb,
                        "front_image": rgb,
                        "right_image": rgb,
                        "pcd": pcd,
                        # "gripper_pose": gripper_pose_full,
                        "state": state,
                        "lang_goal": instruction.strip(),
                        "action": gripper_pose_full,
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

    sys.argv = [
        sys.argv[0],  # 脚本名
        "--data_dir",
        os.path.join(HOME, "vla/pi0_bridge/datasets/dobot_formate_0611"),  # 数据目录
        "--push_to_hub",
    ]
    tyro.cli(main)
