"""
将pai0_microwave数据集转换为 LeRobot 格式的脚本。

本脚本将pai0_microwave格式的数据（parquet文件+MP4视频）转换为LeRobot格式。

用法：
python convert_bridge_traj_data_from_lerobot_to_lerobot.py --data_dir /path/to/pai0_microwave

如果你想将数据集推送到 Hugging Face Hub，可以使用以下命令：
python convert_bridge_traj_data_from_lerobot_to_lerobot.py --data_dir /path/to/pai0_microwave --push_to_hub

注意：运行本脚本需要安装相关依赖：
pandas, opencv-python, lerobot等

转换后的数据集将被保存到 $LEROBOT_HOME 目录下。
"""

import json
import os
from pathlib import Path
import shutil

import cv2
from einops import rearrange
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
import tyro
import tqdm

# from openpi.bridgevla.libs.point_renderer_main.point_renderer. \
#     rvt_renderer import RVTBoxRenderer as BoxRenderer

# from openpi.bridgevla.utils.rvt_utils import move_pc_in_bound
# from openpi.bridgevla.mvt.utils import place_pc_in_cube

# os.environ["http_proxy"] = "http://localhost:10808"
# os.environ["https_proxy"] = "http://localhost:10808"


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

DOBOT_CR5_JOINTS = [360, 360, 160, 360, 360, 360]
is_use_delta_pose = True

def read_episode_data(data_dir, episode_index):
    """
    使用LeRobot官方方法读取pai0_microwave格式中的episode数据
    Args:
        data_dir: 数据集根目录
        episode_index: episode索引
    Returns:
        dict: 包含hf_dataset切片和任务信息
    """
    # 读取任务信息
    import json

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    tasks_path = os.path.join(data_dir, "meta", "tasks.jsonl")
    tasks_dict = {}
    with open(tasks_path) as f:
        for line in f:
            task_data = json.loads(line.strip())
            tasks_dict[task_data["task_index"]] = task_data["task"]
    # 用LeRobotDataset读取
    lero_dataset = LeRobotDataset(
        repo_id="pai0_microwave",  # 可自定义
        root=data_dir,
        episodes=[episode_index],
    )
    hf_dataset = lero_dataset.load_hf_dataset()
    # 取出当前episode的所有数据
    ep_data = hf_dataset.filter(lambda x: x["episode_index"] == episode_index)
    # 获取该episode的任务
    task_index = ep_data["task_index"][0]
    task_name = tasks_dict.get(task_index, "unknown_task")
    return {
        "dataframe": ep_data,
        "task_name": task_name,
        "task_index": task_index,
        "episode_length": len(ep_data)
    }

def extract_frame_from_video(video_path, frame_index):
    """从MP4视频中提取指定帧
    Args:
        video_path: str, 视频文件路径
        frame_index: int, 帧索引
    Returns:
        numpy array, shape (C, 256, 256)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # 设置帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_index} from {video_path}")

    # OpenCV读取的是BGR格式，转换为RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 转换为连续数组
    img = np.ascontiguousarray(img)
    # 确保数据类型是uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    # HWC -> CHW
    img = rearrange(img, "h w c -> c h w")
    # 下采样到256x256
    return downsample_nn(img, out_h=256, out_w=256)

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


def convert_state_to_gripper_format(state_data):
    """
    将pai0_microwave的14维state数据转换为gripper格式
    Args:
        state_data: numpy array, 14维状态数据 [left_px, left_py, left_pz, left_rx, left_ry, left_rz, left_gripper,
                                            right_px, right_py, right_pz, right_rx, right_ry, right_rz, right_gripper]
    Returns:
        dict: 包含position, orientation, claw_status等字段
    """
    # 提取左臂数据 (前7维)
    left_position = state_data[:3] / 1000.0  # mm -> m
    left_orientation = state_data[3:6]  # 欧拉角
    left_gripper = state_data[6]

    # 提取右臂数据 (后7维)
    right_position = state_data[7:10] / 1000.0  # mm -> m
    right_orientation = state_data[10:13]  # 欧拉角
    right_gripper = state_data[13]

    # 转换为四元数
    left_quat = R.from_euler("xyz", left_orientation, degrees=True).as_quat()
    right_quat = R.from_euler("xyz", right_orientation, degrees=True).as_quat()

    # 合并为单一格式（使用左臂作为主臂，右臂信息也保留）
    combined_position = np.concatenate([left_position, left_quat, [left_gripper/100.0]])

    return {
        "left_position": left_position,
        "left_orientation": left_quat,
        "left_gripper": left_gripper/100.0,
        "right_position": right_position,
        "right_orientation": right_quat,
        "right_gripper": right_gripper/100.0,
        "combined_state": combined_position.astype(np.float32)
    }

def convert_state_to_delta_pose(current_state, next_state):
    """
    按照convert_pose_to_delta_pose的规则处理pai0_microwave 14维state/action
    Args:
        current_state: numpy array, shape (14,)
        next_state: numpy array, shape (14,)
    Returns:
        numpy array, shape (14,)
    """
    # 关节极限（假设与DOBOT_CR5_JOINTS一致，左臂+右臂）
    # [x, y, z, rx, ry, rz, gripper] * 2
    JOINT_LIMITS = np.array([360, 360, 160, 360, 360, 360, 100, 360, 360, 160, 360, 360, 360, 100])
    delta = next_state - current_state
    for i in range(14):
        # 只对前6维和后6维做极限修正，gripper不做
        if i % 7 < 6:
            if delta[i] > 300 or delta[i] < -300:
                abs_delta = JOINT_LIMITS[i] - abs(delta[i])
                if current_state[i] >= 0:
                    delta[i] = abs_delta
                else:
                    delta[i] = -abs_delta
        else:
            # claw state 100:open 0:close
            if delta[i] > 25:
                delta[i] = 1.
            elif delta[i] < -25:
                delta[i] = -1.
            else:
                delta[i] = 0
    return delta.astype(np.float32)


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

def get_gripper_pos_from_state(state_data):
    """
    从pai0_microwave的state数据获取gripper位置
    Args:
        state_data: numpy array, 14维状态数据
    Returns:
        numpy array: 处理后的gripper状态
    """
    converted = convert_state_to_gripper_format(state_data)
    return converted["combined_state"]

def main(data_dir: str, device: str, horizon: int = 1, *, push_to_hub: bool = False):

    image_size = 256
    image_channel = 3
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
            "wrist_image": {
                "dtype": "image",
                "shape": (image_channel, image_size, image_size),
                "names": ["channel", "height", "width"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14 * horizon,),  # 14维双臂机器人状态
                "names": ["state"],
            },
            "lang_goal": {
                "dtype": "string",
                "shape": (1,),
                "names": ["instruction"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14 * horizon,),  # 14维双臂机器人动作
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    # 读取episode信息
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes_info = []
    with open(episodes_path) as f:
        for line in f:
            episodes_info.append(json.loads(line.strip()))

    print(f"Found {len(episodes_info)} episodes to process")

    # 处理每个episode
    for episode_info in episodes_info:
        episode_index = episode_info["episode_index"]
        print(f"Processing episode {episode_index}")

        try:
            # 读取episode数据
            episode_data = read_episode_data(data_dir, episode_index)
            df = episode_data["dataframe"]
            task_name = episode_info["tasks"][0]

            # 获取视频路径
            video_path = os.path.join(data_dir, "videos", "chunk-000", "observation.images.overhead_cam", f"episode_{episode_index:06d}.mp4")
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}, skipping episode {episode_index}")
                continue

            # 处理数据
            ep_data = df  # 现在df是datasets.Dataset
            episode_length = len(ep_data)
            state_delta_list = []
            action_delta_list = []
            action_list = []
            if horizon == 1:
                for i in tqdm.tqdm(range(1, episode_length)):
                    prev_state = ep_data["observation.state"][i-1]
                    curr_state = ep_data["observation.state"][i]
                    prev_action = ep_data["action"][i-1]
                    curr_action = ep_data["action"][i]
                    action_list.append(np.array(curr_action, dtype=np.float32))
                    # 计算delta
                    state_delta = convert_state_to_delta_pose(np.array(prev_state), np.array(curr_state))
                    action_delta = convert_state_to_delta_pose(np.array(prev_action), np.array(curr_action))
                    state_delta_list.append(state_delta)
                    action_delta_list.append(action_delta)
                    # 取图像
                    top_rgb = extract_frame_from_video(video_path, i)
                    wrist_rgb = extract_frame_from_video(video_path, i)
                    # 存储
                    dataset.add_frame({
                        "top_image": top_rgb,
                        "wrist_image": wrist_rgb,
                        "state": state_delta,
                        "lang_goal": task_name,
                        "action": action_delta,
                        "task": task_name,
                    })
            else:
                for frame_idx in tqdm.tqdm(range((episode_length - 1) // horizon)):
                    data_idx = frame_idx * horizon
                    top_rgb = extract_frame_from_video(video_path, data_idx)
                    wrist_rgb = extract_frame_from_video(video_path, data_idx)
                    states = []
                    actions = []
                    for step in range(horizon):
                        current_idx = data_idx + step
                        next_idx = current_idx + 1
                        if next_idx >= episode_length:
                            break
                        current_state = ep_data["observation.state"][current_idx]
                        next_action = ep_data["action"][next_idx]
                        states.append(np.array(current_state, dtype=np.float32))
                        actions.append(np.array(next_action, dtype=np.float32))
                    if len(states) < horizon:
                        continue
                    states = np.array(states)
                    actions = np.array(actions)
                    states_delta = np.zeros_like(states)
                    actions_delta = np.zeros_like(actions)
                    if len(states) > 1:
                        for i in range(1, len(states)):
                            states_delta[i] = convert_state_to_delta_pose(states[i-1], states[i])
                            actions_delta[i] = convert_state_to_delta_pose(actions[i-1], actions[i])
                    # 第0帧delta默认就是0
                    state_delta_list.extend(states_delta)
                    action_delta_list.extend(actions_delta)
                    states_flat = states_delta.flatten()
                    actions_flat = actions_delta.flatten()
                    dataset.add_frame({
                        "top_image": top_rgb,
                        "wrist_image": wrist_rgb,
                        "state": states_flat,
                        "lang_goal": task_name,
                        "action": actions_flat,
                        "task": task_name,
                    })
                    
            colors = plt.cm.tab20(np.linspace(0, 1, 14))
            # episode结束后画图
            # if len(state_delta_list) > 0:
            #     state_delta_arr = np.stack(state_delta_list, axis=0)
            #     plt.figure(figsize=(16, 8))
            #     # 使用不同的颜色映射确保每个维度都有不同颜色
            #     for i in range(14):
            #         plt.plot(state_delta_arr[:, i], label=f"state_dim_{i}", color=colors[i])
            #     plt.title(f"Episode {episode_index} State Delta")
            #     plt.xlabel("Frame")
            #     plt.ylabel("Delta Value")
            #     plt.legend()
            #     plt.savefig(f"{output_path}/state_delta_plot_ep{episode_index}.png")
            #     plt.close()
            if len(action_delta_list) > 0:
                action_delta_arr = np.stack(action_delta_list, axis=0)
                plt.figure(figsize=(16, 8))
                # 使用不同的颜色映射确保每个维度都有不同颜色
                for i in range(14):
                    plt.plot(action_delta_arr[:, i], label=f"action_dim_{i}", color=colors[i])
                plt.title(f"Episode {episode_index} Action Delta")
                plt.xlabel("Frame")
                plt.ylabel("Delta Value")
                plt.legend()
                plt.savefig(f"{output_path}/action_delta_plot_ep{episode_index}.png")
                plt.close()
                
            if len(action_list) > 0:
                action_arr = np.stack(action_list, axis=0)
                plt.figure(figsize=(16, 8))
                # 使用不同的颜色映射确保每个维度都有不同颜色
                for i in range(14):
                    plt.plot(action_arr[:, i], label=f"action_dim_{i}", color=colors[i])
                plt.title(f"Episode {episode_index} Action")
                plt.xlabel("Frame")
                plt.ylabel("Action Value")
                plt.legend()
                plt.savefig(f"{output_path}/action_plot_ep{episode_index}.png")
                plt.close()

            dataset.save_episode()
            
        except Exception as e:
            print(f"Error processing episode {episode_index}: {e}")
            continue

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
        "--data_dir", f"{HOME}/vla/pi0_bridge/datasets/pai0_microwave",  # pai0_microwave数据目录
        "--device", device,
        "--horizon", HORIZON,  # 添加horizon参数
        # "--push_to_hub",
    ]
    tyro.cli(main)
