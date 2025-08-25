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

# os.environ["http_proxy"] = "http://localhost:10808"
# os.environ["https_proxy"] = "http://localhost:10808"


# 使用用户主目录下的路径
HOME = str(Path.home())
os.environ["LEROBOT_HOME"] = os.path.join(HOME, "pi0_bridge/datasets")
# REPO_NAME = "lddddl/dobot_formate_0611"  # 数据集名称
REPO_NAME = None
OUTPUT_PATH = os.path.join(HOME, "vla/pi0_bridge/datasets/converted_dataset")  # 输出目录
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
GRIPPER_RATIO = 25.

JOINT_LIMITS = np.array([360, 360, 160, 360, 360, 360, 100, 360, 360, 160, 360, 360, 360, 100])
is_use_delta_pose = True

def read_episode_data(data_dir, episode_index):
    """
    使用LeRobot官方方法读取microwave格式中的episode数据
    Args:
        data_dir: 数据集根目录
        episode_index: episode索引
    Returns:
        dict: 包含hf_dataset切片和任务信息
    """
    # 读取任务信息
    import json

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_name = data_dir.split("/")[-1]
    tasks_path = os.path.join(data_dir, "meta", "tasks.jsonl")
    tasks_dict = {}
    with open(tasks_path) as f:
        for line in f:
            task_data = json.loads(line.strip())
            tasks_dict[task_data["task_index"]] = task_data["task"]
    # 用LeRobotDataset读取
    lero_dataset = LeRobotDataset(
        repo_id=repo_name,  # 可自定义
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
            if delta[i] > 25 or delta[i] < -25:
                delta[i] = delta[i] / GRIPPER_RATIO
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


def _bgr_frame_to_chw256(frame_bgr):
    """将OpenCV读取的BGR帧转换为 CHW uint8 并缩放到 256x256。"""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    # HWC -> CHW
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    return chw

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

    # 根据CPU核数调整写盘并发
    num_cpus = os.cpu_count() or 8
    image_writer_processes = max(8, max(8, num_cpus // 2))
    image_writer_threads = 8

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
            "wrist_image_left": {
                "dtype": "image",
                "shape": (image_channel, image_size, image_size),
                "names": ["channel", "height", "width"],
            },
            "wrist_image_right": {
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
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


    # 读取episode信息
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes_info = []
    with open(episodes_path) as f:
        for line in f:
            episodes_info.append(json.loads(line.strip()))

    print(f"Found {len(episodes_info)} episodes to process")

    # 预加载源数据集并建立 episode -> 行索引 映射，避免每个episode重复加载/过滤
    # 注意：repo_id 应为数据集名称（非路径），root 指向本地数据集根目录
    src_reader = LeRobotDataset(repo_id=dataset_name, root=data_dir)
    hf_dataset_all = src_reader.hf_dataset
    if len(hf_dataset_all) == 0:
        print(f"[Warning] Preloaded HF dataset is empty at root={data_dir}. Will fallback to per-episode loading.")
    episode_index_column = hf_dataset_all["episode_index"]
    episode_to_indices = {}
    for row_index, epi_idx in enumerate(episode_index_column):
        episode_to_indices.setdefault(epi_idx.item(), []).append(row_index)

    # 处理每个episode
    for episode_info in episodes_info:
        episode_index = episode_info["episode_index"]
        print(f"Processing episode {episode_index}")

        try:
            # 读取episode数据（优先使用预加载hf_dataset的索引选择；若无则回退到按episode单次加载）
            indices = episode_to_indices.get(episode_index, [])
            if indices:
                ep_data = hf_dataset_all.select(indices)
            else:
                # fallback：仅加载该episode
                single_reader = LeRobotDataset(repo_id=dataset_name, root=data_dir, episodes=[episode_index])
                ep_data = single_reader.hf_dataset
                if len(ep_data) == 0:
                    print(f"No data rows found for episode {episode_index}, skip.")
                    continue
            task_name = episode_info["tasks"][0]

            # 获取视频路径
            video_path = os.path.join(data_dir, "videos", "chunk-000", "observation.images.overhead_cam_rgb", f"episode_{episode_index:06d}.mp4")
            left_video_path = os.path.join(data_dir, "videos", "chunk-000", "observation.images.wrist_cam_left", f"episode_{episode_index:06d}.mp4")
            right_video_path = os.path.join(data_dir, "videos", "chunk-000", "observation.images.wrist_cam_right", f"episode_{episode_index:06d}.mp4")
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}, skipping episode {episode_index}")
                continue
            if not os.path.exists(left_video_path):
                print(f"Video file not found: {left_video_path}, skipping episode {episode_index}")
                continue
            if not os.path.exists(right_video_path):
                print(f"Video file not found: {right_video_path}, skipping episode {episode_index}")
                continue

            # 处理数据
            episode_length = len(ep_data)
            # 将需要的列一次性转为numpy，减少循环内开销
            ep_states_arr = np.array(ep_data["observation.state"], dtype=np.float32)
            ep_actions_arr = np.array(ep_data["action"], dtype=np.float32)
            state_delta_list = []
            action_delta_list = []
            action_list = []

            # 每个episode只打开一次视频并按需顺序读取
            top_cap = cv2.VideoCapture(video_path)
            left_cap = cv2.VideoCapture(left_video_path)
            right_cap = cv2.VideoCapture(right_video_path)
            if not top_cap.isOpened():
                print(f"Cannot open video: {video_path}, skipping episode {episode_index}")
                continue
            if not left_cap.isOpened():
                print(f"Cannot open video: {left_video_path}, skipping episode {episode_index}")
                continue
            if not right_cap.isOpened():
                print(f"Cannot open video: {right_video_path}, skipping episode {episode_index}")
                continue

            if horizon == 1:
                # 读掉第0帧（循环从i=1开始时需要帧i）
                top_ret, _ = top_cap.read()
                left_ret, _ = left_cap.read()
                right_ret, _ = right_cap.read()
                if not top_ret or not left_ret or not right_ret:
                    top_cap.release()
                    left_cap.release()
                    right_cap.release()
                    print(f"Cannot read first frame from {video_path}, skipping episode {episode_index}")
                    continue
                for i in tqdm.tqdm(range(1, episode_length)):
                    prev_state = ep_states_arr[i-1]
                    prev_action = ep_actions_arr[i-1]
                    curr_action = ep_actions_arr[i]
                    action_list.append(np.array(curr_action, dtype=np.float32))
                    # 计算delta
                    action_delta = convert_state_to_delta_pose(prev_action, curr_action)
                    state_delta_list.append(prev_state)
                    action_delta_list.append(action_delta)
                    # 顺序读取当前帧
                    ret, frame_bgr = top_cap.read()
                    left_ret, left_frame_bgr = left_cap.read()
                    right_ret, right_frame_bgr = right_cap.read()
                    if not ret or not left_ret or not right_ret:
                        break
                    top_rgb = _bgr_frame_to_chw256(frame_bgr)
                    left_rgb = _bgr_frame_to_chw256(left_frame_bgr)
                    right_rgb = _bgr_frame_to_chw256(right_frame_bgr)
                    # 存储
                    dataset.add_frame({
                        "top_image": top_rgb,
                        "wrist_image_left": left_rgb, 
                        "wrist_image_right": right_rgb, 
                        "state": prev_state,
                        "lang_goal": task_name,
                        "action": action_delta,
                        "task": task_name,
                    })

            else:
                # 预计算需要用到的帧索引（避免随机seek）
                num_samples = (episode_length - 1) // horizon
                needed_indices = {frame_idx * horizon for frame_idx in range(num_samples)}
                frame_cache = {}
                if len(needed_indices) > 0:
                    max_needed = max(needed_indices)
                    current_index = 0
                    while current_index <= max_needed:
                        ret, frame_bgr = top_cap.read()
                        if not ret:
                            break
                        if current_index in needed_indices:
                            frame_cache[current_index] = _bgr_frame_to_chw256(frame_bgr)
                        current_index += 1

                for frame_idx in tqdm.tqdm(range((episode_length - 1) // horizon)):
                    data_idx = frame_idx * horizon
                    top_rgb = frame_cache.get(data_idx)
                    left_rgb = frame_cache.get(data_idx)
                    right_rgb = frame_cache.get(data_idx)
                    if top_rgb is None or left_rgb is None or right_rgb is None:
                        # 若缓存缺失该帧，则跳过
                        continue
                    states = []
                    actions = []
                    for step in range(horizon):
                        current_idx = data_idx + step
                        next_idx = current_idx + 1
                        if next_idx >= episode_length:
                            break
                        current_state = ep_states_arr[current_idx]
                        next_action = ep_actions_arr[next_idx]
                        states.append(current_state)
                        actions.append(next_action)
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
                        "wrist_image_left": left_rgb,
                        "wrist_image_right": right_rgb,
                        "state": states_flat,
                        "lang_goal": task_name,
                        "action": actions_flat,
                        "task": task_name,
                    })

            # 释放视频句柄
            top_cap.release()
            left_cap.release()
            right_cap.release()
                    
            colors = plt.cm.tab20(np.linspace(0, 1, 14))
            # episode结束后画图
            if len(state_delta_list) > 0:
                state_delta_arr = np.stack(state_delta_list, axis=0)
                plt.figure(figsize=(16, 8))
                # 使用不同的颜色映射确保每个维度都有不同颜色
                for i in range(14):
                    plt.plot(state_delta_arr[:, i], label=f"state_dim_{i}", color=colors[i])
                plt.title(f"Episode {episode_index} State Delta")
                plt.xlabel("Frame")
                plt.ylabel("Delta Value")
                plt.legend()
                plt.savefig(f"{output_path}/state_delta_plot_ep{episode_index}.png")
                plt.close()
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
                
            dataset.save_episode()
            ### test
        except Exception as e:
            print(f"Error processing episode {episode_index}: {e}")
            continue

    plt.imsave(f"{output_path}/top_image_ep{episode_index}_frame{i}.png", top_rgb.transpose(1, 2, 0))
    plt.imsave(f"{output_path}/wrist_image_left_ep{episode_index}_frame{i}.png", left_rgb.transpose(1, 2, 0))
    plt.imsave(f"{output_path}/wrist_image_right_ep{episode_index}_frame{i}.png", right_rgb.transpose(1, 2, 0))

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
        "--data_dir", f"{HOME}/vla/pi0_bridge/datasets/openmicrowave_3camera_dualarm_joint",  # pai0_microwave数据目录
        "--device", device,
        "--horizon", HORIZON,  # 添加horizon参数
        # "--push_to_hub",
    ]
    tyro.cli(main)
