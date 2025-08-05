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
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
import tyro

from openpi.bridgevla.libs.point_renderer_main.point_renderer. \
    rvt_renderer import RVTBoxRenderer as BoxRenderer
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

def _preprocess_inputs_real(replay_sample, cameras):
    obs, pcds = [], []

    for n in cameras:

        rgb = replay_sample[n]["rgb"]
        pcd = replay_sample[n]["pcd"]

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    return pc, img_feat

def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert scene_bounds is not None
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans



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

def main(data_dir: str, device: str, *, push_to_hub: bool = False):

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
        fps=10,
        features={
            "top_image": {
                "dtype": "image",
                "shape": (image_channel, image_size, image_size),
                "names": ["channel", "height", "width"],
            },
            # "front_image": {
            #     "dtype": "image",
            #     "shape": (image_channel, image_size, image_size),
            #     "names": ["channel", "height", "width"],
            # },
            # "right_image": {
            #     "dtype": "image",
            #     "shape": (image_channel, image_size, image_size),
            #     "names": ["channel", "height", "width"],
            # },
            "pcd": {
                "dtype": "float32",
                "shape": (image_channel, image_size, image_size),
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


    renderer = BoxRenderer(
            device=device,
            img_size=(image_size, image_size),
            three_views=mvt_cfg.rend_three_views,
            with_depth=mvt_cfg.add_depth,
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
                # low_dim_state = np.array([current_gripper_state, time], dtype=np.float32)

                # 读取并处理RGB图像
                with open(os.path.join(rgb_3rd, f"{step}.pkl"), "rb") as f:
                    rgb = process_image(pickle.load(f))

                # 读取并处理点云数据
                with open(os.path.join(pcd_3rd, f"{step}.pkl"), "rb") as f:
                    pcd = process_pcd(pickle.load(f), extrinsic_path=os.path.join(episode_path, "extrinsic_matrix.pkl"))

                if PCD2RGB:
                    # convert point cloud to img
                    obs_dict = {
                        "3rd":{
                            "pcd":torch.from_numpy(pcd[None, ...]).to(device),
                            "rgb":torch.from_numpy(rgb[None, ...]).to(device)
                        }
                    }
                    obs, pcd_list = _preprocess_inputs_real(obs_dict, ["3rd"])
                    pc, img_feat = get_pc_img_feat(obs, pcd_list)
                    pc, img_feat = move_pc_in_bound(pc, img_feat, SCENE_BOUNDS_REAL, no_op=not move_pc_in_bound)
                    pc = [
                            place_pc_in_cube(
                                _pc,
                                with_mean_or_bounds=PLACE_WITH_MEAN,
                                scene_bounds=None if PLACE_WITH_MEAN else SCENE_BOUNDS_REAL,
                            )[0]
                            for _pc in pc
                        ]
                    # 确保数据类型一致
                    pc = pc[0].float()
                    img_feat = img_feat[0].float()
                    if dyn_cam_info is None:
                        dyn_cam_info_itr = (None,)
                    else:
                        dyn_cam_info_itr = dyn_cam_info
                    img = [
                        renderer(
                            _pc,
                            _img_feat,  # 只传图像特征，不拼接点云坐标
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if _dyn_cam_info is not None
                            else None,
                        ).squeeze()
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            [pc], [img_feat], dyn_cam_info_itr
                        )
                    ]
                    img = torch.cat(img, 0).squeeze().detach().cpu()
                    img = np.asarray(img)

                    # from matplotlib import pyplot as plt
                    # img = (img.clip(0,1) if img.max()<=1 else img/255.0)  # 归一化
                    # for i in range(img.shape[0]):
                    #     plt.imsave(f'img{i}.png', img[i])

                    # 添加帧到数据集
                    dataset.add_frame(
                        {
                            "top_image": img[0],
                            "front_image": img[1],
                            "right_image": img[2],
                            "pcd": pcd,
                            # "gripper_pose": gripper_pose_full,
                            "state": state,
                            "lang_goal": instruction.strip(),
                            "action": gripper_pose_full,
                            "task": task,
                        }
                    )
                else:

                    # 添加帧到数据集
                    dataset.add_frame(
                        {
                            "top_image": rgb,
                            # "front_image": img[1],
                            # "right_image": img[2],
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

    use_cuda_id = 0
    assert torch.cuda.is_available()
    device = f"cuda:{use_cuda_id}"

    sys.argv = [
        sys.argv[0],  # 脚本名
        # "--data_dir", '/home/BridgeVLA/data/202507013',  # 数据目录
        "--data_dir", f"{HOME}/vla/pi0_bridge/datasets/dobot_formate_0611",  # 数据目录

        "--device", device,
        # "--push_to_hub",
    ]
    tyro.cli(main)
