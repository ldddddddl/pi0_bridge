import os
import yaml
import sys
import numpy as np
import pickle
import torch
import cv2
import time
import collections

from IPython.core.splitinput import line_split
from scipy.spatial.transform import Rotation as R
import copy
from transforms3d.euler import euler2mat
import transforms3d
import dataclasses

from openpi_client import websocket_client_policy as _websocket_client_policy

sys.path.append("/home/zk/Projects/RVT_fiveage")
sys.path.append("/home/zk/Projects/RVT_fiveage/rvt_our/")

print(sys.path)


from multiprocessing import Value
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"


from botarm import Point

import open3d as o3d


CAMERAS = ["3rd"]
SCENE_BOUNDS = [
    -1.1,
    -0.6,
    -0.2,
    0.2,
    0.5,
    0.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000
    replan_steps: int = 1


def get_cam_extrinsic():
    trans = np.array([1.028818510131928, -0.04212360892289513, 0.6338377191806316])
    quat = np.array([-0.6333204911007358, -0.6400364927579377, 0.3240327100190967, 0.29027787777872616])  # x y z w

    transform = np.eye(4)
    rot = R.from_quat(quat)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = trans.T

    return transform


def vis_pcd(pcd, rgb):
    # 将点云和颜色转换为二维的形状 (N, 3)
    pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

    # 将点云和颜色信息保存为 PLY 文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
    pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
    # o3d.io.write_point_cloud(save_path, pcd)
    # 创建原点坐标系（size 可以根据需要设置）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])


def convert_pcd_to_base(
        type="3rd",
        pcd=[],
        extrinsic_matrix=None
):
    transform = extrinsic_matrix
    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)

    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


def vis_pcd_with_end_pred(pcd, rgb, end_pose=None, pred_pose=None, gt_pose=None,bounds=None):
    # 转换点云和颜色形状
    if bounds:
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        inv_pnt = (
            (pcd[:, :, 0] < x_min)
            | (pcd[:, :, 0] > x_max)
            | (pcd[:, :, 1] < y_min)
            | (pcd[:, :, 1] > y_max)
            | (pcd[:, :, 2] < z_min)
            | (pcd[:, :, 2] > z_max)
            | np.isnan(pcd[:, :, 0])
            | np.isnan(pcd[:, :, 1])
            | np.isnan(pcd[:, :, 2])
        )

        # TODO: move from a list to a better batched version
        pcd = [pcd[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        pcd = np.concatenate(pcd)
        rgb = [rgb[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        rgb = np.concatenate(rgb)
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # 显示原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # -- 处理end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3])*0.001
    angles_deg_end = np.array(end_pose[3:6])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T_end)

    # -- 处理pred_pose
    if isinstance(pred_pose, str):
        pred_pose = [float(x) for x in pred_pose.strip('{}').split(',')]
    else:
        pred_pose = [float(x) for x in pred_pose]
    pos_pred = np.array(pred_pose[:3])*0.001
    angles_deg_pred = np.array(pred_pose[3:])
    angles_rad_pred = np.deg2rad(angles_deg_pred)
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
    T_pred = np.eye(4)
    T_pred[:3, :3] = rot_mat_pred
    T_pred[:3, 3] = pos_pred
    axis_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_pred.transform(T_pred)

    if gt_pose is not None:
        if isinstance(gt_pose, str):
            gt_pose = [float(x) for x in gt_pose.strip('{}').split(',')]
        else:
            gt_pose = [float(x) for x in gt_pose]
        pos_pred = np.array(gt_pose[:3])*0.001
        angles_deg_pred = np.array(gt_pose[3:6])
        angles_rad_pred = np.deg2rad(angles_deg_pred)
        rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
        T_pred = np.eye(4)
        T_pred[:3, :3] = rot_mat_pred
        T_pred[:3, 3] = pos_pred
        axis_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        axis_gt.transform(T_pred)

    # 显示所有内容
    if gt_pose is None:
        o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_pred, axis_end])
    else:
        o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_pred, axis_gt ,axis_end])


def get_pose(pose_dir, num):
    pose_path = os.path.join(pose_dir, f"{num}.pkl")
    with open(pose_path, "rb") as f:
        pose_data = pickle.load(f)
    pose_quat = pose_data[3:7]
    gripper_state = pose_data[-1]
    pose_eurl = transforms3d.euler.quat2euler(pose_quat, axes='sxyz')
    pose_eurl = np.rad2deg(np.asarray(pose_eurl))
    pose = [float(pose_data[0]*1000.), float(pose_data[1]*1000.), float(pose_data[2]*1000.), float(pose_eurl[0]), float(pose_eurl[1]), float(pose_eurl[2])]
    return pose,gripper_state

def convert_endpos2gripperpos(gripper_pose, gripper_state):
    '''
    param:
    @input[x, y, z, r, p, y]
    @output[x, y, z, w, xi, yj, zk, gripper_state]
    '''
    gripper_pose = [float(i) for i in gripper_pose]
    gripper_pose_xyz = np.array(gripper_pose[:3]) / 1000  # mm -> m
    gripper_pose_euler = gripper_pose[3:]
    gripper_pose_quat = R.from_euler("xyz", gripper_pose_euler, degrees=True).as_quat()
    gripper_pose_full = np.concatenate(
        (gripper_pose_xyz, gripper_pose_quat, gripper_state), axis=0
    ).astype(np.float32)

    return gripper_pose_full



def _eval(args: Args):
    base_path = "/home/zk/Projects/3d_vla/checkpoints/wbl_pipeline"
    model_path = os.path.join(base_path, "model_250_0714.pth")
    exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")
    mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
    episode_length = 30
    gripper_thres = 0.07
    cameras_view = ["3rd"]

    device = f"cuda:0"
    observation = {}
    data_path = "/home/zk/Projects/Datasets/data_0722/put_the_RedBull_in_the_bottom_shelf_0722/1"  #
    pcd_dir = os.path.join(data_path, "3rd_cam_pcd")
    rgb_dir = os.path.join(data_path, "3rd_cam_rgb")
    pose_dir = os.path.join(data_path, "actions")
    extrinsic_matrix = os.path.join(data_path, "extrinsic_matrix.npy")
    instruction_path = os.path.join(data_path, "instruction.pkl")

    if not os.path.exists(extrinsic_matrix):
        extrinsic_matrix = get_cam_extrinsic()
    else:
        with open(extrinsic_matrix, 'rb') as f:
            extrinsic_matrix = np.load(f)
    print(extrinsic_matrix)

    if not os.path.exists(instruction_path):
        print("instruction file not exists")
        # instructions = [[["put the bottle in the microwave"]]]        #  放瓶子进微波炉
        instructions = [[["put the soda water in the bottom shelf"]]]          #
        # instructions = [[["put the yellow block in the blue plate"]]]             #  放方块进盘子
        # instructions = [[["press the bottle"]]]                       #  按压洗发水
        # instructions = [[["open the door"]]]                       #  开门

        # instructions = [[["put the lemonade in the microwave"]]]      #  金橘柠檬
        # instructions = [[["put the sprite in the microwave"]]]        #  雪碧
        # instructions = [[["put the green tea in the microwave"]]]     #  绿茶
        # instructions = [[["put the red tea in the microwave"]]]       #  冰红茶
        # instructions = [[["put the orange soda in the microwave"]]]     #  芬达

        # instructions = [[["put wolf in the drawer"]]]                 #  狼
        # instructions = [[["put giraffe in the drawer"]]]              #  长颈鹿
        # instructions = [[["put the tiger in the upper drawer"]]]                #  斑马

    else:
        with open(instruction_path, 'rb') as f:
            instructions = [[[pickle.load(f)]]]

    observation["language_goal"] = instructions[0][0][0]
    agent = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    action_plan = collections.deque()

    for i in range(4):
        pcd_path = os.path.join(pcd_dir, f"{i}.pkl")
        rgb_path = os.path.join(rgb_dir, f"{i}.pkl")
        pose_current, gripper_current = get_pose(pose_dir, i)
        # if i < 4:
        pose_next, gripper_next = get_pose(pose_dir, i + 1)
        # else:
        #     pose_next, gripper_next = pose_current, gripper_current
        with open(pcd_path, 'rb') as f:
            pcd_data = pickle.load(f)
        with open(rgb_path, 'rb') as f:
            rgb_data = pickle.load(f)

        pcd_data = pcd_data[:, :, :3]
        observation["3rd"] = {}
        observation["3rd"]["pcd"] = pcd_data
        rgb_data = rgb_data[:, :, :3]
        observation["3rd"]["rgb"] = rgb_data
        # pose_current = get_pose(pose_data, i + 1)
        # pose_next = get_pose(pose_data, i + 2)

        # 使用正则表达式匹配大括号中的内容
        claw_status = 1 if i == 2 or i == 3 else 0
        current_gripper = claw_status

        current_time = (1. - (i / float(episode_length - 1))) * 2. - 1.
        observation['low_dim_state'] = np.concatenate(
            [[current_gripper], [current_time]]).astype(np.float32)
        observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"], extrinsic_matrix)
        # observation["3rd"]["pcd"] = observation["3rd"]["pcd"]
        # observation["3rd"]["rgb"] = cv2.cvtColor(observation["3rd"]["rgb"], cv2.COLOR_RGB2BGR)
        end_pose = pose_current
        gripper_pos = convert_endpos2gripperpos(end_pose, np.asarray([bool(current_gripper)]))
        # vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])
        observation_origen = copy.deepcopy(observation)
        # for key, v in observation.items():
        #     if isinstance(v, dict):
        #         for sub_k, sub_v in v.items():
        #             if sub_k in ["rgb", "pcd"]:
        #                 v[sub_k] = np.transpose(v[sub_k], [2, 0, 1])
        #                 v[sub_k] = torch.from_numpy(v[sub_k]).to(device).unsqueeze(0).float().contiguous()
        #     elif isinstance(v, np.ndarray):
        #         observation[key] = torch.from_numpy(v).to(device).unsqueeze(0).contiguous()

        img = cv2.resize(observation["3rd"]["rgb"],  (256, 256), interpolation=cv2.INTER_LINEAR)
        element = {
                    "observation/image": img,  # shape[256,256,3]
                    "observation/state": gripper_pos,
                    "prompt": observation["language_goal"],
                }
        action_chunk = agent.infer(element)["actions"]
        assert (
                len(action_chunk) >= args.replan_steps
            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
        action_plan.extend(action_chunk[: args.replan_steps])
        action = action_plan.popleft()
        target_pos, target_quat_, target_gripper = action[:3], action[3:-1], action[-1:],
        if target_gripper >= 0.9:
            target_gripper = 1
        elif target_gripper <= 0.2:
            target_gripper = 0
        else:
            target_gripper = current_gripper
        # target_quat = [target_quat_[3], target_quat_[0], target_quat_[1], target_quat_[2]]
        target_quat = target_quat_
        if target_quat[0] < 0:
            # print("quat changed!")
            # print("before quat:",target_quat)
            target_quat = np.array(target_quat)
            target_quat = target_quat * (-1)
            # print("after quat:",target_quat)

        # x_range = (-0.222, 0.316)  # dobot
        # y_range = (-0.617, -0.20)
        # z_range = (0.06, 0.2)
        # target_pos[0] = np.clip(target_pos[0], x_range[0], x_range[1]) * 1000
        # target_pos[1] = np.clip(target_pos[1], y_range[0], y_range[1]) * 1000
        # target_pos[2] = np.clip(target_pos[2], z_range[0], z_range[1]) * 1000

        target_point = Point(target_pos*1000., target_quat, target_gripper)
        print("target_gripper", target_gripper)
        print("step:", i)
        print("groundtruth:", pose_next[:3])
        print("pred : ", target_pos*1000.)
        # print("Predicted target pos: ", target_pos, "Predicted target eurl: ", target_point.position_quaternion_claw,
        #       "Predicted target gripper: ", target_gripper)

        print("****************************")

        if target_gripper == 0:
            target_gripper = 1
        elif target_gripper == 1:
            target_gripper = 0
        else:
            assert False  # 训练时0开1关，测试时，1开0关

        vis_pcd_with_end_pred(observation_origen["3rd"]["pcd"], observation_origen["3rd"]["rgb"], end_pose,
                              target_point.position_quaternion_claw, gt_pose=pose_next,bounds=SCENE_BOUNDS)


if __name__ == '__main__':
    _eval(Args)
