# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
import os
# import yaml
import sys
# from scipy.spatial.transform import Rotation as R
import copy
from multiprocessing import Value
from copy import deepcopy
import collections

from transforms3d.euler import euler2mat
import pickle
import torch
import cv2
import time
import dataclasses
import numpy as np
import open3d as o3d


from src.openpi.bridgevla.mvt import mvt
from openpi_client import websocket_client_policy as _websocket_client_policy
from botarm import *
# from src.openpi.bridgevla.mvt import config as default_mvt_cfg
# # import rvt_our.mvt.config as default_mvt_cfg
# # import rvt_our.models.rvt_agent as rvt_agent
# import config as default_exp_cfg

sys.path.append("/home/zk/Projects/RVT_fiveage")
# sys.path.append('/home/zk/Projects/3d_vla/RVT/rvt/libs/point-renderer')
from utils.real_camera_utils_new import Camera, get_cam_extrinsic

# from tensorflow.python.summary.summary_iterator import summary_iterator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

CAMERAS = ["3rd"]
SCENE_BOUNDS = [
    -1.3,
    -1.5,
    -0.1,
    0.4,
    0.7,
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
        pcd=[]
):
    transform = get_cam_extrinsic(type,'left')

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)

    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


def vis_pcd_with_end_pred(pcd, rgb, end_pose, pred_pose):
    # SCENE_BOUNDS: [x_min, y_min, z_min, x_max, y_max, z_max]
    SCENE_BOUNDS = [
        -1.3,
        -1.5,
        -0.1,
        0.4,
        0.7,
        0.6,
    ]

    # 转换点云和颜色形状
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # --- 截取点云 ---
    x_min, y_min, z_min, x_max, y_max, z_max = SCENE_BOUNDS
    mask = (
            (pcd_flat[:, 0] >= x_min) & (pcd_flat[:, 0] <= x_max) &
            (pcd_flat[:, 1] >= y_min) & (pcd_flat[:, 1] <= y_max) &
            (pcd_flat[:, 2] >= z_min) & (pcd_flat[:, 2] <= z_max)
    )
    pcd_flat = pcd_flat[mask]
    rgb_flat = rgb_flat[mask]
    # -----------------

    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # 显示原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # -- 处理end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3]) * 0.001
    angles_deg_end = np.array(end_pose[3:])
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
    pos_pred = np.array(pred_pose[:3]) * 0.001
    angles_deg_pred = np.array(pred_pose[3:])
    angles_rad_pred = np.deg2rad(angles_deg_pred)
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
    T_pred = np.eye(4)
    T_pred[:3, :3] = rot_mat_pred
    T_pred[:3, 3] = pos_pred
    axis_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_pred.transform(T_pred)

    # 显示所有内容
    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end, axis_pred])

def euler_xyz_to_quaternion(self, roll, pitch, yaw):
    """Convert Euler angles (XYZ order) to quaternion."""
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    # Compute quaternion
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz

    return (w, x, y, z)

def save_to_local(observation, end_pose, current_gripper, step, local_path):
    end_pose_f = []
    for p in end_pose:
        p = float(p)
        end_pose_f.append(p)
    end_pose = np.asarray(end_pose_f)
    pcd_folder_name = "3rd_cam_pcd"
    rgb_folder_name = "3rd_cam_rgb"
    pose_folder_name = "actions"
    pcd_path = os.path.join(local_path, pcd_folder_name, f"{step}.pkl")
    rgb_path = os.path.join(local_path, rgb_folder_name, f"{step}.pkl")
    pose_path = os.path.join(local_path, pose_folder_name, f"{step}.pkl")
    os.makedirs(os.path.dirname(pcd_path),exist_ok=True)
    os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)



    pcd = observation["3rd"]["pcd"]
    rgb = observation["3rd"]["rgb"]
    pose_T = end_pose[0:3]
    pose_T *= 0.001
    pose_R = end_pose[3:7]
    pose_R = np.deg2rad(pose_R)
    pose_R = transforms3d.euler.euler2quat(*pose_R, axes='sxyz')
    current_gripper = np.array([float(current_gripper)])
    pose = np.concatenate((pose_T, pose_R, current_gripper))

    with open(pcd_path, "wb") as f:
        pickle.dump(pcd, f)

    with open(rgb_path, "wb") as f:
        pickle.dump(rgb, f)

    with open(pose_path, "wb") as f:
        pickle.dump(pose,f)

def convert_endpos2gripperpos(gripper_pose):
    '''
    param:
    @input[x, y, z, r, p, y]
    @output[x, y, z, w, xi, yj, zk, gripper_state]
    '''
    gripper_pose_xyz = np.array(gripper_pose["position"]) / 1000  # mm -> m
    gripper_pose_euler = gripper_pose["orientation"]
    gripper_pose_quat = R.from_euler("xyz", gripper_pose_euler, degrees=True).as_quat()
    gripper_pose_full = np.concatenate(
        (gripper_pose_xyz, gripper_pose_quat, [gripper_pose["claw_status"]]), axis=0
    ).astype(np.float32)

    return gripper_pose_full


def _eval(args: Args):
    base_path = "/home/zk/Projects/3d_vla/checkpoints/wbl_pipeline"
    # base_path = "/data2/lpy/3D_VLA/code/checkpoints/8e5_3rdq_new_new"q
    model_path = os.path.join(base_path, "model_400_0724_DPo.pth")#$
    # model_path = os.path.join(base_path, "rlbench_pretrain_model_10_300_debug_color.pth")
    # model_path = os.path.join(base_path, "with_pretrain_debug_300.pth")
    # model_path = os.path.join(base_path, "rlbench_allcameras_no_proprio_70_no_finetune_real_smallsize.pth")
    exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")
    mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
    episode_length = 25
    gripper_thres = 0.07

    output_arm_flag = False

    # cameras_view=["3rd", "wrist"]
    cameras_view = ["3rd"]



    device = f"cuda:0"
    observation = {}

    # instructions = [[["put the bottle in the microwave"]]]        #  放瓶子进微波炉
    instructions = [[["put the tiger in the upper drawer"]]]          #  放方块在架子上
    # instructions = [[["put the yellow cube in the blue plate"]]]             #  放方块进盘子
    # instructions = [[["put the sanitizer in the top shelf"]]]                       #  放洗发水


    # instructions = [[["put the Redbull in the top shelf"]]]       #
#@
    # instructions = [[["put the red bottle in the top shelf"]]]
    # instructions = [[["put the Redbull in the bottom shelf"]]]  #

    # instructions = [[["put the soda water in the bottom shelf"]]]  #
    # instructions = [[["put the soda water in the top shelf"]]]
    # instructions = [[["put the coke can in the top shelf"]]]

    # instructions = [[["put the yellow bottle in the top shelf"]]]


    # instructions = [[["put the dack blueb block in the top shelf"]]]

    # instructions = [[["press sanitizer_red"]]]

    # instructions = [[["press sanitizer_white"]]]

    # instructions = [[["put the tiger in the upper drawer"]]]q


    # instructions = [[["put the tiger in the lower drawer"]]]


    # instructions = [[["put the wolf in the upper drawer"]]]


    # instructions = [[["put the wolf in the bottom drawer"]]]

    # instructions = [[["put the leopard in the upper drawer"]]]

    # instructions = [[["put the leopard in the lower drawer"]]]
    # instructions = [[["put the orange cube in the green plate"]]]

    # instructions = [[["put the red cube in the purple plate"]]]

    # instructions = [[["put the red cube in the blue plate"]]]

    # instructions = [[["put the leopard in the upper drawer"]]]

    # instructions = [[["put the tiger in the upper drawer"]]]
    # instructions = [[["put the lemonade in the microwave"]]]      #  金橘柠檬
    # instructions = [[["put the coke in the bottom shelf"]]]        #  雪碧
    # instructions = [[["put the green tea in the microwave"]]]     #  绿茶
    # instructions = [[["put the red tea in the microwave"]]]       #  冰红茶
    # instructions = [[["put the orange soda in the microwave"]]]     #  芬达

    # instructions = [[["put wolf in the drawer"]]]                 #  狼
    # instructions = [[["put giraffe in the drawer"]]]              #  长颈鹿
    # instructions = [[["put zebra in the drawer"]]]                #  斑马



    cameras = Camera(camera_type="3rd")  # modify it
    # time.sleep(1)
    # fa = FrankaArm()
    # fa.reset_joints()
    # fa.open_gripper()
    # time.sleep(2)
    agent = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


    observation["language_goal"] = instructions

    # 初始化机械臂
    if output_arm_flag:
        # 初始化左臂
        server_l = Server('192.168.201.1', 29999, '192.168.110.143', 12345)
        server_l.sock.connect((server_l.ip, server_l.port))
        bot_l = DobotController(server_l.sock)
        server_l.init_com(bot_l)
        bot_l._initialize(server_l)

        # 初始化右臂
        server_r = Server('192.168.5.2', 29999, '192.168.110.143', 12346)
        server_r.sock.connect((server_r.ip, server_r.port))
        bot_r = DobotController(server_r.sock)
        server_r.init_com(bot_r)
        bot_r._initialize(server_r)
    else:
        # 初始化单臂
        server = Server('192.168.201.1', 29999, '192.168.110.143', 12345)
        # 连接到 DoBot 机械臂的 Dashboard 端口 (29999)
        server.sock.connect((server.ip, server.port))
        bot = DobotController(server.sock)
        server.init_com(bot)

        # 初始化机器人
        bot._initialize(server)
        
    action_plan = collections.deque()
    try:
        for step in range(episode_length - 1):
            print("step", step)
            camera_info = cameras.capture()
            observation["3rd"] = camera_info["3rd"]
            # observation["wrist"] = camera_info["wrist"]
            observation["3rd"]["rgb"] = observation["3rd"]["rgb"].copy()

            if output_arm_flag:
                # 读取左右臂夹爪状态
                current_gripper_state_l = bot_l.claws_read_command(server_l.modbusRTU, 258, 1)
                current_gripper_state_r = bot_r.claws_read_command(server_r.modbusRTU, 258, 1)
                import re
                match_l = re.search(r'\{(.*?)\}', str(current_gripper_state_l))
                match_r = re.search(r'\{(.*?)\}', str(current_gripper_state_r))
                current_gripper_l = match_l.group(1)
                current_gripper_r = match_r.group(1)
                print("左臂夹爪状态（1 关 0 开）", current_gripper_l)
                print("右臂夹爪状态（1 关 0 开）", current_gripper_r)
                current_time = (1. - (step / float(episode_length - 1))) * 2. - 1.
                observation['low_dim_state'] = np.concatenate(
                    [[current_gripper_l], [current_gripper_r], [current_time]]).astype(np.float32)
                
                end_pose_l = bot_l.get_pose()
                end_pose_r = bot_r.get_pose()
            else:
                current_gripper_state = bot.claws_read_command(server.modbusRTU, 258, 1)  # 读取夹爪状态 1 关   0  开
                import re
                # 使用正则表达式匹配大括号中的内容
                match = re.search(r'\{(.*?)\}', str(current_gripper_state))
                current_gripper = match.group(1)  # 输出当前夹爪状态
                print("当前夹爪状态（1 关 0 开）", current_gripper)
                current_time = (1. - (step / float(episode_length - 1))) * 2. - 1.
                observation['low_dim_state'] = np.concatenate(
                    [[current_gripper], [current_time]]).astype(np.float32)

                end_pose = bot.get_pose()
                gripper_pos = convert_endpos2gripperpos(end_pose)
            # 保存rgb，pcd， pose
            local_path = "/home/zk/Projects/local_data_0604_25/"
            # save_to_local(observation, end_pose, current_gripper, step, local_path)

            observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"])
            observation["3rd"]["rgb"] = cv2.cvtColor(observation["3rd"]["rgb"], cv2.COLOR_RGB2BGR)
            # vis_pcd(test_pcd, test_rgb)
            # vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])
            observation_origen = copy.deepcopy(observation)

            for key, v in observation.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if sub_k in ["rgb", "pcd"]:
                            v[sub_k] = np.transpose(v[sub_k], [2, 0, 1])
                            v[sub_k] = torch.from_numpy(v[sub_k]).to(device).unsqueeze(0).float().contiguous()

                elif isinstance(v, np.ndarray):
                    observation[key] = torch.from_numpy(v).to(device).unsqueeze(0).contiguous()
            
            element = {
                        "observation/image": observation["3rd"]["rgb"],  # shape[224,224,3]
                        "observation/state": gripper_pos,
                        "prompt": observation["language_goal"],
                    }
            time1 = time.time()
            if output_arm_flag:
                target_pos, target_quat_, target_gripper, target_arm_flag = agent.act_real(observation, cameras_view)
            else:
                action_chunk = agent.infer(element)["actions"]
                assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])
            action = action_plan.popleft()
            target_pos, target_quat_, target_gripper = action[:3], action[3:-1], action[-1:],
            time2 = time.time()
            print("推理一次需要时间:", time2 - time1)
            # target_quat=[target_quat_[3],target_quat_[0],target_quat_[1],target_quat_[2]]
            target_quat = target_quat_
            if target_quat[0] < 0:
                # print("quat changed!")
                # print("before quat:",target_quat)
                target_quat = np.array(target_quat)
                target_quat = target_quat * (-1)
                # print("after quat:",target_quat)

            # 安全工作范围
            # SCENE_BOUNDS x: (-0.6,0.4)  y:(-0.9,0.1)  z:(-0.1,0.6)

            # x_range = (-0.4, 0.2)  # dobot 放瓶子进微波炉任务安全距离
            # y_range = (-0.7, -0.3)
            # z_range = (0.05, 0.15)

            # x_range = (-0.3, 0.3)  # dobot 放方块进盘子任务安全距离
            # y_range = (-0.9, -0.2)
            # z_range = (0.02, 0.4)

            # x_range = (-0.6, 0.25)  # dobot 放方块上架子任务安全距离
            # y_range = (-0.9, -0.15)
            # z_range = (-0.05, 0.7)

            # x_range = (-0.3, 0.35)  # dobot 开门任务安全距离
            # y_range = (-0.95, -0.2)
            # z_range = (0.05, 0.3)

            #
            # x_range = (-0.2, 0.2)  # dobot 按压洗手液任务安全距离
            # y_range = (-0.9, -0.2)
            # z_range = (0, 0.7)
            # x_range = (-0., 0.2)  # dobot 按压洗手液任务安全距离
            y_range = (-0.6, 0.3)
            z_range = (-0.1, 0.5)


            # x_range = (-0.5, 0.25)  # dobot 放动物进抽屉任务安全距离
            # y_range = (-0.95, -0.25)
            # z_range = (0.05, 0.7)q

            # target_pos[0] = np.clip(target_pos[0], x_range[0], x_range[1])
            target_pos[1] = np.clip(target_pos[1], y_range[0], y_range[1])
            target_pos[2] = np.clip(target_pos[2], z_range[0], z_range[1])

            if 'coke' in instructions[0][0][0]:
                if step>=3:
                    target_pos[2] += 0.02

            target_pos[0] = target_pos[0] * 1000
            target_pos[1] = target_pos[1] * 1000
            target_pos[2] = target_pos[2] * 1000
            #
            # # 只适用于微波炉
            # if instructions in ["put the bottle in the microwave", "put the lemonade in the microwave",
            #                     "put the sprite in the microwave", "put the green tea in the microwave",
            #                     "put the red tea in the microwave", "put the orange soda in the microwave"]:
            #     if target_pos[2] > 150:
            #         target_pos[2] = 150  # 控制高度仅适用于微波炉任务

            print("Predicted target pos: ", target_pos, "Predicted target quat: ", target_quat,
                  "Predicted target gripper: ", target_gripper)
            if target_gripper == 0:
                target_gripper = 1
            elif target_gripper == 1:
                target_gripper = 0
            else:
                assert False  # 训练时0开1关，测试时，1开0关

            target_point = Point(target_pos, target_quat, target_gripper)

            # vis_pcd_with_end_pred(observation_origen["3rd"]["pcd"], observation_origen["3rd"]["rgb"], end_pose,
            #                       target_point.position_quaternion_claw)  # 可视化

            # if not bot.wait_and_control():
            #     continue

            print('target_point', type(target_point.euler))
            # if step>=1:
            #     if target_point.euler[0] - current_point.euler[0] >= 180:
            #         target_point.euler[0] -= 180
            #     if target_point.euler[1] - current_point.euler[1] >= 180:
            #         target_point.euler[1] -= 180
            #     if target_point.euler[2] - current_point.euler[2] >= 180:
            #         target_point.euler[2] -= 180



            # 添加极小的随机噪声
            current_point = copy.deepcopy(target_point)
            if output_arm_flag:
                if target_arm_flag == 0:
                    pos_response = bot_l.point_control(target_point)  # 到达位置
                else:
                    pos_response = bot_r.point_control(target_point)  # 到达位置
            else:
                pos_response = bot.point_control(target_point)  # 到达位置

            print('pos_response', pos_response)
            if pos_response != 'Success':
                continue
            time.sleep(3)
            if output_arm_flag:
                if target_arm_flag == 0:
                    bot_l.claws_control(target_gripper, server_l.modbusRTU)  # 开闭合夹爪
                else:
                    bot_r.claws_control(target_gripper, server_r.modbusRTU)  # 开闭合夹爪
            else:
                bot.claws_control(target_gripper, server.modbusRTU)  # 开闭合夹爪

            # time.sleep(2)

            if output_arm_flag:
                if target_arm_flag == 0:
                    bot_l.wait_and_prompt()  # 如不需要等待，而是直接运行全过程，可注释掉这一句.
                else:
                    bot_r.wait_and_prompt()  # 如不需要等待，而是直接运行全过程，可注释掉这一句
            else:
                bot.wait_and_prompt()  # 如不需要等待，而是直接运行全过程，可注释掉这一句

    finally:
        print("close servers")
        if output_arm_flag:
            bot_l.interrupt_close()
            bot_r.interrupt_close()
            server_l.sock.close()
            server_r.sock.close()
            server_l.app.close()
            server_r.app.close()
        else:
            bot.interrupt_close()
            server.sock.close()
            server.app.close()
        # listener.stop()


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    _eval()