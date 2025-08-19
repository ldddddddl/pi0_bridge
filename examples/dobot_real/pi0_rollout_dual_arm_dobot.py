import logging
import time
import queue
import threading
from pprint import pformat
from dataclasses import asdict
import dataclasses

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.camera_zed import Camera
from utils.dobotpy import Server, DobotController

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import sys

from openpi_client import websocket_client_policy as _websocket_client_policy


# ========== 全局控制变量 ==========
degree_queue_L = queue.Queue()
degree_queue_R = queue.Queue()
stop_flag = threading.Event()
thread_L = None
thread_R = None

start_pose_L = [382.8289, -155.2568, 515.627, 9.0372, 1.9933, -3.7694]
start_pose_R = [-375.1244, -467.0635, 539.5372, -6.2825, -6.8407, 110.6101]
task = "simple_pick_place"
# ===================================


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000
    replan_steps: int = 1


def get_observation(bot_L, bot_R, camera, task="task_name"):
    image_bgr, _, _ = camera.get_image()
    image_rgb = image_bgr[..., ::-1].copy()  # BGR -> RGB
    pose_L = bot_L.get_angle
    pose_R = bot_R.get_angle
#    degree_L = bot_L.changingtek_get_degree()
#    degree_R = bot_R.changingtek_get_degree()

    degree_L = 100
    degree_R = 100

    image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    observation = {
        "observation/image": image_rgb,
        "observation/wrist_image": image_rgb,
        "observation/state": np.array(list(pose_L) + [degree_L] + list(pose_R) + [degree_R], dtype=np.float32),
        "prompt": task
    }
    return observation


def gripper_thread_func(bot, server, degree_queue: queue.Queue, stop_flag: threading.Event, name=""):
    last_degree = None
    while not stop_flag.is_set():
        try:
            degree = degree_queue.get(timeout=0.1)
            if degree != last_degree:
                bot.changingtek_open_degree(degree, server.modbusRTU_id)
                last_degree = degree
        except queue.Empty:
            continue


def initialize_system():
    global thread_L, thread_R

    # 左臂初始化
    server_L = Server("192.168.1.4", 29999, "192.168.1.4", 30004, modbus=True, control=True, feedback=True)
    bot_L = DobotController(server_L)
    bot_L.control_movement(mode='pose', value=start_pose_L)
    bot_L.changingtek_open_degree(100, server_L.modbusRTU_id)

    # 右臂初始化
    server_R = Server("192.168.1.5", 29999, "192.168.1.5", 30004, modbus=True, control=True, feedback=True)
    bot_R = DobotController(server_R)
    bot_R.control_movement(mode='pose', value=start_pose_R)
    bot_R.changingtek_open_degree(100, server_R.modbusRTU_id)

    # 启动夹爪控制线程
    thread_L = threading.Thread(target=gripper_thread_func, args=(bot_L, server_L, degree_queue_L, stop_flag, "L"))
    thread_R = threading.Thread(target=gripper_thread_func, args=(bot_R, server_R, degree_queue_R, stop_flag, "R"))
    thread_L.start()
    thread_R.start()

    start_angle_L = bot_L.get_angle
    start_angle_R = bot_R.get_angle

    print("[INFO] Gripper threads started")

    camera = Camera()
    return camera, bot_L, bot_R, start_angle_L, start_angle_R


def execute_action_sequence_and_get_observation(bot_L: DobotController, bot_R: DobotController, camera, start_pose_L, start_pose_R, action_sequence, task="task_name"):
    """
    执行 delta 动作序列并获取最终 observation。
    
    每个动作是长度为14的列表：
        前7位是左臂 [Δx, Δy, Δz, Δr, Δp, Δy, gripper_degree]
        后7位是右臂 [Δx, Δy, Δz, Δr, Δp, Δy, gripper_degree]

    所有 delta 都是基于 start_pose_{L,R} 的相对增量。

    :return: dict, 包含 image, state, task
    """
    # start_pose_L = list(start_pose_L[:6])
    # start_pose_R = list(start_pose_R[:6])

    last_degree_L = bot_L.changingtek_get_degree
    last_degree_R = bot_R.changingtek_get_degree

    for action in action_sequence:


        curr_end_pos_L = list(bot_L.get_end_pose)
        curr_end_pos_R = list(bot_R.get_end_pose)

        if isinstance(action, dict):
            delta = action["pose"]
        else:
            delta = action

        # assert len(delta) == 14, f"Expected action length 14, got {len(delta)}"

        delta_L = delta[:6]
        delta_R = delta[7:13]


        degree_L = int(delta[6] * 100.)
        degree_R = int(delta[13] * 100.)
        if degree_L > 0.75 and last_degree_L < 75:
            degree_L = 100
        elif degree_L < -0.75 and last_degree_L > 75:
            degree_L = 50
        else:
            degree_L = last_degree_L
        if degree_R > 0.75 and last_degree_R < 75:
            degree_R = 100
        elif degree_R < -0.75 and last_degree_R > 75:
            degree_R = 10
        else:
            degree_R = last_degree_R
        next_pose_L = [d+c for c, d in zip(curr_end_pos_L, delta_L)]
        next_pose_R = [d+c for c, d in zip(curr_end_pos_R, delta_R)]
        print('next_pose_L',next_pose_L)
        print('next_pose_R',next_pose_R)

        bot_L.control_servoj(mode='joint', joint=next_pose_L)
        bot_R.control_servoj(mode='joint', joint=next_pose_R)
        # time.sleep(0.1)
        print('degree_L',degree_L)
        print('degree_R',degree_R)
        #if degree_L is not None and degree_L != last_degree_L:
        # 左夹爪控制逻辑
        if degree_L is not None:
            if degree_L > 90 and last_degree_L < 90:
                time.sleep(0.1)
                degree_queue_L.put(100)  # 打开
                last_degree_L = degree_L
                
            elif degree_L < 60 and last_degree_L > 60:
                time.sleep(0.1)
                degree_queue_L.put(50)   # 关闭
                last_degree_L = degree_L
                

        # 右夹爪控制逻辑
        if degree_R is not None:
            if degree_R > 90 and last_degree_R < 90:
                time.sleep(0.1)
                degree_queue_R.put(100)
                last_degree_R = degree_R
                
            elif degree_R < 20 and last_degree_R > 20:
                time.sleep(0.1) 
                degree_queue_R.put(10)
                last_degree_R = degree_R
                     
        time.sleep(0.04)

    image_bgr, _, _ = camera.get_image()
    image_rgb = image_bgr[..., ::-1].copy()  # BGR -> RGB
    final_pose_L = bot_L.get_angle[:6]
    final_pose_R = bot_R.get_angle[:6]
    #degree_L = bot_L.changingtek_get_degree
    #degree_R = bot_R.changingtek_get_degree
    image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    observation = {
        "observation/image": image_rgb,
        "observation/wrist_image": image_rgb,
        "observation/state": np.array(list(final_pose_L) + [degree_L] + list(final_pose_R) + [degree_R], dtype=np.float32),
        "prompt": task
    }

    #print(observation)
    return observation


def run_policy_loop(args):
    #logging.info(pformat(asdict(cfg)))
    
    device = get_safe_torch_device("cuda", log=True)

    agent = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    # policy.eval()

    camera, bot_L, bot_R, start_angle_L, start_angle_R = initialize_system()

    try:
        task = "put the coke into the microwave"
        observation = get_observation(bot_L, bot_R, camera, task=task)
        step = 0
        while step < 100:
            obs_tensor = observation
            obs_tensor = {k: v for k, v in obs_tensor.items()}

            # action_sequence = action[0][45:]

            # observation = execute_action_sequence_and_get_observation(
            #     bot_L, bot_R, camera, start_angle_L, start_angle_R, action_sequence, task=task
            # )

            action_chunk = agent.infer(obs_tensor)["actions"].clip(-5., 5.)
            
            print('action',action_chunk.shape)
            print('action',action_chunk)
            # action_sequence = action.cpu().numpy().flatten().reshape(1, -1)
            observation = execute_action_sequence_and_get_observation(
                bot_L, bot_R, camera, start_angle_L, start_angle_R, action_chunk, task=task
            )

            step += 1

    finally:
        print("[INFO] Cleaning up system...")
        degree_queue_L.put(100)
        #degree_queue_R.put(100)
        bot_L.control_movement(mode='pose', value=start_pose_L)
        #bot_R.control_movement(mode='pose', value=start_pose_R)
        time.sleep(1)
        stop_flag.set()
        thread_L.join()
        thread_R.join()
        print("[INFO] Gripper threads exited.")

# @parser.wrap()
def main():
    run_policy_loop(Args)

if __name__ == '__main__':
    main()

