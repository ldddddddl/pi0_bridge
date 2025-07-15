# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# DROID 数据采集频率 -- 我们减慢执行速度以匹配此频率
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # 硬件参数
    left_camera_id: str = "<your_camera_id>"  # 例如 "24259877"
    right_camera_id: str = "<your_camera_id>"  # 例如 "24514023"
    wrist_camera_id: str = "<your_camera_id>"  # 例如 "13062452"

    # 策略参数
    external_camera: str | None = None  # 应该传递给策略的外部相机，从 ["left", "right"] 中选择

    # rollout 参数
    max_timesteps: int = 600
    # 从预测的动作块中执行多少个动作后再次请求策略服务器
    # 8 通常是一个好的默认值（等于 0.5 秒的动作执行）。
    open_loop_horizon: int = 8

    # 远程服务器参数
    remote_host: str = "0.0.0.0"  # 指向策略服务器的 IP 地址，例如 "192.168.1.100"
    remote_port: int = 8000  # 指向策略服务器的端口，openpi 服务器的默认端口为 8000


# 我们使用 Ctrl+C 可选地提前终止 rollout -- 但如果在策略服务器等待新动作块时按下 Ctrl+C，会抛出异常并导致服务器连接断开。
# 此上下文管理器临时阻止 Ctrl+C，并在服务器调用完成后延迟处理。
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """临时阻止键盘中断，在受保护代码后再处理。"""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # 确保用户指定了 external camera -- 策略只用一个外部相机
    assert args.external_camera is not None and args.external_camera in [
        "left",
        "right",
    ], f"请指定用于策略的外部相机，从 ['left', 'right'] 中选择，但得到 {args.external_camera}"

    # 初始化 Panda 环境。使用关节速度动作空间和夹爪位置动作空间非常重要。
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # 连接到策略服务器
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # rollout 参数
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # 准备保存 rollout 的视频
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # 获取当前观测
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # 保存第一帧观测到磁盘
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # 如果需要预测新动作块，则发送 websocket 请求到策略服务器
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # 在机器人笔记本上调整图像大小，以减少发送到策略服务器的数据量并提升延迟。
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # 用上下文管理器包裹服务器调用，防止 Ctrl+C 中断
                    # Ctrl+C 会在服务器调用完成后处理
                    with prevent_keyboard_interrupt():
                        # 返回动作块 [10, 8]，即 10 个关节速度动作（7）+ 夹爪位置（1）
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # 从动作块中选择当前要执行的动作
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # 二值化夹爪动作
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # 将所有动作维度裁剪到 [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # 休眠以匹配 DROID 数据采集频率
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input("本次 rollout 是否成功？（输入 y 表示 100%，n 表示 0%，或根据评估标准输入 0-100 的数值）")
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success 必须是 [0, 100] 之间的数字，但得到: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("是否继续评测？（输入 y 或 n）").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"结果已保存到 {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # 注意下面的 "left" 指的是立体相机对中的左相机。
        # 模型只在左立体相机上训练，所以只传递这些。
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # 去除 alpha 维度
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # 转为 RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # 除了图像观测，还要捕获本体感受状态
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # 保存图像到磁盘，便于机器人运行时实时查看
    # 合并一张大图，方便实时查看
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
