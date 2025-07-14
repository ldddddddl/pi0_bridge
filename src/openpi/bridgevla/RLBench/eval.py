'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/eval.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import yaml
import csv
import torch
import cv2
import shutil
import numpy as np
from multiprocessing import Value
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from bridgevla.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
)
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
import os 

def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False):
    device = f"cuda:{device}"
    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()


    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    rvt = MVT(
        renderer_device=device,
        **mvt_cfg,
    )

    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )


    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent




@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
    model_name="debug",
    visualize=False,
    visualize_root_dir="",
):
    agent.eval()

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        language_goals=[]
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            if not visualize:
                generator = rollout_generator.generator(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=False,
                    replay_ground_truth=replay_ground_truth,
                )
            else:
                task_name = tasks[task_id]
                lang_goal = eval_env._lang_goal
                visualize_save_dir=os.path.join(visualize_root_dir,task_name,f"episode_{ep}")
                if not os.path.exists(visualize_save_dir):
                    os.makedirs(visualize_save_dir)
                generator = rollout_generator.generator_visualize(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=True,
                    visualize_save_dir=visualize_save_dir,
                    visualize=True,
                )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
                    
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            video_image_folder = f"./tmp/{model_name}/{task_name}"
            palette_image_folder = f"./tmp/{model_name}/palette_folder"
            palette_image_path=os.path.join(palette_image_folder,"palette.png")

            record_fps = 25
            if not visualize:
                record_folder = os.path.join(log_dir, "videos")
            else:
                record_folder = os.path.join(visualize_root_dir,task_name,"videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = language_goals.pop(0)
                    lang_goal=lang_goal.replace(" ", "_")
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            record_folder,
                            f"{lang_goal}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            record_folder, f"{lang_goal}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1
                    video_cnt += 1
                    os.makedirs(video_image_folder, exist_ok=True)
                    os.makedirs(palette_image_folder, exist_ok=True)
                    for idx in range(len(video) - 10):
                        cv2.imwrite(
                            os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                        )
                    images_path = os.path.join(video_image_folder, r"%d.png")
                    os.system(
                        "ffmpeg -i {} -vf palettegen {} -hide_banner -loglevel error".format(
                            images_path, palette_image_path
                        )
                    )
                    
                    os.system(
                        "ffmpeg -framerate {} -i {} -i {} -lavfi paletteuse {} -hide_banner -loglevel error".format(
                            record_fps, images_path, palette_image_path, video_path
                        )
                    )
                    print(f'video saved - {task_name}')
                    os.remove(palette_image_path)
                    shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    if logging:
        csv_fp.close()


    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    model_paths = []
    assert args.model_name is not None
    model_paths.append(os.path.join(args.model_folder, args.model_name))
    tb = TensorboardManager(args.eval_log_dir)
    for model_path in model_paths:
        tasks_to_eval = deepcopy(args.tasks)
        model_idx = get_model_index(model_path)
        if model_idx is None:
            model_idx = 0

  
        agent = load_agent(
            model_path=model_path,
            exp_cfg_path=args.exp_cfg_path,
            mvt_cfg_path=args.mvt_cfg_path,
            eval_log_dir=args.eval_log_dir,
            device=args.device,
            use_input_place_with_mean=args.use_input_place_with_mean,
        )

        agent_eval_log_dir = os.path.join(
            args.eval_log_dir, os.path.basename(model_path).split(".")[0]
        )


        os.makedirs(agent_eval_log_dir, exist_ok=True)
        scores = eval(
            agent=agent,
            tasks=tasks_to_eval,
            eval_datafolder=args.eval_datafolder,
            start_episode=args.start_episode,
            eval_episodes=args.eval_episodes,
            episode_length=args.episode_length,
            replay_ground_truth=args.ground_truth,
            device=args.device,
            headless=args.headless,
            visualize=args.visualize,
            logging=True,
            log_dir=agent_eval_log_dir,
            verbose=True,
            save_video=args.save_video,
            model_name=args.model_name,
            visualize_root_dir=args.visualize_root_dir,
        )
        print(f"model {model_path}, scores {scores}")
        task_scores = {}
        for i in range(len(tasks_to_eval)):
            task_scores[tasks_to_eval[i]] = scores[i]

        print("save ", task_scores)
        tb.update("eval", model_idx, task_scores)
        tb.writer.flush()

    tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)
