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
Adapted from https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/actioner.py

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.utils.rvt_utils import load_agent as load_agent_state
from bridgevla.mvt.mvt import MVT
from utils.peract_utils_gembench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)

class MyActioner(object):
    def __init__(self,base_path,model_epoch=40):
        model_path = os.path.join(base_path, f"model_{model_epoch}.pth")
        print("your model path:") 
        print(model_path) 
        exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")  
        mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")  
        self.agent = load_agent(
            model_path=model_path,
            exp_cfg_path=exp_cfg_path,
            mvt_cfg_path=mvt_cfg_path,
            device=0,
            use_input_place_with_mean=False,
        )
    
    def predict(self, taskvar, episode_id, step_id, instruction, obs_state_dict):
        '''Args:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        # obs_state_dict["lang_goal_tokens"] = clip.tokenize(instruction).to(self.agent._device)
        # import pdb;
        # pdb.set_trace()
        for idx, cam in enumerate(self.agent.cameras):
            obs_state_dict[f"{cam}_rgb"] = np.transpose(obs_state_dict["rgb"][idx], [2, 0, 1])[None]
            obs_state_dict[f"{cam}_point_cloud"] = np.transpose(obs_state_dict["pc"][idx], [2, 0, 1])[None]
        
        del obs_state_dict["rgb"]
        del obs_state_dict["pc"]
        del obs_state_dict['arm_links_info']
        del obs_state_dict['depth']
        del obs_state_dict['gripper']
        
        for k, v in obs_state_dict.items():
            if isinstance(v, np.ndarray):
                obs_state_dict[k] = torch.from_numpy(v).to(self.agent._device)
            elif isinstance(v, list):
                obs_state_dict[k] = torch.tensor(v).to(self.agent._device)
            elif isinstance(v, torch.Tensor):
                obs_state_dict[k] = v.to(self.agent._device)    
            obs_state_dict[k] = obs_state_dict[k].unsqueeze(0)
        obs_state_dict["language_goal"] =   [[[instruction]]]
        action = self.agent.act(step=step_id,observation=obs_state_dict,return_gembench_action=True)
        return action
    

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


