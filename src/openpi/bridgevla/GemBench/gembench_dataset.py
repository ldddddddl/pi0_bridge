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

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import torch
import os
import numpy as np
from tqdm import tqdm
import time
import random
import json
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import copy
class Gembench_Dataset(torch.utils.data.Dataset):
    def __init__(   
                self,
                data_path,
                device,
                cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
                ep_per_task=1000,
            ):
        self.device = device
        self.data_path = data_path ## folder will .pkl data files one for each example
        self.train_data = []
        self.cameras=cameras
        time.sleep(5)
        self.construct_dataset(ep_per_task)

        
    def construct_dataset(self, ep_per_task):
        instruction_path = os.path.join(self.data_path, "taskvars_instructions_new.json")
        instruction_dict = json.load(open(instruction_path, "r"))
        
        episode_path = os.path.join(self.data_path, "keysteps_bbox/seed0")
        self.num_tasks=len(os.listdir(episode_path))
        self.num_task_paths=0
        for task in tqdm(os.listdir(episode_path)):
            task_path = os.path.join(episode_path, task)
            task_all_episode = lmdb.open(
                task_path,
                readonly=True,
                meminit=False,
                lock=False      
            )
            for i, data in enumerate(task_all_episode.begin().cursor()):
                episode = data[1]
                if i >= ep_per_task:
                    break
                self.num_task_paths+=1
                episode = msgpack.unpackb(episode)
                num_steps = len(episode["key_frameids"])
                    
                for i in range(num_steps-1):
                    sample = {}
                    for cam_idx, cam in enumerate(self.cameras):
                        sample[cam] = {
                            "pcd": np.transpose(episode["pc"][i][cam_idx], (2, 0, 1)),
                            "rgb": np.transpose(episode["rgb"][i][cam_idx], (2, 0, 1))
                           
                        }
                    sample["gripper_pose"] = episode["action"][i+1]
                    time = (1. - (i / float(num_steps - 1))) * 2. - 1.
                    sample["low_dim_state"] = np.concatenate(
                        [sample["gripper_pose"], [time]]).astype(np.float32)
                    
                    sample["ignore_collisions"] = 1.0
                    sample["lang_goal"] = instruction_dict[task]
                    sample["tasks"] = task
                    self.train_data.append(copy.deepcopy(sample))
                    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample=self.train_data[idx]
        sample["lang_goal"] = random.choice(sample["lang_goal"])   # randomly choose one instruction for every fetching. This is important for generalization.
        return sample
    
if __name__ == "__main__":
    dataset = Gembench_Dataset(data_path="/PATH_TO_GEMBENCH_TRAIN_DATA/train_dataset", device="cuda:0",ep_per_task=3)
    data=dataset[0]
    print(data["lang_goal"])
    # for data in dataset:
    #     print(data.keys())
    #     print(data["lang_goal"])
    #     break
