import os
import time


import torch
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import gc
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data import DataLoader


class Real_Dataset(torch.utils.data.Dataset):
    def __init__(   
                self,
                config,
                # data_path,
                # device,
                # cameras,
                # ep_per_task=10,
                # output_arm_flag=False
            ):
        self.config = config
        self.device = config.device
        self.data_path = config.data_path ## folder will .pkl data files one for each example
        self.train_data = []
        self.cameras= config.cameras
        self.output_arm_flag = config.output_arm_flag
        print(f"You use {config.ep_per_task} episodes per task!")
        if self.output_arm_flag:
            print("Output arm_flag is enabled!")
        time.sleep(5)
        self.construct_dataset(config.ep_per_task)

    def convert_pcd_to_base(
            self, 
            extrinsic_path,
            type="3rd",
            pcd=[]
        ):
        with open(extrinsic_path, "rb") as f:
            data = pickle.load(f)
            transform = np.array(data)
        # zed相机是RGBA，所以pcd的形状为(1080, 1920, 4)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3) #去掉A
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 

    def construct_dataset(self,ep_per_task=10):
        self.num_tasks=len([  path_name  for path_name in  os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path,path_name))])
        self.num_task_paths=0
        for task in os.listdir(self.data_path):
            task_path = os.path.join(self.data_path, task)
            if os.path.isdir(task_path): 
                for episode_num in tqdm(os.listdir(task_path)):
                    print('episode_num',episode_num)
                    if int(episode_num) >=ep_per_task:
                        print(f"episode num {episode_num} is larger than {ep_per_task}")
                        continue
                    # if int(episode_num) %2==0:
                    #     print(f"episode num {episode_num} is even, skip it")
                    #     continue
                    self.num_task_paths+=1
                    episode_path = os.path.join(task_path, episode_num)
                    
                    action_path = os.path.join(episode_path, 'pose.pkl')
                    rgb_3rd = os.path.join(episode_path, "zed_rgb")
                    pcd_3rd = os.path.join(episode_path, "zed_pcd")
                    gripper_pose = read_action_file(action_path)

                    num_steps = sum(1 for file_name in os.listdir(rgb_3rd) if file_name.endswith('.pkl')) 
                    # num_steps=5 # hardcode
                    for step in range(num_steps-1):
                        sample = {}
                        # Next pose action
                       
                        # sample["gripper_pose"] = np.concatenate((gripper_pose[step+1]["position"], gripper_pose[step+1]["orientation"]), axis=0)
                        # print("before:",sample["gripper_pose"][3:7])
                        # sample["gripper_pose"][3:7] = sample["gripper_pose"][[4, 5, 6, 3]] # x y z w 作为最终的输入
                        
                        gripper_pose_xyz=np.array(gripper_pose[step+1]["position"])/1000 # mm -> m
                        gripper_pose_euler=gripper_pose[step+1]["orientation"]
                        gripper_pose_quat=R.from_euler('xyz', gripper_pose_euler, degrees=True).as_quat() # check it
                        sample["gripper_pose"] = np.concatenate((gripper_pose_xyz, gripper_pose_quat,[gripper_pose[step+1]["claw_status"]]), axis=0)

                        current_gripper_state = gripper_pose[step]["claw_status"]

                        time = (1. - (step / float(num_steps - 1))) * 2. - 1.
                        sample['low_dim_state'] = np.concatenate(
                            [[current_gripper_state], [time]]).astype(np.float32)
                            
                        sample["ignore_collisions"] = 1.0
                        
                        sample["3rd"], sample["wrist"] = {}, {}
                        if "3rd" in self.cameras:
                            with open(os.path.join(rgb_3rd, f"{step}.pkl"), 'rb') as f:
                                sample["3rd"]["rgb"] = pickle.load(f)[:, :, :3]
                                sample["3rd"]["rgb"] = np.ascontiguousarray(sample["3rd"]["rgb"])  #   check it  the final image should be RGB
                                sample["3rd"]["rgb"] = np.transpose(sample["3rd"]["rgb"], [2, 0, 1])  # 转为（C,H,W）
                            with open(os.path.join(pcd_3rd, f"{step}.pkl"), 'rb') as f:
                                sample["3rd"]["pcd"] = pickle.load(f)[:, :, :3]  
                                sample["3rd"]["pcd"] = self.convert_pcd_to_base(pcd=sample["3rd"]["pcd"], type="3rd",extrinsic_path=os.path.join(episode_path, "extrinsic_matrix.pkl"))
                                sample["3rd"]["pcd"] = np.transpose(sample["3rd"]["pcd"], [2, 0, 1]).astype(np.float32)
                            
                        if "wrist"  in self.cameras:
                            assert False
                            with open(os.path.join(rgb_wrist, f"{step}.pkl"), 'rb') as f:
                                sample["wrist"]["rgb"] = pickle.load(f)
                                sample["wrist"]["rgb"] = np.ascontiguousarray(sample["wrist"]["rgb"][:, :, ::-1])  # BGR转RGB（H,W,C）
                                sample["wrist"]["rgb"] = np.transpose(sample["wrist"]["rgb"], [2, 0, 1])
                            with open(os.path.join(pcd_wrist, f"{step}.pkl"), 'rb') as f:
                                sample["wrist"]["pcd"] = pickle.load(f)
                                sample["wrist"]["pcd"] = self.convert_pcd_to_base(pcd=sample["wrist"]["pcd"], type="wrist")
                                sample["wrist"]["pcd"] = np.transpose(sample["wrist"]["pcd"], [2, 0, 1])
                        
                        # import open3d as o3d
                        # def vis_pcd(pcd, rgb):

                        #     # 将点云和颜色转换为二维的形状 (N, 3)
                        #     pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
                        #     rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

                        #     # 将点云和颜色信息保存为 PLY 文件
                        #     pcd = o3d.geometry.PointCloud()
                        #     pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
                        #     pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
                        #     # o3d.io.write_point_cloud(save_path, pcd)
                        #     o3d.visualization.draw_geometries([pcd])
                        # test_pcd = np.concatenate((sample["3rd"]["pcd"], sample["wrist"]["pcd"]), axis=0)
                        # test_rgb = np.concatenate((sample["3rd"]["rgb"], sample["wrist"]["rgb"]), axis=0)
                        # vis_pcd(test_pcd, test_rgb)
                                
                                                
                        with open(os.path.join(episode_path, f"instruction.pkl"), 'rb') as f:
                            instruction = pickle.load(f)


                        sample["lang_goal"] = instruction.strip()
                        
                        sample["tasks"] = task
                        
                        # 如果启用output_arm_flag，则添加arm_flag到样本中
                        if self.output_arm_flag:
                            sample["arm_flag"] = gripper_pose[step+1]["arm_flag"]
                        
                        self.train_data.append(sample)           
        gc.collect()
        torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]
    
    

def read_action_file(action_path):
    '''
    文件内容类似如下
    Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz) Claw Status
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236
    2025-04-27_15-35-39-699 -196.8765 -642.8876 119.9657 89.8462 -8.2865 -87.1616
    2025-04-27_15-35-21-250 -53.4814 -643.2122 133.6393 90.0660 -9.3319 -89.6236
    2025-04-27_15-34-58-985 126.8121 -441.9641 60.3471 91.3858 -4.3865 -48.2481
    2025-04-27_15-34-33-519 166.5982 -165.0889 168.8611 88.0599 -2.6958 -90.3270
    '''
    with open(action_path, "rb") as f:
        data_str = pickle.load(f)
    
    # Split the string into lines
    lines = data_str.strip().split('\n')
    
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
        
        # Determine claw status based on position in sequence
        # claw_status = 1 if i % 4 == 1 or i % 4 == 0 else 0 # 1表示开，0表示关
        if len(parts) == 9:
            claw_status = int(parts[7])
            arm_flag = int(parts[8])
        elif len(parts) == 8:
            claw_status = int(parts[7])
            arm_flag = 0
        else:
            if i == 1 or i == 2 or i ==5:
                claw_status = 0
            else:
                claw_status = 1
            arm_flag = 0
        
        # Create dictionary for this entry
        entry = {
            'timestamp': timestamp,
            'position': position,
            'orientation': orientation,
            'claw_status': claw_status,
            'arm_flag': arm_flag,
        }
        
        result.append(entry)
    
    return result


def create_bridge_dataloader(dataset, rank, world_size, batch_size, num_workers, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True  # 是否打乱数据
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True, # 丢弃最后一个不完整批次
            pin_memory=True  # 加速数据加载
        )
        return dataloader, sampler
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        return dataloader, None