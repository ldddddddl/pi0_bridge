"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
"""

import os
import pprint
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import os

from ..mvt.augmentation import apply_se3_aug_con
from ..mvt.augmentation import aug_utils
from ..mvt import utils as mvt_utils
from ..utils import rvt_utils
from ..GemBench.utils import peract_utils_gembench as gembench_utils
from PIL import Image
from PIL import ImageDraw
from ..Real.utils import peract_utils as real_utils
from ..RLBench.utils import peract_utils_rlbench as rlbench_utils

from ..libs.YARR.yarr.agents.agent import ActResult


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3
    assert colors.shape[1] == 3

    # Automatically detect color value range and convert to 0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)

    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    # Write to PLY file
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,  #  (3, 3, 224, 224)
    gray_tensor: torch.Tensor,  #  (224, 224, 3)
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug",
) -> None:
    """
    1. original_0.png, original_1.png, original_2.png   (original image)
    2. gray_0.png, gray_1.png, gray_2.png              (gray image)
    3. overlay_0.png, overlay_1.png, overlay_2.png     (transparent image + annotation)
    """
    os.makedirs(save_dir, exist_ok=True)

    color_imgs = color_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    gray_imgs = gray_tensor.cpu().numpy().transpose(2, 0, 1)

    for i in range(3):
        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))

        gray_img = np.clip(gray_imgs[i], 0, 1) * 255
        gray_img = gray_img.astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))

        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img
        rgba[..., 3] = 77

        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)

        max_pos = np.unravel_index(gray_imgs[i].argmax(), gray_imgs[i].shape)
        x = max_pos[1]
        y = max_pos[0]

        point_radius = 5
        draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=(255, 0, 0, 255))

        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def apply_channel_wise_softmax(gray_tensor):
    """
    Apply softmax normalization independently to each grayscale channel
    Input shape: (H, W, C) -> Output shape: (H, W, C)
    All elements in each channel are processed by softmax and sum to 1
    """
    # Convert to PyTorch tensor (if not already)
    if not isinstance(gray_tensor, torch.Tensor):
        gray_tensor = torch.tensor(gray_tensor, dtype=torch.float32)

    # Separate each channel (C, H, W)
    channels = gray_tensor.permute(2, 0, 1)

    # Apply softmax to each channel and flatten
    softmax_channels = []
    for c in range(channels.shape[0]):
        channel = channels[c].flatten()
        softmax_channel = torch.softmax(channel, dim=0)
        softmax_channels.append(softmax_channel.view_as(channels[c]))

    # Merge channels and restore original shape (H, W, C)
    return torch.stack(softmax_channels, dim=2)


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"].cpu().numpy().item())

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"].cpu().numpy().item())
        eval_rot_y.append(eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"].cpu().numpy().item())
        eval_rot_z.append(eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"].cpu().numpy().item())

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (
        eval_trans,
        eval_rot_x,
        eval_rot_y,
        eval_rot_z,
        eval_grip,
        eval_coll,
    ) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if task not in self.eval_trans:
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS,
        cameras: list = rlbench_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds
        self.cameras = cameras

        print("Cameras:", self.cameras)
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        print("rot_ver", self.rot_ver)
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        params_to_optimize = filter(lambda p: p.requires_grad, self._network.parameters())

        self._optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self._lr,
            weight_decay=self._lambda_weight_l2,
        )

    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
        action_arm_flag=None,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        if action_arm_flag is not None:
            action_arm_flag_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(gt_rot, self._rotation_resolution)
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

            if action_arm_flag is not None:
                gt_flag = action_arm_flag[b]
                action_arm_flag_one_hot[b, gt_flag] = 1

        if action_arm_flag is not None:
            return (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,
                action_collision_one_hot,
                action_arm_flag_one_hot,
            )
        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218) or (bs, 219) if output_arm_flag is True
            feat = out["feat"].view(bs, -1)
            print("self._net_mod.output_arm_flag", self._net_mod.output_arm_flag)
            if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                # Last dimension is arm_flag
                rot_q = feat[:, 0 : self.num_all_rot]
                grip_q = feat[:, self.num_all_rot : self.num_all_rot + 2]
                collision_q = feat[:, self.num_all_rot + 2 : self.num_all_rot + 4]
                arm_flag = feat[:, -1:]  # (bs, 1)
            else:
                rot_q = feat[:, 0 : self.num_all_rot]
                grip_q = feat[:, self.num_all_rot : self.num_all_rot + 2]
                collision_q = feat[:, self.num_all_rot + 2 : self.num_all_rot + 4]
                arm_flag = None
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]), dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:4]
            arm_flag = (
                out["feat_ex_rot"].view(bs, -1)[:, 4:]
                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag
                else None
            )
        else:
            assert False

        y_q = None

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            return q_trans, rot_q, grip_q, collision_q, y_q, pts, arm_flag
        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][:, -1].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][:, -1].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        tasks = replay_sample["tasks"]
        return_out = {}

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound)
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(bs, action_rot, action_grip, action_ignore_collisions, device=self._device)

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(-self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"],
        )

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            q_trans, rot_q, grip_q, collision_q, y_q, pts, arm_flag = self.get_q(out, dims=(bs, nc, h, w))
        else:
            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(out, dims=(bs, nc, h, w))

        action_trans = self.get_action_trans(wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w))

        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(
                q_trans, action_trans
            ).mean()  # Soft-label cross-entropy loss. The target has the same shape as the input and is no longer one-hot encoded, but represented by class probabilities.
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()

                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()

                collision_loss = self._cross_entropy_loss(collision_q, action_collision_one_hot.argmax(-1)).mean()

            total_loss = trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss

            self._optimizer.zero_grad(set_to_none=True)

            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out

    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 8)

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)

        action_grip = action_gripper_pose[:, -1].int()  # (b,)
        return_out = {}

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )
            import open3d as o3d

            def vis_pcd(pc, rgb, save_path):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound)
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(bs, action_rot, action_grip, action_ignore_collisions, device=self._device)

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(-self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"],
        )

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            q_trans, rot_q, grip_q, collision_q, y_q, pts, arm_flag = self.get_q(out, dims=(bs, nc, h, w))
        else:
            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(out, dims=(bs, nc, h, w))

        action_trans = self.get_action_trans(wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w))

        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()

                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()

                collision_loss = self._cross_entropy_loss(collision_q, action_collision_one_hot.argmax(-1)).mean()

            total_loss = trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss
            self._optimizer.zero_grad(set_to_none=True)

            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out

    def update_real(
        self,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
        cameras=["3rd", "wrist"],
    ) -> dict:
        # assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        # assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        # assert replay_sample["low_dim_state"].shape[1:] == (
        #     1,
        #     self._net_mod.proprio_dim,
        # )

        # sample
        # action_rot_grip = replay_sample["rot_grip_action_indicies"][
        #     :, -1
        # ].int()  # (b, 4) of int

        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample[
            "gripper_pose"
        ]  # (b, 8)    对于quat 要求输入格式是 x y z w 但我们采集的真机数据是 w x y z

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            action_arm_flag = replay_sample["arm_flag"]  # (b, 1) of int

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        # print("before shape",action_rot.shape)
        # print("before first element",action_rot[0])

        # print("after shape",action_rot.shape)
        # print("after first element",action_rot[0])

        action_grip = action_gripper_pose[:, -1].int()  # (b,)
        # tasks = replay_sample["tasks"]
        # self._net_mod.tasks = tasks
        # self._net_mod.renderer.tasks = tasks
        # proprio = replay_sample["low_dim_state"]  # (b, 4)
        # if isinstance(self._network, DistributedDataParallel):
        #     if self._network.module.add_proprio:
        #         pass
        #         assert False
        #         # print("proprio is used !!!!")
        #     else:
        #         proprio = None
        # else:
        #     if self._network.add_proprio:
        #         pass
        #         assert False
        #         # print("proprio is used !!!!")
        #     else:
        #         proprio = None
        return_out = {}

        obs, pcd = real_utils._preprocess_inputs_real(replay_sample, cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            # 确保数据类型一致
            pc = pc.float()
            img_feat = img_feat.float()

            import open3d as o3d

            def vis_pcd(pc, rgb, save_path):
                # 将点云和颜色转换为二维的形状 (N, 3)
                # pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
                # rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

                # 将点云和颜色信息保存为 PLY 文件
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)  # 设置点云位置
                pcd.colors = o3d.utility.Vector3dVector(rgb)  # 设置对应的颜色
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])

            # vis_pcd(pc[0].cpu().numpy(),img_feat[0].cpu().numpy(),"/mnt/data1/3D_VLA/BridgeVLA/rvt_our/test_ori.ply")
            if self._transform_augmentation and backprop:
                # print("------ apply_se3_aug_con")
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize

            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound)
            # vis_pcd(pc[0].cpu().numpy(),img_feat[0].cpu().numpy(),"/mnt/data1/3D_VLA/BridgeVLA/rvt_our/test_ori.ply")

            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]
            # import pdb;pdb.set_trace()
            # vis_pcd(pc[0].cpu().numpy(),img_feat[0].cpu().numpy(),"/mnt/data1/3D_VLA/BridgeVLA/rvt_our/test_ori.ply")
            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
                action_arm_flag_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs,
                action_rot,
                action_grip,
                action_ignore_collisions,
                device=self._device,
                action_arm_flag=action_arm_flag,
            )
        else:
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(-self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes

        wpt_local = wpt_local.float()

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            # proprio=proprio,
            # lang_emb=lang_goal_embs,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"],
        )
        mvt1_img = out["mvt1_ori_img"][0, :, 3:6]
        # visualize mvt1_img
        # visualize_images_2(mvt1_img,mvt1_img,save_dir="/mnt/data1/3D_VLA/BridgeVLA/rvt_our/test_ori.png")
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            q_trans, rot_q, grip_q, collision_q, y_q, pts, arm_flag_q = self.get_q(out, dims=(bs, nc, h, w))
        else:
            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(out, dims=(bs, nc, h, w))
        # for key,item in out.items():
        #     if key != 'rev_trans' and key != 'mvt2':
        #         print(key,item.dtype)
        #     if key == 'mvt2':
        #         for _key,_item in item.items():
        #             print(_key,_item.dtype)

        action_trans = self.get_action_trans(wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w))

        loss_log = {}
        if backprop:
            # 计算损失
            # cross-entropy loss
            trans_loss = (
                self._cross_entropy_loss(q_trans, action_trans).mean()
            )  #  软标签交叉熵损失，target与输入的形状相同，且不再采用one-hot编码，而是用一个class probabilities来表示
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()

                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()

                collision_loss = self._cross_entropy_loss(collision_q, action_collision_one_hot.argmax(-1)).mean()

                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    arm_flag_loss = self._cross_entropy_loss(
                        arm_flag_q,
                        action_arm_flag_one_hot.argmax(-1),
                    ).mean()

            if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                total_loss = (
                    trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss + arm_flag_loss
                )
            else:
                total_loss = trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss
            # total_loss=trans_loss

            # check update
            # 初始化权重
            # layer_name = "mvt1.feat_fc_x.0"
            # initial_weights = dict(self._net_mod.named_modules())[layer_name].weight.data.clone()
            # for param in dict(self._net_mod.named_modules())[layer_name].parameters():
            #     param.requires_grad = False  # 禁止 conv1 层的梯度计算
            # 优化器步骤
            self._optimizer.zero_grad(set_to_none=True)
            # total_loss=total_loss.float()

            total_loss.backward()  # 标准精度下的反向传播
            # 存储优化前的参数值
            # before_params = {name: param.detach().clone() for name, param in self._network.named_parameters()}
            self._optimizer.step()
            # for name, param in self._network.named_parameters():
            #     if "language_model" in name :
            #         if torch.equal(before_params[name], param.detach()):
            #             print(f"{name} did not change.")
            #             continue
            #         else:
            #             # print(f"{name} was updated.")
            #             pass
            # if self.use_scheduler:
            #     self._lr_sched.step()
            # self.check_if_params_updated(self._net_mod,layer_name, initial_weights)

            if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                loss_log = {
                    "total_loss": total_loss.item(),
                    "trans_loss": trans_loss.item(),
                    "rot_loss_x": rot_loss_x.item(),
                    "rot_loss_y": rot_loss_y.item(),
                    "rot_loss_z": rot_loss_z.item(),
                    "grip_loss": grip_loss.item(),
                    "collision_loss": collision_loss.item(),
                    "arm_flag_loss": arm_flag_loss.item(),
                    "lr": self._optimizer.param_groups[0]["lr"],
                }
            else:
                loss_log = {
                    "total_loss": total_loss.item(),
                    "trans_loss": trans_loss.item(),
                    "rot_loss_x": rot_loss_x.item(),
                    "rot_loss_y": rot_loss_y.item(),
                    "rot_loss_z": rot_loss_z.item(),
                    "grip_loss": grip_loss.item(),
                    "collision_loss": collision_loss.item(),
                    "lr": self._optimizer.param_groups[0]["lr"],
                }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)
        # to do: eval loss flag
        # 当backprop=False且eval_log=True时，计算损失但不进行反向传播
        if not backprop and eval_log:
            with torch.no_grad():
                # 计算损失
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if self.add_rgc_loss:
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[
                            :,
                            0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                        ],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[
                            :,
                            1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                        ],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[
                            :,
                            2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                        ],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()

                    grip_loss = self._cross_entropy_loss(
                        grip_q,
                        action_grip_one_hot.argmax(-1),
                    ).mean()

                    collision_loss = self._cross_entropy_loss(collision_q, action_collision_one_hot.argmax(-1)).mean()

                    if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                        arm_flag_loss = self._cross_entropy_loss(
                            arm_flag_q,
                            action_arm_flag_one_hot.argmax(-1),
                        ).mean()

                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    total_loss = (
                        trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss + arm_flag_loss
                    )
                else:
                    total_loss = trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss

                eval_loss_log = {
                    "eval_total_loss": total_loss.item(),
                    "eval_trans_loss": trans_loss.item(),
                    "eval_rot_loss_x": rot_loss_x.item(),
                    "eval_rot_loss_y": rot_loss_y.item(),
                    "eval_rot_loss_z": rot_loss_z.item(),
                    "eval_grip_loss": grip_loss.item(),
                    "eval_collision_loss": collision_loss.item(),
                }

                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    eval_loss_log["eval_arm_flag_loss"] = arm_flag_loss.item()

                return_out.update(eval_loss_log)

        return return_out

    def _compute_log_probs(
        self,
        q_trans,
        rot_q,
        grip_q,
        collision_q,
        action_trans,
        action_rot_x_one_hot,
        action_rot_y_one_hot,
        action_rot_z_one_hot,
        action_grip_one_hot,
        action_collision_one_hot,
    ):
        """计算各个组件的log概率"""
        # 计算trans的log概率
        trans_log_probs = torch.log_softmax(q_trans, dim=-1)
        target_trans_indices = action_trans.argmax(-1)
        trans_target_log_probs = trans_log_probs.gather(-1, target_trans_indices.unsqueeze(-1)).squeeze(-1)

        # 计算rotation的log概率
        rot_x_log_probs = torch.log_softmax(
            rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes], dim=-1
        )
        rot_y_log_probs = torch.log_softmax(
            rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes], dim=-1
        )
        rot_z_log_probs = torch.log_softmax(
            rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes], dim=-1
        )

        target_rot_x_indices = action_rot_x_one_hot.argmax(-1)
        target_rot_y_indices = action_rot_y_one_hot.argmax(-1)
        target_rot_z_indices = action_rot_z_one_hot.argmax(-1)

        rot_x_target_log_probs = rot_x_log_probs.gather(-1, target_rot_x_indices.unsqueeze(-1)).squeeze(-1)
        rot_y_target_log_probs = rot_y_log_probs.gather(-1, target_rot_y_indices.unsqueeze(-1)).squeeze(-1)
        rot_z_target_log_probs = rot_z_log_probs.gather(-1, target_rot_z_indices.unsqueeze(-1)).squeeze(-1)

        # 计算grip和collision的log概率
        grip_log_probs = torch.log_softmax(grip_q, dim=-1)
        collision_log_probs = torch.log_softmax(collision_q, dim=-1)

        target_grip_indices = action_grip_one_hot.argmax(-1)
        target_collision_indices = action_collision_one_hot.argmax(-1)

        grip_target_log_probs = grip_log_probs.gather(-1, target_grip_indices.unsqueeze(-1)).squeeze(-1)
        collision_target_log_probs = collision_log_probs.gather(-1, target_collision_indices.unsqueeze(-1)).squeeze(-1)

        # 合并所有log概率
        total_log_probs = (
            trans_target_log_probs.mean()  # 对trans取平均，因为它是空间分布
            + rot_x_target_log_probs
            + rot_y_target_log_probs
            + rot_z_target_log_probs
            + grip_target_log_probs
            + collision_target_log_probs
        )

        return total_log_probs

    def dpo_update_real(
        self,
        replay_sample: dict,
        beta: float = 0.1,  # DPO温度参数
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
        cameras=["3rd", "wrist"],
    ) -> dict:
        """
        使用DPO (Direct Preference Optimization) 训练网络
        假设replay_sample包含正例和负例数据，通过key来区分
        包含KL散度项来约束与参考策略的差异
        """
        return_out = {}

        # 提取正例和负例数据
        positive_sample = replay_sample["positive"]
        negative_sample = replay_sample["negative"]

        # 处理正例数据
        pos_action_ignore_collisions = positive_sample["ignore_collisions"].unsqueeze(1).int()
        pos_action_gripper_pose = positive_sample["gripper_pose"]

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            pos_action_arm_flag = positive_sample["arm_flag"]

        pos_action_trans_con = pos_action_gripper_pose[:, 0:3]
        pos_action_rot = pos_action_gripper_pose[:, 3:7]
        pos_action_grip = pos_action_gripper_pose[:, -1].int()

        # 处理负例数据
        neg_action_ignore_collisions = negative_sample["ignore_collisions"].unsqueeze(1).int()
        neg_action_gripper_pose = negative_sample["gripper_pose"]

        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            neg_action_arm_flag = negative_sample["arm_flag"]

        neg_action_trans_con = neg_action_gripper_pose[:, 0:3]
        neg_action_rot = neg_action_gripper_pose[:, 3:7]
        neg_action_grip = neg_action_gripper_pose[:, -1].int()

        # 预处理正例数据
        pos_obs, pos_pcd = real_utils._preprocess_inputs_real(positive_sample, cameras)

        with torch.no_grad():
            pos_pc, pos_img_feat = rvt_utils.get_pc_img_feat(pos_obs, pos_pcd)
            pos_pc = pos_pc.float()
            pos_img_feat = pos_img_feat.float()

            if self._transform_augmentation and backprop:
                pos_action_trans_con, pos_action_rot, pos_pc = apply_se3_aug_con(
                    pcd=pos_pc,
                    action_gripper_pose=pos_action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                pos_action_trans_con = torch.tensor(pos_action_trans_con).to(pos_pc.device)
                pos_action_rot = torch.tensor(pos_action_rot).to(pos_pc.device)

            # 标准化四元数
            pos_action_rot = pos_action_rot.cpu().numpy()
            for i, _action_rot in enumerate(pos_action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                pos_action_rot[i] = _action_rot

            pos_pc, pos_img_feat = rvt_utils.move_pc_in_bound(
                pos_pc, pos_img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            pos_wpt = [x[:3] for x in pos_action_trans_con]

            pos_wpt_local = []
            pos_rev_trans = []
            for _pc, _wpt in zip(pos_pc, pos_wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                pos_wpt_local.append(a.unsqueeze(0))
                pos_rev_trans.append(b)

            pos_wpt_local = torch.cat(pos_wpt_local, axis=0)

            # TODO: Vectorize
            pos_pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pos_pc
            ]

            pos_bs = len(pos_pc)
            pos_nc = self._net_mod.num_img
            pos_h = pos_w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                pos_img_aug = self.img_aug
            else:
                pos_img_aug = 0

        # 预处理负例数据
        neg_obs, neg_pcd = real_utils._preprocess_inputs_real(negative_sample, cameras)

        with torch.no_grad():
            neg_pc, neg_img_feat = rvt_utils.get_pc_img_feat(neg_obs, neg_pcd)
            neg_pc = neg_pc.float()
            neg_img_feat = neg_img_feat.float()

            if self._transform_augmentation and backprop:
                neg_action_trans_con, neg_action_rot, neg_pc = apply_se3_aug_con(
                    pcd=neg_pc,
                    action_gripper_pose=neg_action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                neg_action_trans_con = torch.tensor(neg_action_trans_con).to(neg_pc.device)
                neg_action_rot = torch.tensor(neg_action_rot).to(neg_pc.device)

            # 标准化四元数
            neg_action_rot = neg_action_rot.cpu().numpy()
            for i, _action_rot in enumerate(neg_action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                neg_action_rot[i] = _action_rot

            neg_pc, neg_img_feat = rvt_utils.move_pc_in_bound(
                neg_pc, neg_img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            neg_wpt = [x[:3] for x in neg_action_trans_con]
            neg_wpt_local = []
            neg_rev_trans = []
            for _pc, _wpt in zip(neg_pc, neg_wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                neg_wpt_local.append(a.unsqueeze(0))
                neg_rev_trans.append(b)
            neg_wpt_local = torch.cat(neg_wpt_local, axis=0)

            neg_pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in neg_pc
            ]

            neg_bs = len(neg_pc)
            neg_nc = self._net_mod.num_img
            neg_h = neg_w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                neg_img_aug = self.img_aug
            else:
                neg_img_aug = 0

        # 生成正例的one-hot标签
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            (
                pos_action_rot_x_one_hot,
                pos_action_rot_y_one_hot,
                pos_action_rot_z_one_hot,
                pos_action_grip_one_hot,
                pos_action_collision_one_hot,
                pos_action_arm_flag_one_hot,
            ) = self._get_one_hot_expert_actions(
                pos_bs,
                pos_action_rot,
                pos_action_grip,
                pos_action_ignore_collisions,
                device=self._device,
                action_arm_flag=pos_action_arm_flag,
            )
        else:
            (
                pos_action_rot_x_one_hot,
                pos_action_rot_y_one_hot,
                pos_action_rot_z_one_hot,
                pos_action_grip_one_hot,
                pos_action_collision_one_hot,
            ) = self._get_one_hot_expert_actions(
                pos_bs, pos_action_rot, pos_action_grip, pos_action_ignore_collisions, device=self._device
            )

        # 生成负例的one-hot标签
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            (
                neg_action_rot_x_one_hot,
                neg_action_rot_y_one_hot,
                neg_action_rot_z_one_hot,
                neg_action_grip_one_hot,
                neg_action_collision_one_hot,
                neg_action_arm_flag_one_hot,
            ) = self._get_one_hot_expert_actions(
                neg_bs,
                neg_action_rot,
                neg_action_grip,
                neg_action_ignore_collisions,
                device=self._device,
                action_arm_flag=neg_action_arm_flag,
            )
        else:
            (
                neg_action_rot_x_one_hot,
                neg_action_rot_y_one_hot,
                neg_action_rot_z_one_hot,
                neg_action_grip_one_hot,
                neg_action_collision_one_hot,
            ) = self._get_one_hot_expert_actions(
                neg_bs, neg_action_rot, neg_action_grip, neg_action_ignore_collisions, device=self._device
            )

        # 处理旋转增强
        pos_rot_x_y = None
        neg_rot_x_y = None
        if self.rot_ver == 1:
            pos_rot_x_y = torch.cat(
                [
                    pos_action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    pos_action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            neg_rot_x_y = torch.cat(
                [
                    neg_action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    neg_action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                pos_rot_x_y += torch.randint(-self.rot_x_y_aug, self.rot_x_y_aug, size=pos_rot_x_y.shape).to(
                    pos_rot_x_y.device
                )
                pos_rot_x_y %= self._num_rotation_classes
                neg_rot_x_y += torch.randint(-self.rot_x_y_aug, self.rot_x_y_aug, size=neg_rot_x_y.shape).to(
                    neg_rot_x_y.device
                )
                neg_rot_x_y %= self._num_rotation_classes

        pos_wpt_local = pos_wpt_local.float()
        neg_wpt_local = neg_wpt_local.float()

        # 前向传播正例
        pos_out = self._network(
            pc=pos_pc,
            img_feat=pos_img_feat,
            lang_emb=None,
            img_aug=pos_img_aug,
            wpt_local=pos_wpt_local if self._network.training else None,
            rot_x_y=pos_rot_x_y if self.rot_ver == 1 else None,
            language_goal=positive_sample["lang_goal"],
        )

        # 前向传播负例
        neg_out = self._network(
            pc=neg_pc,
            img_feat=neg_img_feat,
            lang_emb=None,
            img_aug=neg_img_aug,
            wpt_local=neg_wpt_local if self._network.training else None,
            rot_x_y=neg_rot_x_y if self.rot_ver == 1 else None,
            language_goal=negative_sample["lang_goal"],
        )

        # 获取正例的Q值
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            pos_q_trans, pos_rot_q, pos_grip_q, pos_collision_q, pos_y_q, pos_pts, pos_arm_flag_q = self.get_q(
                pos_out, dims=(pos_bs, pos_nc, pos_h, pos_w)
            )
        else:
            pos_q_trans, pos_rot_q, pos_grip_q, pos_collision_q, pos_y_q, pos_pts = self.get_q(
                pos_out, dims=(pos_bs, pos_nc, pos_h, pos_w)
            )

        # 获取负例的Q值
        if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
            neg_q_trans, neg_rot_q, neg_grip_q, neg_collision_q, neg_y_q, neg_pts, neg_arm_flag_q = self.get_q(
                neg_out, dims=(neg_bs, neg_nc, neg_h, neg_w)
            )
        else:
            neg_q_trans, neg_rot_q, neg_grip_q, neg_collision_q, neg_y_q, neg_pts = self.get_q(
                neg_out, dims=(neg_bs, neg_nc, neg_h, neg_w)
            )

        # 计算正例的action_trans
        pos_action_trans = self.get_action_trans(
            pos_wpt_local, pos_pts, pos_out, None, dims=(pos_bs, pos_nc, pos_h, pos_w)
        )

        # 计算负例的action_trans
        neg_action_trans = self.get_action_trans(
            neg_wpt_local, neg_pts, neg_out, None, dims=(neg_bs, neg_nc, neg_h, neg_w)
        )

        # 计算正例的log概率
        pos_log_probs = self._compute_log_probs(
            pos_q_trans,
            pos_rot_q,
            pos_grip_q,
            pos_collision_q,
            pos_action_trans,
            pos_action_rot_x_one_hot,
            pos_action_rot_y_one_hot,
            pos_action_rot_z_one_hot,
            pos_action_grip_one_hot,
            pos_action_collision_one_hot,
        )

        # 计算负例的log概率
        neg_log_probs = self._compute_log_probs(
            neg_q_trans,
            neg_rot_q,
            neg_grip_q,
            neg_collision_q,
            neg_action_trans,
            neg_action_rot_x_one_hot,
            neg_action_rot_y_one_hot,
            neg_action_rot_z_one_hot,
            neg_action_grip_one_hot,
            neg_action_collision_one_hot,
        )

        # 计算参考模型的log概率（如果存在reference_model）
        pos_ref_log_probs = None
        neg_ref_log_probs = None
        if hasattr(self, "reference_model"):
            with torch.no_grad():
                # 参考模型前向传播正例
                pos_ref_out = self.reference_model(
                    pc=pos_pc,
                    img_feat=pos_img_feat,
                    lang_emb=None,
                    img_aug=pos_img_aug,
                    wpt_local=pos_wpt_local if self.reference_model.training else None,
                    rot_x_y=pos_rot_x_y if self.rot_ver == 1 else None,
                    language_goal=positive_sample["lang_goal"],
                )

                # 参考模型前向传播负例
                neg_ref_out = self.reference_model(
                    pc=neg_pc,
                    img_feat=neg_img_feat,
                    lang_emb=None,
                    img_aug=neg_img_aug,
                    wpt_local=neg_wpt_local if self.reference_model.training else None,
                    rot_x_y=neg_rot_x_y if self.rot_ver == 1 else None,
                    language_goal=negative_sample["lang_goal"],
                )

                # 获取参考模型的Q值
                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    (
                        pos_ref_q_trans,
                        pos_ref_rot_q,
                        pos_ref_grip_q,
                        pos_ref_collision_q,
                        pos_ref_y_q,
                        pos_ref_pts,
                        pos_ref_arm_flag_q,
                    ) = self.get_q(pos_ref_out, dims=(pos_bs, pos_nc, pos_h, pos_w))
                    (
                        neg_ref_q_trans,
                        neg_ref_rot_q,
                        neg_ref_grip_q,
                        neg_ref_collision_q,
                        neg_ref_y_q,
                        neg_ref_pts,
                        neg_ref_arm_flag_q,
                    ) = self.get_q(neg_ref_out, dims=(neg_bs, neg_nc, neg_h, neg_w))
                else:
                    pos_ref_q_trans, pos_ref_rot_q, pos_ref_grip_q, pos_ref_collision_q, pos_ref_y_q, pos_ref_pts = (
                        self.get_q(pos_ref_out, dims=(pos_bs, pos_nc, pos_h, pos_w))
                    )
                    neg_ref_q_trans, neg_ref_rot_q, neg_ref_grip_q, neg_ref_collision_q, neg_ref_y_q, neg_ref_pts = (
                        self.get_q(neg_ref_out, dims=(neg_bs, neg_nc, neg_h, neg_w))
                    )

                # 计算参考模型的log概率
                pos_ref_log_probs = self._compute_log_probs(
                    pos_ref_q_trans,
                    pos_ref_rot_q,
                    pos_ref_grip_q,
                    pos_ref_collision_q,
                    pos_action_trans,
                    pos_action_rot_x_one_hot,
                    pos_action_rot_y_one_hot,
                    pos_action_rot_z_one_hot,
                    pos_action_grip_one_hot,
                    pos_action_collision_one_hot,
                )

                neg_ref_log_probs = self._compute_log_probs(
                    neg_ref_q_trans,
                    neg_ref_rot_q,
                    neg_ref_grip_q,
                    neg_ref_collision_q,
                    neg_action_trans,
                    neg_action_rot_x_one_hot,
                    neg_action_rot_y_one_hot,
                    neg_action_rot_z_one_hot,
                    neg_action_grip_one_hot,
                    neg_action_collision_one_hot,
                )

        loss_log = {}
        if backprop:
            # 计算DPO损失
            # DPO损失公式: -log(sigmoid(beta * (log_p_w - log_p_l)))
            log_diff = pos_log_probs - neg_log_probs
            dpo_loss = -torch.log(torch.sigmoid(beta * log_diff)).mean()

            # 计算KL散度损失（如果存在参考模型）
            kl_loss = 0.0
            if hasattr(self, "reference_model") and pos_ref_log_probs is not None:
                # KL散度 = E[log(p_current) - log(p_ref)]
                pos_kl = (pos_log_probs - pos_ref_log_probs).mean()
                neg_kl = (neg_log_probs - neg_ref_log_probs).mean()
                kl_loss = pos_kl + neg_kl

            # 可选：添加额外的监督损失
            if self.add_rgc_loss:
                # 正例的监督损失
                pos_trans_loss = self._cross_entropy_loss(pos_q_trans, pos_action_trans).mean()
                pos_rot_loss_x = self._cross_entropy_loss(
                    pos_rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes],
                    pos_action_rot_x_one_hot.argmax(-1),
                ).mean()
                pos_rot_loss_y = self._cross_entropy_loss(
                    pos_rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes],
                    pos_action_rot_y_one_hot.argmax(-1),
                ).mean()
                pos_rot_loss_z = self._cross_entropy_loss(
                    pos_rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes],
                    pos_action_rot_z_one_hot.argmax(-1),
                ).mean()
                pos_grip_loss = self._cross_entropy_loss(
                    pos_grip_q,
                    pos_action_grip_one_hot.argmax(-1),
                ).mean()
                pos_collision_loss = self._cross_entropy_loss(
                    pos_collision_q, pos_action_collision_one_hot.argmax(-1)
                ).mean()

                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    pos_arm_flag_loss = self._cross_entropy_loss(
                        pos_arm_flag_q,
                        pos_action_arm_flag_one_hot.argmax(-1),
                    ).mean()

                # # 负例的监督损失（权重较小）
                # neg_trans_loss = self._cross_entropy_loss(neg_q_trans, neg_action_trans).mean()
                # neg_rot_loss_x = self._cross_entropy_loss(
                #     neg_rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes],
                #     neg_action_rot_x_one_hot.argmax(-1),
                # ).mean()
                # neg_rot_loss_y = self._cross_entropy_loss(
                #     neg_rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes],
                #     neg_action_rot_y_one_hot.argmax(-1),
                # ).mean()
                # neg_rot_loss_z = self._cross_entropy_loss(
                #     neg_rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes],
                #     neg_action_rot_z_one_hot.argmax(-1),
                # ).mean()
                # neg_grip_loss = self._cross_entropy_loss(
                #     neg_grip_q,
                #     neg_action_grip_one_hot.argmax(-1),
                # ).mean()
                # neg_collision_loss = self._cross_entropy_loss(
                #     neg_collision_q, neg_action_collision_one_hot.argmax(-1)
                # ).mean()

                # 总损失 = DPO损失 + KL散度损失 + 正例监督损失 + 负例监督损失（权重较小）
                total_loss = (
                    dpo_loss
                    + 0.1 * kl_loss  # KL散度权重
                    + 0.5
                    * (
                        pos_trans_loss
                        + pos_rot_loss_x
                        + pos_rot_loss_y
                        + pos_rot_loss_z
                        + pos_grip_loss
                        + pos_collision_loss
                        + pos_arm_flag_loss
                    )
                    #  + 0.1 * (neg_trans_loss + neg_rot_loss_x + neg_rot_loss_y + neg_rot_loss_z +
                    #     neg_grip_loss + neg_collision_loss + neg_arm_flag_loss)
                )
            else:
                # 总损失 = DPO损失 + KL散度损失
                total_loss = dpo_loss + 0.1 * kl_loss

            # 反向传播
            self._optimizer.zero_grad(set_to_none=True)

            total_loss.backward()
            self._optimizer.step()

            # 记录损失
            loss_log = {
                "dpo_total_loss": total_loss.item(),
                "dpo_loss": dpo_loss.item(),
                "kl_loss": kl_loss.item(),
                "pos_log_probs": pos_log_probs.mean().item(),
                "neg_log_probs": neg_log_probs.mean().item(),
                "log_diff": log_diff.mean().item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }

            if hasattr(self, "reference_model") and pos_ref_log_probs is not None:
                loss_log.update(
                    {
                        "pos_ref_log_probs": pos_ref_log_probs.mean().item(),
                        "neg_ref_log_probs": neg_ref_log_probs.mean().item(),
                    }
                )

            if self.add_rgc_loss:
                loss_log.update(
                    {
                        "pos_trans_loss": pos_trans_loss.item(),
                        "pos_rot_loss_x": pos_rot_loss_x.item(),
                        "pos_rot_loss_y": pos_rot_loss_y.item(),
                        "pos_rot_loss_z": pos_rot_loss_z.item(),
                        "pos_grip_loss": pos_grip_loss.item(),
                        "pos_collision_loss": pos_collision_loss.item(),
                    }
                )
                if hasattr(self._net_mod, "output_arm_flag") and self._net_mod.output_arm_flag:
                    loss_log.update(
                        {
                            "pos_arm_flag_loss": pos_arm_flag_loss.item(),
                        }
                    )

            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        # 评估模式下的损失计算
        if not backprop and eval_log:
            with torch.no_grad():
                log_diff = pos_log_probs - neg_log_probs
                dpo_loss = -torch.log(torch.sigmoid(beta * log_diff)).mean()

                kl_loss = 0.0
                if hasattr(self, "reference_model") and pos_ref_log_probs is not None:
                    pos_kl = (pos_log_probs - pos_ref_log_probs).mean()
                    neg_kl = (neg_log_probs - neg_ref_log_probs).mean()
                    kl_loss = pos_kl + neg_kl

                eval_loss_log = {
                    "eval_dpo_loss": dpo_loss.item(),
                    "eval_kl_loss": kl_loss.item(),
                    "eval_pos_log_probs": pos_log_probs.mean().item(),
                    "eval_neg_log_probs": neg_log_probs.mean().item(),
                    "eval_log_diff": log_diff.mean().item(),
                }

                if hasattr(self, "reference_model") and pos_ref_log_probs is not None:
                    eval_loss_log.update(
                        {
                            "eval_pos_ref_log_probs": pos_ref_log_probs.mean().item(),
                            "eval_neg_ref_log_probs": neg_ref_log_probs.mean().item(),
                        }
                    )

                return_out.update(eval_loss_log)

        return return_out

    @torch.no_grad()
    def act(
        self,
        step: int,
        observation: dict,
        visualize=False,
        visualize_save_dir="",
        return_gembench_action=False,
    ) -> ActResult:
        language_goal = observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound)
        pc_ori = pc[0].clone()
        img_feat_ori = img_feat[0].clone()
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            language_goal=language_goal,
        )
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        if visualize:
            print("Visualizing")
            save_dir = visualize_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(save_dir, f"step{step!s}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mvt1_img = out["mvt1_ori_img"][0, :, 3:6]
            mvt2_img = out["mvt2_ori_img"][0, :, 3:6]
            q_trans_1 = q_trans[0, :, :3].clone().view(224, 224, 3)
            q_trans_2 = q_trans[0, :, 3:6].clone().view(224, 224, 3)
            q_trans_1 = apply_channel_wise_softmax(q_trans_1) * 100
            q_trans_2 = apply_channel_wise_softmax(q_trans_2) * 100
            visualize_images(mvt1_img, q_trans_1, save_dir=os.path.join(save_dir, "mvt1"))
            visualize_images(mvt2_img, q_trans_2, save_dir=os.path.join(save_dir, "mvt2"))
            save_point_cloud_with_color(
                os.path.join(save_dir, "point_cloud.ply"),
                pc_ori.cpu().numpy(),
                img_feat_ori.cpu().numpy(),
                pred_wpt[0].cpu().numpy(),
            )
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
                # [1.0],  # debug!!!!!!
            )
        )

        if return_gembench_action:
            continuous_action = np.concatenate(
                [
                    pred_wpt[0].cpu().numpy(),
                    pred_rot_quat[0],
                    pred_grip[0].cpu().numpy(),
                ],
                -1,
            )
            return continuous_action
        return ActResult(continuous_action)

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q)

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(pred_rot.cpu(), self._rotation_resolution)
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1), mvt1_or_mvt2=True, dyn_cam_info=dyn_cam_info, out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
