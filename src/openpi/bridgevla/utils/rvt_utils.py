# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/utils/rvt_utils.py
import pdb
import argparse
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from openpi.bridgevla.models.peract_official import PreprocessAgent2

import numpy as np

import numpy as np

def get_pc_img_feat(obs, pcd, bounds=None):
    """
    Preprocess single‐example data (no batch dimension) from NumPy arrays.
    
    Args:
        obs: list of image feature arrays, each of shape (C_img, H, W)
        pcd: list of point‐cloud arrays, each of shape (3, H, W)
        bounds: unused (kept for API compatibility)
    
    Returns:
        pc:       NumPy array of shape (total_num_points, 3)
        img_feat: NumPy array of shape (total_num_points, C_img), normalized to [0,1]
    """
    # --- concatenate point clouds from all cameras ---
    # For each p in pcd of shape (3, H, W):
    #   1) transpose to (H, W, 3)
    #   2) reshape to (H*W, 3)
    pc_list = [
        p.transpose(1, 2, 0).reshape(-1, 3)
        for p in pcd
    ]
    # final shape: (sum(H*W over cameras), 3)
    pc = np.concatenate(pc_list, axis=0)
    
    # --- extract and concatenate image features ---
    # obs is a list of arrays of shape (C_img, H, W)
    
    C_img = obs[0][0].shape[0]
    feat_list = [
        f.transpose(1, 2, 0).reshape(-1, C_img)
        for f in obs[0]
    ]
    # final shape: (sum(H*W over cameras), C_img)
    img_feat = np.concatenate(feat_list, axis=0)
    
    # --- normalize image features from [-1,1] → [0,1] ---
    img_feat = (img_feat + 1.0) / 2.0
    
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


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
         "--tasks", type=str, nargs="+", default=["all"]
    )
    parser.add_argument("--model-folder", type=str,default="")
    parser.add_argument("--eval-datafolder", type=str,default="")
    parser.add_argument("--visualize_root_dir", type=str,default="")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-name", type=str, default="test/1")
    parser.add_argument("--model-name", type=str, default="model_80.pth")
    parser.add_argument("--use-input-place-with-mean", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize", action="store_true",default=False)    
    return parser





def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in mvt1. "
                    "Be cautious if you are using a two stage network."
                )
                model.mvt1.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)
    return epoch



RLBENCH_TASKS = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]


COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_box",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "insert_onto_square_peg",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula",
    "close_laptop_lid",
    "stack_cups",
]