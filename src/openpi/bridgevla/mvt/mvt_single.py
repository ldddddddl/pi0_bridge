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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt_single.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import torch
from torch import nn
from einops import rearrange
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.attn import (
    FixedPositionalEncoding,
)
from bridgevla.mvt.raft_utils import ConvexUpSample
from PIL import Image



class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        img_feat_dim,
        feat_dim,
        im_channels,
        activation,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
        load_pretrain=False,
        pretrain_path=None,
        output_arm_flag=False,
    ):
        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat
        self.output_arm_flag = output_arm_flag

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img
        # Modify it to adapt to vlm. 16**2 is the number of patches in the image
        self.num_pat_img = 16  

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1


        # Hardcoded for vlm
        self.vlm_dim=2048  

        self.up0 = ConvexUpSample(
            in_dim=self.vlm_dim,
            out_dim=1,
            up_ratio=self.img_patch_size,
        )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.vlm_dim
            # Because we will concatenate the max-pooled image tokens and the image tokens corresponding to the waypoint later.
            if self.cvx_up:
                feat_fc_dim += self.vlm_dim
            else:
                feat_fc_dim += self.final_dim
            

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            if self.rot_ver == 0:
                out_dim = feat_out_size + (2 if self.output_arm_flag else 0)
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    out_dim,
                )
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
                feat_out_size_ex_rot = feat_out_size_ex_rot + (2 if self.output_arm_flag else 0)
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, feat_out_size_ex_rot
                    )

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from bridgevla.mvt.renderer import select_feat_from_hm

        from transformers import (
            PaliGemmaProcessor,
            PaliGemmaForConditionalGeneration,
        )
        from safetensors import safe_open
        import json

        def load_all_params(checkpoint_dir):
            # Load the index file
            with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
                index = json.load(f)
            
            all_params = {}
            for shard_file in set(index["weight_map"].values()):
                with safe_open(f"{checkpoint_dir}/{shard_file}", framework="pt") as f:
                    for key in f.keys():
                        # Remove the "module." prefix
                        clean_key = key.replace("module.", "")
                        all_params[clean_key] = f.get_tensor(key)
            return all_params


        model_id = "google/paligemma-3b-pt-224"
        if load_pretrain:
            assert pretrain_path is not None

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            self.processor = PaliGemmaProcessor.from_pretrained(model_id) 
            pretrained_dir=pretrain_path
            print("The pretrained path is:",pretrained_dir)
            all_params = load_all_params(pretrained_dir)

            # Separate the base model parameters (assuming the original model parameter names do not contain "up0")
            base_params = {k: v for k, v in all_params.items() if not k.startswith("up0.")}

            # Separate the custom layer parameters
            custom_params = {k.replace("up0.",""): v for k, v in all_params.items() if k.startswith("up0.")}
            # Load parameters (strict mode)
            missing_keys, unexpected_keys = self.model.load_state_dict(base_params, strict=False)
            print("Missing keys  base:", missing_keys)  # Should be an empty list
            print("Unexpected keys base:", unexpected_keys) # Should be an empty list
            # Load parameters
            missing_keys_up0, unexpected_keys_up0 = self.up0.load_state_dict(custom_params, strict=True)
            print("Missing keys up0:", missing_keys_up0)  # Should be an empty list
            print("Unexpected keys up0 :", unexpected_keys_up0) # Should be an empty list
            import time
            time.sleep(5)

            
        else:

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)   
            print("You are loading original paligemma model!")

        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        Transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    @staticmethod
    def trans_cuda_tensor_2_PIL(cuda_tensor):
        # Default c,h,w, and 0,1
        # 1. Move the tensor from GPU to CPU
        tensor_cpu = cuda_tensor.cpu()

        # 2. Convert to a numpy array and adjust the dimension order [3, 224, 224] -> [224, 224, 3]
        image = tensor_cpu.permute(1, 2, 0).numpy()

        # 3. Convert the values from [0, 1] to integers in [0, 255] and cast to uint8 type
        image = (image * 255).astype('uint8')

        # 4. Create a PIL image object
        pil_image = Image.fromarray(image)

        # 5. Convert to RGB format (ensure the image is RGB)
        pil_image_rgb = pil_image.convert("RGB")
        return pil_image_rgb

    def forward(
        self,
        img,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        forward_no_feat=False,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param rot_x_y: (bs, 2)
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        assert num_img == self.num_img
        assert h == w == self.img_size
        # only use rgb part
        # print("input image feature shape:",img.shape)
        img = img[:,:, 3:6, :, :] # bs,3,3,224,224


        prompts =[ text[0][0] for text in language_goal]# ["text1","text2"...]
        # print("The prompts:",prompts)
        images = [[MVT.trans_cuda_tensor_2_PIL(example)for example in examples] for examples in img]# bs,3


        assert len(prompts)==len(images)
        model_inputs = self.processor(text=prompts, images=images, return_tensors="pt",padding="longest")
        model_inputs = model_inputs.to(self.model.dtype).to(self.model.device)
        outputs = self.model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  

        x = hidden_states[-1]  # get the features of the last layer


        # get image tokens
        image_tokens= []

        # Process every batch
        for i in range(bs):
            # Get the ids and output of the current batch
            current_ids = model_inputs["attention_mask"][i]
            current_output = x[i]
            
            # Extract tokens corresponding to non-zero ids
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]  # Find the indices of non-zero ids
            non_zero_output = current_output[non_zero_indices]  # Extract the token outputs corresponding to these non-zero ids
            
            # Take the first 256 tokens (if the number of non-zero tokens is greater than 256, take the first 256)
            assert non_zero_output.shape[0] > 256*self.num_img
            non_zero_output = non_zero_output[:256*self.num_img]
            
            # Add the processed output to the new output list
            image_tokens.append(non_zero_output)

        # concat all the output
        image_tokens = torch.stack(image_tokens)
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=self.num_img, h1=self.num_pat_img, h2=self.num_pat_img) 
        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32)
        
        trans = self.up0(x)
        trans = trans.view(bs, self.num_img, h, w)


        if not forward_no_feat:

            # get wpt_local while testing
            if not self.training:
                wpt_local = self.get_wpt(
                    out={"trans": trans.clone().detach()},
                    dyn_cam_info=None,
                )

            # projection
            # (bs, 1, num_img, 2)
            wpt_img = self.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                dyn_cam_info=None,
            )
            wpt_img = wpt_img.reshape(bs * self.num_img, 2)

            # add noise to wpt image while training
            if self.training:
                wpt_img = mvt_utils.add_uni_noi(
                    wpt_img, self.wpt_img_aug * self.img_size
                )
                wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

            _wpt_img = wpt_img / self.img_patch_size
            _u = x
            assert (
                0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
            ), print(_wpt_img, x.shape)

            _wpt_img = _wpt_img.unsqueeze(1)
            _feat = select_feat_from_hm(_wpt_img, _u)[0]
            _feat = _feat.view(bs, -1)
            feat.append(_feat)
            feat = torch.cat(feat, dim=-1)

            if self.rot_ver == 0:
                feat = self.feat_fc(feat)
                out = {"feat": feat}
            elif self.rot_ver == 1:
                # features except rotation
                feat_ex_rot = self.feat_fc_ex_rot(feat)

                # batch normalized features for rotation
                feat_rot = self.feat_fc_init_bn(feat)
                # feat_rot = self.feat_fc_init_bn(feat)
                feat_x = self.feat_fc_x(feat_rot)

                if self.training:
                    rot_x = rot_x_y[..., 0].view(bs, 1)
                else:
                    # sample with argmax
                    rot_x = feat_x.argmax(dim=1, keepdim=True)

                # rot_x_pe = self.feat_fc_pe(rot_x).to(torch.bfloat16)
                rot_x_pe = self.feat_fc_pe(rot_x)
                feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

                if self.training:
                    rot_y = rot_x_y[..., 1].view(bs, 1)
                else:
                    rot_y = feat_y.argmax(dim=1, keepdim=True)
                rot_y_pe = self.feat_fc_pe(rot_y)
                # rot_y_pe = self.feat_fc_pe(rot_y).to(torch.bfloat16)
                feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
                out = {
                    "feat_ex_rot": feat_ex_rot,
                    "feat_x": feat_x,
                    "feat_y": feat_y,
                    "feat_z": feat_z,
                }
        
        else:
            out = {}

        out.update({"trans": trans})

        return out





    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt


    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()



