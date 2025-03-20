from functools import partial
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from src.adamatcher.backbone.feature_interaction import FeatureAttention
from src.utils.plotting import make_matching_fine

from loguru import logger

# import pdb


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class FineModule(nn.Module):
    def __init__(self, resolution, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.scale_l0, self.scale_l1, self.scale_l2 = resolution
        self.scale_l1l2 = self.scale_l1 // self.scale_l2
        self.count = 0

        self.attention = FeatureAttention(layer_num=1, d_model=128)
        self.W = 5
        d_model_c = 256
        d_model_f = 128
        self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
        self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)
        self.post_scale = True

        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(32, self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 1, (1, 1)),
        )

    def get_expected_locs(self,
                          kptsfeat1,
                          kptsfeat0_from1,
                          patch0_center_coord_l2,
                          scale,
                          flag=False,
                          heatmap_zs=None):
        """
        kptsfeat1:       [k, ww, c]
        kptsfeat0_from1: [k, ww, c]
        """
        # scale = scale**0.5
        k, WW, C = kptsfeat1.shape
        W = int(math.sqrt(WW))
        # self.W, self.WW, self.C = W, WW, C
        if flag and scale > 1.0:
            WW0 = kptsfeat0_from1.shape[1]
            W0 = int(math.sqrt(WW0))
            kptsfeat0_from1 = (kptsfeat0_from1.permute(0, 2, 1).view(
                k, C, W0, W0).contiguous())
            kptsfeat0_from1 = F.interpolate(
                kptsfeat0_from1,
                scale_factor=scale,
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=True,
            )  # [k, c, nw, nw]
            kptsfeat0_from1 = kptsfeat0_from1.flatten(2).permute(
                0, 2, 1).contiguous()
        _, NWW, _ = kptsfeat0_from1.shape
        nW = int(math.sqrt(NWW))

        kptsfeat1_picked = kptsfeat1[:, WW // 2, :].unsqueeze(1)
        att = torch.einsum('blc,bnc->bln', kptsfeat0_from1,
                           kptsfeat1_picked)  # [k, (hw), 1]
        sim_matrix = rearrange(kptsfeat0_from1 * att,
                               'n (h w) c -> n c h w',
                               h=nW,
                               w=nW).contiguous()
        heatmap = (self.heatmap_conv(sim_matrix).permute(0, 2, 3, 1).flatten(
            1, 2).contiguous().squeeze(1))
        softmax_temp = 1.0  # 1. / C ** .5
        heatmap = torch.softmax(softmax_temp * heatmap, dim=1).view(-1, nW, nW)

        # if heatmap_zs is not None:
        #     heatmap = torch.cat([heatmap, heatmap_zs], dim=0)

        # compute coordinates from heatmap
        relative_kpts0from1 = dsnt.spatial_expectation2d(heatmap[None],
                                                         True)[0]  # [M, 2]
        kpts0_from1_l2 = patch0_center_coord_l2 + relative_kpts0from1 * (
            W // 2)  # (W // 2)

        # compute std over <x, y>
        grid_normalized = create_meshgrid(nW, nW,
                                          True, heatmap.device).reshape(
                                              1, -1, 2)  # [1, NWW, 2]
        var = (
            torch.sum(grid_normalized**2 * heatmap.view(-1, NWW, 1), dim=1) -
            relative_kpts0from1**2)  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)),
                        -1)  # [M]  clamp needed for numerical stability

        return relative_kpts0from1, kpts0_from1_l2, std
    
    def compute_zeroshot_fine_loss(self, feat_f0, feat_f1, radius, W, data):
        M, WW, C = feat_f0.shape
        Nz = len(data['zs_b_ids']) if 'zs_b_ids' in data else 0
        Ng = len(data['b_ids']) if 'b_ids' in data else 0
        logger.info(f"Nz, Ng: {Nz}, {Ng}")
        pt0_f_int = data['zs_pt0_f_int']
        pt0_f_float = data['zs_pt0_f_float']  # (Nz, 2) in hw_f coordinates
        pt_x = (pt0_f_float[:, 0] - pt0_f_int[:, 0]) / radius
        pt_y = (pt0_f_float[:, 1] - pt0_f_int[:, 1]) / radius
        grid = torch.stack([pt_x, pt_y], dim=1)[:, None, None]  # (Nz, 1, 1, 2)
        grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')
        feat_f0_picked = rearrange(feat_f0[-Nz:], 'n (h w) c -> n c h w', h=W, w=W)
        feat_f0_picked = grid_sample(feat_f0_picked, grid).squeeze()  # [(Nz, c)]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1[-Nz:])  # (Nz, ww)
        softmax_temp = 1. / C ** .5
        heatmap_z = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)  # (Nz, w, w)

        self.spvc_zeroshot_fine(radius, data)

        return heatmap_z

    @torch.no_grad()
    def spvc_zeroshot_fine(self, radius, data):
        pt1_f_int = data['zs_pt1_f_int']
        pt1_f_float = data['zs_pt1_f_float']
        expec_f_zs = (pt1_f_float - pt1_f_int) / radius
        data.update({"expec_f_zs": expec_f_zs})

    def forward(
        self,
        data,
        feat_d2_0,
        feat_d2_1,
        feat_d8_0,
        feat_d8_1,
    ):
        """
        feat_d2_0: [N, c, h0_l2, w0_l2]
        """
        self.bs = data['image0'].size(0)
        self.device = feat_d2_0.device
        m_bids = data['m_bids']
        patch_size_l1l2 = self.scale_l1 // self.scale_l2

        feat_map1_unfold_pre_pre = F.unfold(
            feat_d2_1,
            kernel_size=(self.W, self.W),
            stride=patch_size_l1l2,
            padding=self.W // 2,
        ).contiguous()
        feat_map1_unfold_pre = rearrange(feat_map1_unfold_pre_pre,
                                     'n (c ww) l -> n l ww c',
                                     ww=self.W**2).contiguous()
        feat_map1_unfold = rearrange(feat_map1_unfold_pre,
                                     'n l (w1 w2) c -> n l w1 w2 c',
                                     w1=self.W,
                                     w2=self.W).contiguous()

        feat_map0_unfold_pre_pre = F.unfold(
            feat_d2_0,
            kernel_size=(self.W, self.W),
            stride=patch_size_l1l2,
            padding=self.W // 2,
        ).contiguous()
        feat_map0_unfold_pre = rearrange(feat_map0_unfold_pre_pre,
                                     'n (c ww) l -> n l ww c',
                                     ww=self.W**2).contiguous()
        feat_map0_unfold = rearrange(feat_map0_unfold_pre,
                                     'n l (w1 w2) c -> n l w1 w2 c',
                                     w1=self.W,
                                     w2=self.W).contiguous()
        


        if data['zs'].sum():
            W = self.W
            radius = W // 2
            zs = data['zs']
            pt0_i = data['zs_pt0_i']
            pt1_i = data['zs_pt1_i']
            logger.info(f"pt0_i, pt1_i: {pt0_i.shape}, {pt1_i.shape}")
            zs_b_ids = data['zs_b_ids']
            scale_c = data['hw0_i'][0] / data['hw0_c'][0]  # 8.0
            scale_f = data['hw0_i'][0] / data['hw0_f'][0]  # 2.0
            pt0_c_int = (pt0_i / scale_c).round().long()
            pt1_c_int = (pt1_i / scale_c).round().long()
            pt0_f_int = pt0_c_int * patch_size_l1l2 # stride
            pt1_f_int = pt1_c_int * patch_size_l1l2 # stride
            pt0_f_float = pt0_i / scale_f
            pt1_f_float = pt1_i / scale_f
            indices = ((pt0_f_float[:, 0] - pt0_f_int[:, 0]).abs() <= radius) & \
                      ((pt0_f_float[:, 1] - pt0_f_int[:, 1]).abs() <= radius) & \
                      ((pt1_f_float[:, 0] - pt1_f_int[:, 0]).abs() <= radius) & \
                      ((pt1_f_float[:, 1] - pt1_f_int[:, 1]).abs() <= radius)
            zs_ci_ids = pt0_c_int[:, 0] + pt0_c_int[:, 1] * data['hw0_c'][1]
            zs_cj_ids = pt1_c_int[:, 0] + pt1_c_int[:, 1] * data['hw1_c'][1]
            feat_f0_z = feat_map0_unfold[zs][zs_b_ids[indices], zs_ci_ids[indices]]  # [n, ww, cf]
            feat_f1_z = feat_map1_unfold[zs][zs_b_ids[indices], zs_cj_ids[indices]]  #TODO: leverage

            data.update({
                'zs_b_ids': zs_b_ids[indices],
                'zs_cj_ids': zs_cj_ids[indices],
                'zs_ci_ids': zs_ci_ids[indices],
                'zs_pt0_f_int': pt0_f_int[indices],
                'zs_pt1_f_int': pt1_f_int[indices],
                'zs_pt0_f_float': pt0_f_float[indices],
                'zs_pt1_f_float': pt1_f_float[indices],
            })

        hw1_d2, hw0_d2 = data['hw1_d2'], data['hw0_d2']
        overlap_scores0 = data['overlap_scores0']
        overlap_scores1 = data['overlap_scores1']
        scales = data['scales']

        kpts1_l2, kpts0from1_l2 = [], []
        relative_kpts0from1_l2 = []
        patch0_center_coord_l2 = []
        std0, i_ids1_l2, j_ids1_l2 = [], [], []

        kpts0_l2, kpts1from0_l2 = [], []
        relative_kpts1from0_l2 = []
        patch1_center_coord_l2 = []
        std1, i_ids0_l2, j_ids0_l2 = [], [], []

        kpts1_l1, kpts0from1_l1 = data['kpts1_l1'], data['kpts0from1_l1']
        b_ids1_l1, i_ids1_l1, j_ids1_l1 = (
            data['b_ids1_l1'],
            data['i_ids1_l1'],
            data['j_ids1_l1'],
        )
        kpts0_l1, kpts1from0_l1 = data['kpts0_l1'], data['kpts1from0_l1']
        b_ids0_l1, i_ids0_l1, j_ids0_l1 = (
            data['b_ids0_l1'],
            data['i_ids0_l1'],
            data['j_ids0_l1'],
        )

        for bs_id in range(self.bs):
            if (overlap_scores1[bs_id] > overlap_scores0[bs_id]
                ):  # >=  and overlap_scores1[bs_id] > 0:    # query:0  mask:1
                o_scale0 = scales[bs_id]
                bs_mask = b_ids1_l1 == bs_id
                bs_b_ids1_l1 = b_ids1_l1[bs_mask]
                bs_i_ids1_l1 = i_ids1_l1[bs_mask]
                bs_j_ids1_l1 = j_ids1_l1[bs_mask]
                bs_kpts1_l1 = kpts1_l1[bs_mask]
                bs_kpts0from1_l1 = kpts0from1_l1[bs_mask]

                if len(bs_j_ids1_l1) > 0:
                    # level2 kpts
                    bs_kpts1_l2 = bs_kpts1_l1 * self.scale_l1l2
                    bs_patch0_center_coord_l2 = bs_kpts0from1_l1 * self.scale_l1l2

                    # fine level feature
                    bs_kptsfeat1 = feat_map1_unfold[bs_b_ids1_l1,
                                                    bs_j_ids1_l1].flatten(
                                                        1, 2)  # [k, ww, c]
                    bs_kptsfeat0_from1 = feat_map0_unfold[
                        bs_b_ids1_l1, bs_i_ids1_l1].flatten(1,
                                                            2)  # [k, nwnw, c]
                    feat_d8 = [
                        feat_f0_z[zs_b_ids[indices], zs_ci_ids[indices]],
                        feat_f1_z[zs_b_ids[indices], zs_cj_ids[indices]]
                    ]
                    # if data["zs"].sum() > 0:
                    #     # feat_d8.append(feat_f0_z.flatten(1, 2))
                    #     # feat_d8.append(feat_f1_z.flatten(1, 2))
                    #     feat_d8.append(feat_f0_z[zs_b_ids[indices], zs_ci_ids[indices]])
                    #     feat_d8.append(feat_f1_z[zs_b_ids[indices], zs_ci_ids[indices]])

                    logger.info(f"feat_d8 shapes: {[feat.shape for feat in feat_d8]}, , {torch.cat([bs_kptsfeat0_from1, bs_kptsfeat1],0).shape}")

                    bs_feat_c = self.down_proj(torch.cat(feat_d8, dim=0))  # [2n, 2c->c]
                    
                    bs_feat_cf = self.merge_feat(
                        torch.cat(
                            [
                                torch.cat([bs_kptsfeat0_from1, bs_kptsfeat1],
                                          0),
                                repeat(
                                    bs_feat_c, 'n c -> n ww c', ww=self.W**2),
                            ],
                            -1,
                        ))
                    bs_kptsfeat0_from1, bs_kptsfeat1 = torch.chunk(bs_feat_cf,
                                                                   2,
                                                                   dim=0)
                    logger.info(f"1) bs_kptsfeat0_from1, bs_kptsfeat1: {bs_kptsfeat0_from1.shape}, {bs_kptsfeat1.shape}")
                    ###########################################################################
                    bs_kptsfeat1, bs_kptsfeat0_from1 = self.attention(
                        bs_kptsfeat1, bs_kptsfeat0_from1, flag=1)   

                    logger.info(f"2) bs_kptsfeat0_from1, bs_kptsfeat1: {bs_kptsfeat0_from1.shape}, {bs_kptsfeat1.shape}")

                    heatmap_zs = None
                    if data["zs"].sum() > 0:
                        heatmap_zs = self.compute_zeroshot_fine_loss(bs_kptsfeat0_from1, bs_kptsfeat1, radius, W, data)

                    (
                        bs_relative_kpts0from1_l2,
                        bs_kpts0from1_l2,
                        bs_std0,
                    ) = self.get_expected_locs(
                        bs_kptsfeat1,
                        bs_kptsfeat0_from1,
                        bs_patch0_center_coord_l2,
                        o_scale0,
                        flag=self.post_scale,
                        heatmap_zs=heatmap_zs
                    )

                    bs_i_ids1_l2 = (
                        bs_patch0_center_coord_l2[:, 0] +
                        bs_patch0_center_coord_l2[:, 1] * hw0_d2[1])
                    bs_j_ids1_l2 = bs_kpts1_l2[:,
                                               0] + bs_kpts1_l2[:,
                                                                1] * hw1_d2[1]

                    kpts1_l2.append(bs_kpts1_l2)
                    kpts0from1_l2.append(bs_kpts0from1_l2)
                    relative_kpts0from1_l2.append(bs_relative_kpts0from1_l2)
                    patch0_center_coord_l2.append(bs_patch0_center_coord_l2)
                    std0.append(bs_std0)
                    i_ids1_l2.append(bs_i_ids1_l2)
                    j_ids1_l2.append(bs_j_ids1_l2)

            else:  # elif overlap_scores0[bs_id] > 0:
                o_scale1 = scales[bs_id]
                bs_mask = b_ids0_l1 == bs_id
                bs_b_ids0_l1 = b_ids0_l1[bs_mask]
                bs_i_ids0_l1 = i_ids0_l1[bs_mask]
                bs_j_ids0_l1 = j_ids0_l1[bs_mask]
                bs_kpts0_l1 = kpts0_l1[bs_mask]
                bs_kpts1from0_l1 = kpts1from0_l1[bs_mask]

                if len(bs_j_ids0_l1) > 0:
                    bs_kpts0_l2 = bs_kpts0_l1 * self.scale_l1l2
                    bs_patch1_center_coord_l2 = bs_kpts1from0_l1 * self.scale_l1l2

                    # fine level featur0
                    bs_kptsfeat0 = feat_map0_unfold[bs_b_ids0_l1,
                                                    bs_j_ids0_l1].flatten(
                                                        1, 2)  # [k, ww, c]
                    bs_kptsfeat1_from0 = feat_map1_unfold[
                        bs_b_ids0_l1, bs_i_ids0_l1].flatten(1,
                                                            2)  # [k, nwnw, c]

                    feat_d8 = [
                        # feat_d8_0[bs_b_ids0_l1, bs_j_ids0_l1],
                        # feat_d8_1[bs_b_ids0_l1, bs_i_ids0_l1],
                        feat_f0_z[zs_b_ids[indices], zs_ci_ids[indices]],
                        feat_f1_z[zs_b_ids[indices], zs_cj_ids[indices]]
                    ]
                    # if data["zs"].sum() > 0:
                    #     # feat_d8.append(feat_f0_z.flatten(1, 2))
                    #     # feat_d8.append(feat_f1_z.flatten(1, 2))
                    #     feat_d8.append(feat_f0_z[zs_b_ids[indices], zs_ci_ids[indices]])
                    #     feat_d8.append(feat_f1_z[zs_b_ids[indices], zs_ci_ids[indices]])

                    logger.info(f"feat_d8 shapes: {[feat.shape for feat in feat_d8]}, {torch.cat([bs_kptsfeat0, bs_kptsfeat1_from0],0).shape}")

                    bs_feat_c = self.down_proj(torch.cat(feat_d8, dim=0))  # [2n, 2c->c]
                    
                    bs_feat_cf = self.merge_feat(
                        torch.cat(
                            [
                                torch.cat([bs_kptsfeat0, bs_kptsfeat1_from0],
                                          0),
                                repeat(
                                    bs_feat_c, 'n c -> n ww c', ww=self.W**2),
                            ],
                            -1,
                        ))
                    bs_kptsfeat0, bs_kptsfeat1_from0 = torch.chunk(bs_feat_cf,
                                                                   2,
                                                                   dim=0)
                    ######################################################################################
                    bs_kptsfeat0, bs_kptsfeat1_from0 = self.attention(
                        bs_kptsfeat0, bs_kptsfeat1_from0, flag=1)
                    
                    heatmap_zs = None
                    if data["zs"].sum() > 0:
                        heatmap_zs = self.compute_zeroshot_fine_loss(bs_kptsfeat1_from0, bs_kptsfeat0, radius, W, data)

                    (
                        bs_relative_kpts1from0_l2,
                        bs_kpts1from0_l2,
                        bs_std1,
                    ) = self.get_expected_locs(
                        bs_kptsfeat0,
                        bs_kptsfeat1_from0,
                        bs_patch1_center_coord_l2,
                        o_scale1,
                        flag=self.post_scale,
                        heatmap_zs=heatmap_zs
                    )

                    bs_i_ids0_l2 = (
                        bs_patch1_center_coord_l2[:, 0] +
                        bs_patch1_center_coord_l2[:, 1] * hw1_d2[1])
                    bs_j_ids0_l2 = bs_kpts0_l2[:,
                                               0] + bs_kpts0_l2[:,
                                                                1] * hw0_d2[1]

                    kpts0_l2.append(bs_kpts0_l2)
                    kpts1from0_l2.append(bs_kpts1from0_l2)
                    relative_kpts1from0_l2.append(bs_relative_kpts1from0_l2)
                    patch1_center_coord_l2.append(bs_patch1_center_coord_l2)
                    std1.append(bs_std1)
                    i_ids0_l2.append(bs_i_ids0_l2)
                    j_ids0_l2.append(bs_j_ids0_l2)

        pts0, pts1 = [], []
        data.update({
            'kpts1_l2':
            torch.cat(kpts1_l2, dim=0) if len(b_ids1_l1) else torch.empty(
                0, 2, device=self.device, dtype=torch.long),
            'kpts0from1_l2':
            torch.cat(kpts0from1_l2, dim=0)
            if len(b_ids1_l1) else torch.empty(0, 2, device=self.device),
            'relative_kpts0from1_l2':
            torch.cat(relative_kpts0from1_l2, dim=0)
            if len(b_ids1_l1) else torch.empty(0, 2, device=self.device),
            'patch0_center_coord_l2':
            torch.cat(patch0_center_coord_l2, dim=0)
            if len(b_ids1_l1) else torch.empty(0, 2, device=self.device),
            'std0':
            torch.cat(std0, dim=0) if len(b_ids1_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'b_ids1_l2':
            b_ids1_l1.clone() if len(b_ids1_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'i_ids1_l2':
            torch.cat(i_ids1_l2, dim=0) if len(b_ids1_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'j_ids1_l2':
            torch.cat(j_ids1_l2, dim=0) if len(b_ids1_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
        })
        if len(b_ids1_l1):
            pts0.append(data['kpts0from1_l2'])
            pts1.append(data['kpts1_l2'])

        data.update({
            'kpts0_l2':
            torch.cat(kpts0_l2, dim=0) if len(b_ids0_l1) else torch.empty(
                0, 2, device=self.device, dtype=torch.long),
            'kpts1from0_l2':
            torch.cat(kpts1from0_l2, dim=0)
            if len(b_ids0_l1) else torch.empty(0, 2, device=self.device),
            'relative_kpts1from0_l2':
            torch.cat(relative_kpts1from0_l2, dim=0)
            if len(b_ids0_l1) else torch.empty(0, 2, device=self.device),
            'patch1_center_coord_l2':
            torch.cat(patch1_center_coord_l2, dim=0)
            if len(b_ids0_l1) else torch.empty(0, 2, device=self.device),
            'std1':
            torch.cat(std1, dim=0) if len(b_ids0_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'b_ids0_l2':
            b_ids0_l1.clone() if len(b_ids0_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'i_ids0_l2':
            torch.cat(i_ids0_l2, dim=0) if len(b_ids0_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
            'j_ids0_l2':
            torch.cat(j_ids0_l2, dim=0) if len(b_ids0_l1) else torch.empty(
                0, device=self.device, dtype=torch.long),
        })
        if len(b_ids0_l1):
            pts1.append(data['kpts1from0_l2'])
            pts0.append(data['kpts0_l2'])

        if len(b_ids1_l1) != 0 or len(b_ids0_l1) != 0:
            pts1 = torch.cat(pts1, dim=0)
            pts0 = torch.cat(pts0, dim=0)

        else:
            pts1 = torch.empty(0, 2, device=self.device)
            pts0 = torch.empty(0, 2, device=self.device)

        if len(m_bids) != 0:
            scale1_l2 = (self.scale_l2 * data['scale1'][m_bids]
                         if 'scale1' in data else self.scale_l2)
            scale0_l2 = (self.scale_l2 * data['scale0'][m_bids]
                         if 'scale0' in data else self.scale_l2)
        else:
            scale1_l2 = scale0_l2 = 0.0

        data.update({
            'mkpts0_f':
            pts0 * scale0_l2
            if len(m_bids) else torch.empty(0, 2, device=self.device),
            'mkpts1_f':
            pts1 * scale1_l2
            if len(m_bids) else torch.empty(0, 2, device=self.device),
        })

    def viz(
        self,
        data,
        kpts0_l2,
        kpts1_l2,
        b_ids0_l2,
        b_ids1_l2,
        patch0_center_coord_l2,
        patch1_center_coord_l2,
        patch_size_l0l2,
    ):
        for b_id in range(data['image0'].size(0)):
            img0 = ((data['image0'][b_id].cpu().numpy() * 255).round().astype(
                np.float32).transpose(1, 2, 0))
            img1 = ((data['image1'][b_id].cpu().numpy() * 255).round().astype(
                np.float32).transpose(1, 2, 0))
            kp1 = kpts1_l2[b_ids1_l2 ==
                           b_id].detach().cpu().numpy() * self.scale_l2
            p0_c = (patch0_center_coord_l2[
                b_ids1_l2 == b_id].detach().cpu().numpy() * self.scale_l2)

            kp0 = kpts0_l2[b_ids0_l2 ==
                           b_id].detach().cpu().numpy() * self.scale_l2
            p1_c = (patch1_center_coord_l2[
                b_ids0_l2 == b_id].detach().cpu().numpy() * self.scale_l2)

            path = './viz/gt_fine/gt_{}_{}.jpg'.format(b_id, self.count)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            make_matching_fine(img0, img1, p0_c, kp1, p1_c, kp0,
                               patch_size_l0l2, path)
