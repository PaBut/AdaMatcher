import pdb
from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils as utils
from numpy.linalg import inv

from src.utils.dataset import (read_scannet_color, read_scannet_depth,
                               read_scannet_gray, read_scannet_intrinsic,
                               read_scannet_pose)


class ScanNetDataset(utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        npz_path,
        intrinsic_path,
        mode='train',
        min_overlap_score=0.4,
        augment_fn=None,
        pose_dir=None,
        **kwargs,
    ):
        """Manage one scene of ScanNet Dataset.

        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = dict(np.load(intrinsic_path))

        # filter images with problems
        filter_imgs = [[658, 0, 155]]
        filter_mask = np.array([False] * len(self.data_names), dtype=np.bool)
        for filter_img in filter_imgs:
            mask0 = self.data_names[:, 0] == filter_img[0]
            mask1 = self.data_names[:, 1] == filter_img[1]
            mask2 = self.data_names[:, 2] == filter_img[2]
            mask3 = self.data_names[:, 3] == filter_img[2]
            filter_mask |= mask0 & mask1 & (mask2 | mask3)
        self.data_names = self.data_names[~filter_mask]

        # for training AdaMatcher
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.pose_dir, scene_name, 'pose', f'{name}.txt')
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)

        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.root_dir, scene_name, 'color',
                             f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color',
                             f'{stem_name_1}.jpg')

        # TODO: Support augmentation & handle seeds for each worker correctly.
        # if 'rots' in self.scene_info:
        #     rot0, rot1 = self.scene_info['rots'][idx]
        #     print('True')
        # else:
        rot0, rot1 = 0, 0
        image0, mask0, scale0, scale_wh0 = read_scannet_color(img_name0,
                                                              resize=(640,
                                                                      480),
                                                              augment_fn=None,
                                                              rotation=rot0)
        image1, mask1, scale1, scale_wh1 = read_scannet_color(img_name1,
                                                              resize=(640,
                                                                      480),
                                                              augment_fn=None,
                                                              rotation=rot1)
        # scale_wh0 = torch.tensor([640, 480], dtype=torch.float)
        # scale_wh1 = torch.tensor([640, 480], dtype=torch.float)
        # image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read the depthmap which is stored as (480, 640)
        if self.mode in ['train', 'val']:
            depth0 = read_scannet_depth(
                osp.join(self.root_dir, scene_name, 'depth',
                         f'{stem_name_0}.png'))
            depth1 = read_scannet_depth(
                osp.join(self.root_dir, scene_name, 'depth',
                         f'{stem_name_1}.png'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(),
                                 dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T_0to1 = torch.tensor(
            self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
            dtype=torch.float32,
        )
        T_1to0 = T_0to1.inverse()

        data = {
            'image0':
            image0,  # (1, h, w)
            'depth0':
            depth0,  # (h, w)
            'image1':
            image1,
            'depth1':
            depth1,
            'T_0to1':
            T_0to1,  # (4, 4)
            'T_1to0':
            T_1to0,
            'scale0':
            scale0,  # [scale_w, scale_h]
            'scale1':
            scale1,
            'scale_wh0':
            scale_wh0,
            'scale_wh1':
            scale_wh1,
            'K0':
            K_0,  # (3, 3)
            'K1':
            K_1,
            'dataset_name':
            'ScanNet',
            'scene_id':
            scene_name,
            'pair_id':
            idx,
            'pair_names': (
                osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                osp.join(scene_name, 'color', f'{stem_name_1}.jpg'),
            ),

            'gt': True,
            'zs': False,
        }

        if mask0 is not None:
            [mask0_d8, mask1_d8] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=1 / 8,
                mode='nearest',
                recompute_scale_factor=False,
            )[0].bool()
            [mask0_l0, mask1_l0] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=1 / 16,
                mode='nearest',
                recompute_scale_factor=False,
            )[0].bool()
            data.update({
                'mask0_l0': mask0_l0,
                'mask1_l0': mask1_l0,
                'mask0_d8': mask0_d8,
                'mask1_d8': mask1_d8,
            })
        return data
