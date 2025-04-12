from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from kornia.geometry.epipolar import numeric
from cv2 import Mat
import torch
from enums import EnvironmentType
from tools.AdaMatcherUtils.adamatcher.utils.cvpr_ds_config import lower_config
from tools.AdaMatcherUtils.adamatcher.adamatcher import AdaMatcher
from tools.AdaMatcherUtils.config.default import get_cfg_defaults
from tools.LoFTR import LoFTR, default_cfg

import cv2
import numpy as np

class FeatureDetector(Enum):
    ORB = 0
    SIFT = 1
    BRISK = 2
    AKAZE = 3

class Matcher(Enum):
    FLANN = 0
    BF = 1

class MatchFinder(ABC):
    @abstractmethod
    def find_matches(self, img1: Mat, img2: Mat) -> tuple[list[int], list[int]]:
        pass

class FeatureDetectorMatchFinder(MatchFinder):
    MIN_MATCH_COUNT = 4
    def __get_flann_index_params(self, detector_type: FeatureDetector) -> dict:
        # TODO: study the params
        if detector_type == FeatureDetector.SIFT:
            return dict(algorithm=1, trees=5)
        else:
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)

    def __get_bf_norm_type(self, detector_type: FeatureDetector):
        if detector_type == FeatureDetector.SIFT:
            return cv2.NORM_L2 # Euclidean distance
        if detector_type == FeatureDetector.ORB:
            return cv2.NORM_HAMMING2
        else:
            return cv2.NORM_HAMMING
    
    def __init__(self, detector_type: FeatureDetector, matcher_type: Matcher = Matcher.BF):
        if detector_type == FeatureDetector.SIFT:
            self.detector = cv2.SIFT.create()
        elif detector_type == FeatureDetector.AKAZE:
            self.detector = cv2.AKAZE.create()
        elif detector_type == FeatureDetector.BRISK:
            self.detector = cv2.BRISK.create()
        elif detector_type == FeatureDetector.ORB:
            self.detector = cv2.ORB.create()
        else:
            raise Exception('Not supported')

        if matcher_type == Matcher.FLANN:
            search_params = dict(checks=50)
            index_params = self.__get_flann_index_params(detector_type)

            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == Matcher.BF:
            self.matcher = cv2.BFMatcher()
        else:
            raise Exception('Not supported')
    def find_matches(self, img1, img2):
        keypoints1, descriptors1 = self.detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(img2, None)

        matches = self.matcher.knnMatch(descriptors2, descriptors1, k = 2)

        good_matches = []

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # print('Good matches:', len(good_matches))
        # if len(good_matches) > self.MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches])

        return src_pts, dst_pts
        # else:
        #     return np.empty((1, 2), dtype=float), np.empty((1, 2), dtype=float)
        

class LoFTRMatchFinder(MatchFinder):
    INDOOR_WEIGHTS_PATH = r'.\tools\LoFTR\weights\indoor_ds_new.ckpt'
    OUTDOOR_WEIGHTS_PATH = r'.\tools\LoFTR\weights\outdoor_ds.ckpt'
    FIXED_WIDTH = 640
    def __init__(self, loftr_type: EnvironmentType):
        _default_cfg = deepcopy(default_cfg)
        if loftr_type == EnvironmentType.Indoor:
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            _default_cfg['match_coarse']['match_type'] = 'dual_softmax'
            weights_path = self.INDOOR_WEIGHTS_PATH
        elif loftr_type == EnvironmentType.Outdoor:
            weights_path = self.OUTDOOR_WEIGHTS_PATH
            # _default_cfg['match_coarse']['match_type'] = 'sinkhorn'
            # _default_cfg['match_coarse']['sparse_spvs'] = False
        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load(weights_path)['state_dict'])
        self.matcher = matcher.eval().cuda()
    def find_matches(self, img0, img1):
        size_difference = img0.shape[1] / self.FIXED_WIDTH
        new_height = int(img0.shape[0] / size_difference)
        input1 = cv2.resize(np.copy(img0), (self.FIXED_WIDTH, new_height))
        input2 = cv2.resize(np.copy(img1), (self.FIXED_WIDTH, new_height))
        img1_torch = torch.from_numpy(cv2.cvtColor(input1, cv2.COLOR_RGB2GRAY))[None][None].cuda() / 255.
        img2_torch = torch.from_numpy(cv2.cvtColor(input2, cv2.COLOR_RGB2GRAY))[None][None].cuda() / 255.
        batch = {'image0': img1_torch, 'image1': img2_torch}
        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()

            return mkpts0 * size_difference, mkpts1 * size_difference
        

class AdaMatcherMatchFinder(MatchFinder):
    FIXED_WIDTH = 640
    FIXED_DIVISION = 32
    WEIGHTS_PATH = r'.\tools\AdaMatcherUtils\weights\adamatcher.ckpt'
    def __init__(self):
        config = get_cfg_defaults()
        # config.merge_from_file(main_config_path)
        _config = lower_config(config)
        
        self.model = AdaMatcher(
            config = _config["adamatcher"],
            training=False
        )  
        state_dict = torch.load(self.WEIGHTS_PATH)["state_dict"]
        
        new_state_dict = {}
        prefix = 'matcher.'
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.eval().cuda()

    def find_matches(self, img1, img2):
        h, w = img1.shape[:2]
        w_difference = w / self.FIXED_WIDTH
        p_height = int(h / w_difference)
        modulo = p_height % self.FIXED_DIVISION

        if modulo > self.FIXED_DIVISION / 2:
            modulo -= self.FIXED_DIVISION

        new_height = p_height - modulo
        h_difference = h / new_height

        input1 = cv2.resize(np.copy(img1), (self.FIXED_WIDTH, new_height))
        input2 = cv2.resize(np.copy(img2), (self.FIXED_WIDTH, new_height))
        img1_torch = torch.from_numpy(cv2.cvtColor(input1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))[None].float().cuda() / 255.
        img2_torch = torch.from_numpy(cv2.cvtColor(input2, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))[None].float().cuda() / 255.
        batch = {'image0': img1_torch, 'image1': img2_torch}
        with torch.no_grad():
            self.model(batch)

            # print(batch.keys())

            pts0 = batch["mkpts0_f"].cpu().numpy()
            pts1 = batch["mkpts1_f"].cpu().numpy()

            return pts0 * np.array([w_difference, h_difference]), pts1 * np.array([w_difference, h_difference])

            # keys_to_save = {"mkpts0_f", "mkpts1_f", "scores"}
            # # pair_names = list(zip(*batch["pair_names"]))
            # bs = batch["image0"].shape[0]
            # dumps = []
            # for b_id in range(bs):
            #     item = {}
            #     mask = batch["m_bids"] == b_id
            #     # item["pair_names"] = pair_names[b_id]
            #     for key in keys_to_save:
            #         if "classification" not in key:
            #             item[key] = batch[key][mask].cpu().numpy()
            #         else:
            #             item[key] = batch[key][b_id].cpu().numpy()
            #     dumps.append(item)


            # return dumps
