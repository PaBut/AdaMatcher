import os
import pdb
import pprint
import subprocess

# import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from einops.einops import rearrange
from loguru import logger

from match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder, Matcher
from src.adamatcher import AdaMatcher
from src.losses.ada_loss import AdaMatcherLoss
from src.utils.metrics import aggregate_metrics  # compute_coarse_error,
from src.utils.metrics import compute_pose_errors, compute_symmetrical_epipolar_errors
from src.utils.misc import flattenList, lower_config
from src.utils.profiler import PassThroughProfiler

# from matplotlib import pyplot as plt

class PL_Tester(pl.LightningModule):
    def __init__(self, config, matcher_type: Literal['sift', 'adamatcher', 'loftr'], pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        self.profiler = profiler or PassThroughProfiler()

        # Matcher: AdaMatcher
        if matcher_type == "sift":
            self.matcher = FeatureDetectorMatchFinder(FeatureDetector.SIFT)
        elif matcher_type == "loftr":
            self.matcher = LoFTRMatchFinder()
        elif matcher_type == "adamatcher":
            self.matcher = AdaMatcherMatchFinder()
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")

        torch.set_float32_matmul_precision('medium')

        # Testing
        self.dump_dir = dump_dir
        self.metric_time = 0.0

        self.test_step_outputs = []
        
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(
                batch
            )  # compute epi_errs for each match
            compute_pose_errors(
                batch, self.config
            )  # compute R_errs, t_errs, pose_errs for each pair
            # compute_coarse_error(batch)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["m_bids"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def test_step(self, batch, batch_idx):
        # with self.profiler.profile("AdaMatcher"):
        k1, k2 = self.matcher.find_matches(batch["image0"], batch["image1"])
        batch["mkpts0_f"] = k1
        batch["mkpts1_f"] = k2
        batch["m_bids"] = [0]

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        # self.metric_time += time.monotonic() - t1

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:

                keys_to_save = {"mkpts0_f", "mkpts1_f", "scores", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        if "classification" not in key:
                            item[key] = batch[key][mask].cpu().numpy()
                        else:
                            item[key] = batch[key][b_id].cpu().numpy()
                    for key in [
                        "R_errs",
                        "t_errs",
                        "inliers",
                    ]:  # 'fp_scores', 'miss_scores']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        outputs = self.test_step_outputs
        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(([_me[k] for _me in _metrics]))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = _dumps  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "Ada_pred_eval", dumps)

        self.test_step_outputs.clear()
