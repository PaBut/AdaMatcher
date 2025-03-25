import argparse
import math
import os
import pdb
import pprint
from distutils.util import strtobool
from pathlib import Path

from pytorch_lightning.strategies import SingleDeviceStrategy

import numpy as np
import pytorch_lightning as pl
from loguru import logger as loguru_logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import torch

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_adamatcher import PL_AdaMatcher
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a custom parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint path, helpful for using a pre-trained coarse-only AdaMatcher",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )
    parser.add_argument(
        "--parallel_load_data",
        action="store_true",
        help="load datasets in with multiple processes.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of GPUs to use (can also be a list like '0,1' but we simplify to int here)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "tpu", "ipu", "hpu", "mps", "ddp", "ddp_spawn", "dp"],
        help="accelerator type for training",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation every n epochs",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1,
        help="log every n steps",
    )
    parser.add_argument(
        "--flush_logs_every_n_steps",
        type=int,
        default=1,
        help="flush logs every n steps",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="limit the number of validation batches (float for percentage, int for exact number)",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=10,
        help="number of sanity validation steps to run before training",
    )
    parser.add_argument(
        "--benchmark",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="enable cudnn.benchmark for faster training with fixed input sizes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="maximum number of epochs to train",
    )

    # parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    # pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    # TensorBoard Logger
    # logger = TensorBoardLogger(save_dir='./OUTPUT/densematching/tb_logs', name=args.exp_name, default_hp_metric=False)
    logger = TensorBoardLogger(
        save_dir="./OUTPUT/densematching",
        name=args.exp_name,
        default_hp_metric=False,
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    last_ckpt_path = str(
        Path(logger.log_dir[:-1] + str(int(logger.log_dir[-1]) - 1))
        / "checkpoints/last.ckpt"
    )
    if os.path.exists(last_ckpt_path):
        args.ckpt_path = last_ckpt_path
    if args.ckpt_path is None:
        pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    else:
        pl.seed_everything(np.random.randint(2**31))  # reproducibility

    # scale lr and warmup-step automatically
    # pdb.set_trace()
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_AdaMatcher(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"AdaMatcher LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"AdaMatcher DataModule initialized!")

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(
        monitor="auc@10",
        verbose=True,
        save_top_k=5,
        mode="max",
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, TQDMProgressBar(leave=True)]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    trainer_params = {
        "accelerator": args.accelerator,
        "devices": 1,  # Maps 'gpus' to 'devices'
        "num_nodes": args.num_nodes,
        "max_epochs": args.max_epochs,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "log_every_n_steps": args.log_every_n_steps,
        # "flush_logs_every_n_steps": args.flush_logs_every_n_steps,
        "limit_val_batches": args.limit_val_batches,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "benchmark": args.benchmark,
    }

    # Lightning Trainer
    trainer = pl.Trainer(
        # strategy=DDPStrategy(
        #     find_unused_parameters=False,  # True,
        #     # num_nodes=args.num_nodes,
        #     # strategy="ddp_sharded",
        #     # sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        # ),
        # strategy=DDPPlugin(find_unused_parameters=False)
        strategy=SingleDeviceStrategy(
            device=torch.device("cuda:0"),
        ),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        # replace_sampler_ddp=False,  # use custom sampler
        # reload_dataloaders_every_n_epoch=0,  # avoid repeated samples!
        enable_model_summary=True,
        # resume_from_checkpoint=args.ckpt_path,
        profiler=profiler,
        enable_progress_bar=True,
        **trainer_params,
        # precision=16,
        # auto_lr_find=True
    )

    # LightningCLI(model_class=model)

    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    # loguru_logger.info(f"{len(trainer.strategy.optimizers[0].param_groups)}")

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
