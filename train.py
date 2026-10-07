"""
Train KAF-Ground on MIMIC-CXR image-report pairs.

    python train.py -c configs/kaf_ground.yaml --train --gpus 0,1

Checkpoints are written to `<output_root>/checkpoints/<experiment>/<timestamp>/`
and logs/config snapshots to `<output_root>/output/...`. Set `output_root` in the
config or override it with `--output_root`.
"""
import argparse
import datetime
import os

import numpy as np
import torch
from dateutil import tz
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

import afloc
from afloc.lightning.pretrain_model import ProgressiveUnfreezeCallback

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to a yaml config")
    parser.add_argument("--train", action="store_true", help="run training")
    parser.add_argument("--test", action="store_true", help="run the Lightning test loop")
    parser.add_argument("--train_pct", type=float, default=1.0,
                        help="fraction of the training set to use")
    parser.add_argument("--splits", type=int, default=1,
                        help="number of seeds (taken from cfg.train.seeds) to train")
    parser.add_argument("--trial_name", type=str, default=None)
    parser.add_argument("-l", "--load_ckpt", type=str, default=None,
                        help="override cfg.train.load_ckpt (AFLoc initialisation)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="override cfg.output_root")
    # --gpus, --max_epochs, --resume_from_checkpoint, ... come from Lightning.
    parser = Trainer.add_argparse_args(parser)
    return parser


class SaveConfigCallback(Callback):
    """Create the run directories and keep a copy of the resolved config."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_init_end(self, trainer):
        # exist_ok: under DDP every process runs this.
        os.makedirs(self.cfg.lightning.logger.save_dir, exist_ok=True)
        os.makedirs(self.cfg.lightning.checkpoint_callback.dirpath, exist_ok=True)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        OmegaConf.save(config=self.cfg,
                       f=os.path.join(self.cfg.output_dir, "config.yaml"))


class GradNormCallback(Callback):
    """
    Log gradient norms of the graph branch (GAT, node projection, fusion gate)
    every 100 steps. A norm that stays at zero means L_SG is not reaching the
    graph encoder.
    """

    _WATCH = ("gat", "graph", "entity", "kg", "node_proj", "fusion_proj")

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_rank != 0 or trainer.global_step % 100 != 0:
            return
        if trainer.logger is None:
            return
        for name, param in pl_module.named_parameters():
            if param.grad is None or not any(x in name.lower() for x in self._WATCH):
                continue
            norm = param.grad.norm().item()
            trainer.logger.log_metrics({f"grad_norm/{name}": norm}, step=trainer.global_step)
            if norm < 1e-7:
                print(f"[dead grad] {name}: {norm:.2e} at step {trainer.global_step}")


def main(cfg, args):
    dm = afloc.builder.build_data_module(cfg)
    model = afloc.builder.build_lightning_model(cfg, dm)

    callbacks = [SaveConfigCallback(cfg), GradNormCallback()]
    checkpoint_callback = None
    if "checkpoint_callback" in cfg.lightning:
        checkpoint_callback = ModelCheckpoint(**cfg.lightning.checkpoint_callback)
        callbacks.append(checkpoint_callback)
    callbacks.append(ProgressiveUnfreezeCallback(
        unfreeze_epoch=getattr(cfg.model.vision, "cnn_unfreeze_epoch", 5),
        lr_cnn=getattr(cfg.model.vision, "cnn_lr", 2e-6),
    ))

    logger = None
    if "logger" in cfg.lightning:
        logger_type = cfg.lightning.logger.pop("logger_type")
        cfg.lightning.logger.name = f"{cfg.experiment_name}_{cfg.extension}"
        logger = getattr(pl_loggers, logger_type)(**cfg.lightning.logger)
        cfg.lightning.logger.logger_type = logger_type

    cfg.lightning.trainer.val_check_interval = args.val_check_interval
    cfg.lightning.trainer.auto_lr_find = args.auto_lr_find
    trainer_args = argparse.Namespace(**cfg.lightning.trainer)

    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"checkpoint not found: {resume_ckpt}")
        # configure_optimizers reads this to restore the optimizer state
        # (self.trainer is not yet attached when it runs).
        os.environ["PL_RESUME_CHECKPOINT"] = resume_ckpt

    trainer = Trainer.from_argparse_args(
        args=trainer_args,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=resume_ckpt,
    )
    model.trainer = trainer

    if trainer_args.auto_lr_find is not False:
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        model.lr = lr_finder.suggestion()
        print(f"learning rate set to {model.lr}")

    if args.train:
        trainer.fit(model, dm)
    if args.test:
        trainer.test(model=model, datamodule=dm)

    if checkpoint_callback is not None:
        checkpoint_callback.to_yaml(filepath=os.path.join(
            cfg.lightning.checkpoint_callback.dirpath, "best_ckpts.yaml"))


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.data.frac = args.train_pct
    cfg.trial_name = args.trial_name
    if cfg.trial_name is not None:
        cfg.experiment_name = f"{cfg.experiment_name}_{cfg.trial_name}"
    cfg.experiment_name = f"{cfg.experiment_name}_{args.train_pct}"
    if args.load_ckpt is not None:
        cfg.train.load_ckpt = args.load_ckpt
    if args.output_root is not None:
        cfg.output_root = args.output_root
    root = cfg.get("output_root", "./runs")

    for split in np.arange(args.splits):
        seed = list(cfg.train.seeds)[split]
        seed_everything(seed)
        print("random seed:", seed)

        timestamp = datetime.datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
        cfg.extension = str(seed) if args.splits != 1 else timestamp
        cfg.output_dir = f"{root}/output/{cfg.experiment_name}/{cfg.extension}"
        cfg.lightning.checkpoint_callback.dirpath = \
            f"{root}/checkpoints/{cfg.experiment_name}/{cfg.extension}"

        main(cfg, args)
