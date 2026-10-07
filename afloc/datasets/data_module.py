"""LightningDataModule for MIMIC-CXR pretraining."""
import functools

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import pretraining_dataset
from .. import builder


class PretrainingDataModule(pl.LightningDataModule):
    """
    Builds the train / validate / test loaders. When a knowledge injector is
    given it is bound into the collate function, so every batch carries the
    RadGraph entity graphs of its reports under `graph_data`.
    """

    def __init__(self, cfg, knowledge_injector=None):
        super().__init__()
        self.cfg = cfg
        self.dataset = pretraining_dataset.MultimodalPretrainingDataset
        self.collate_fn = functools.partial(
            pretraining_dataset.multimodal_collate_fn,
            knowledge_injector=knowledge_injector,
        )
        if knowledge_injector is not None:
            print(f"[DataModule] knowledge injector: {type(knowledge_injector).__name__}")
        else:
            print("[DataModule] knowledge injector: disabled")

        # Under DDP every process loads its own batch, so the per-GPU size has
        # to be given explicitly; falling back to the global batch size would
        # silently multiply the effective batch by the number of GPUs.
        accelerator = getattr(cfg.lightning.trainer, "accelerator", None)
        if accelerator == "ddp":
            if "per_gpu_batchsize" not in self.cfg.train:
                raise ValueError("accelerator='ddp' needs cfg.train.per_gpu_batchsize")
            self.batch_size = self.cfg.train.per_gpu_batchsize
        else:
            self.batch_size = self.cfg.train.batch_size
        print(f"[DataModule] accelerator={accelerator!r} batch_size={self.batch_size}")

        self.pin_memory = False

    def _loader(self, split, transform_split, shuffle, drop_last):
        transform = builder.build_transformation(self.cfg, transform_split)
        dataset = self.dataset(self.cfg, split=split, transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self._loader("train", "train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader("validate", "test", shuffle=False, drop_last=True)

    def test_dataloader(self):
        return self._loader("test", "test", shuffle=False, drop_last=False)
