"""
Factories for the model, data module, Lightning module, optimiser and
transforms.

Training builds things in this order (see train.py):
    dm    = build_data_module(cfg)          # also builds the RadGraph injector
    model = build_lightning_model(cfg, dm)  # reuses the injector from dm
so the RadGraph span table and ClinicalBERT are only loaded once.
"""

import torch
import torchvision.transforms as transforms
from . import models
from . import lightning
from . import datasets
from typing import Union, Optional


def load_model(
    ckpt_path: str = "",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    from_pretrained: bool = False,
    hf_name_override: Optional[str] = None,
    bert_type_override: Optional[str] = None,
):
    """
    Load AFLoc / KAF-Ground from a Lightning checkpoint.

    The checkpoint stores the full training config. If it names local model
    snapshots that do not exist on this machine, pass the HuggingFace ids via
    hf_name_override / bert_type_override.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg       = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    if hf_name_override is not None:
        cfg.model.vision.hf_name = hf_name_override
    if bert_type_override is not None:
        cfg.model.text.bert_type = bert_type_override

    # strip Lightning's "afloc." prefix
    ckpt_dict = {k.split("afloc.")[-1]: v for k, v in ckpt_dict.items()}

    model = build_model(cfg).to(device)
    if from_pretrained:
        return model
    model.load_state_dict(ckpt_dict, strict=False)
    return model


def build_data_module(cfg):
    """
    Build the LightningDataModule. With knowledge_injection.enabled, also build
    ClinicalBERT, the tokenizer and the RadGraphInjector, and keep them on the
    data module for build_lightning_model.
    """
    ki_cfg            = getattr(cfg, "knowledge_injection", None)
    injection_enabled = ki_cfg is not None and ki_cfg.get("enabled", False)

    if cfg.phase.lower() == "pretrain":
        data_module_cls = datasets.DATA_MODULES["pretrain"]
    else:
        data_module_cls = datasets.DATA_MODULES[cfg.data.dataset.lower()]

    if not injection_enabled:
        return data_module_cls(cfg)

    print("[builder] building the RadGraph knowledge injector")
    from transformers import AutoTokenizer

    shared_text_encoder = build_text_model(cfg)
    shared_tokenizer    = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
    knowledge_injector  = build_knowledge_injector(cfg, shared_text_encoder, shared_tokenizer)

    dm = data_module_cls(cfg, knowledge_injector=knowledge_injector)
    dm._shared_text_encoder = shared_text_encoder
    dm._shared_tokenizer    = shared_tokenizer
    dm._knowledge_injector  = knowledge_injector
    return dm


def build_lightning_model(cfg, dm):
    """
    Build the LightningModule from the *current* config.

    The model is built here and handed to PretrainModel through the class
    attribute _injected_afloc, so PretrainModel does not rebuild it from the
    config stored in cfg.train.load_ckpt (the ResNet-50 AFLoc checkpoint, which
    only initialises the text side).
    """
    knowledge_injector  = getattr(dm, "_knowledge_injector",  None)
    shared_text_encoder = getattr(dm, "_shared_text_encoder", None)

    if knowledge_injector is not None:
        try:
            cfg._knowledge_injector  = knowledge_injector
            cfg._shared_text_encoder = shared_text_encoder
        except Exception:
            pass

    model = build_model(
        cfg,
        knowledge_injector=knowledge_injector,
        shared_text_encoder=shared_text_encoder,
    )

    module_cls = lightning.LIGHTNING_MODULES[cfg.phase.lower()]
    module_cls._injected_afloc = model     # read in PretrainModel.__init__
    module = module_cls(cfg)
    module.dm = dm
    return module


def build_model_from_ckpt(ckpt_path: str, cfg=None):
    """
    Build AFLoc and load a checkpoint with strict=False.

    With cfg, the architecture comes from cfg and only matching keys are loaded
    (e.g. the text side of the original AFLoc checkpoint). Without cfg, the
    config stored in the checkpoint is used.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_dict = {k.split("afloc.")[-1]: v for k, v in ckpt["state_dict"].items()}

    if cfg is not None:
        model = build_model(cfg)
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(ckpt_dict.keys())
        print(f"[builder] loading {len(model_keys & ckpt_keys)} keys, "
              f"{len(ckpt_keys - model_keys)} only in checkpoint, "
              f"{len(model_keys - ckpt_keys)} newly initialised")
    else:
        model = build_model(ckpt["hyper_parameters"])

    model.load_state_dict(ckpt_dict, strict=False)
    return model


def build_img_model(cfg):
    """RAD-DINO for model_name "rad_dino", otherwise the original AFLoc encoder."""
    model_name = getattr(cfg.model.vision, "model_name", "resnet_50").lower()

    if model_name in ("rad_dino", "rad-dino", "raddino"):
        from .models.rad_dino_encoder import RadDinoImageEncoder
        return RadDinoImageEncoder(cfg)

    image_model = models.IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)


def build_text_model(cfg):
    """ClinicalBERT text encoder."""
    return models.text_model.BertEncoder(cfg)


def build_optimizer(cfg, lr, model):
    """Optimiser over parameters with requires_grad=True."""
    params = [p for p in model.parameters() if p.requires_grad]

    if cfg.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params, lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
        )


def build_scheduler(cfg, optimizer, dm=None):
    """Learning-rate scheduler."""
    name = cfg.train.scheduler.name

    if name == "warmup":
        def lambda_lr(epoch):
            if epoch <= 3:  return 0.001 + epoch * 0.003
            if epoch >= 22: return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train.scheduler.step_size,
            gamma=cfg.train.scheduler.gamma,
        )
    else:
        scheduler = None

    return {
        "scheduler": scheduler,
        "monitor":   cfg.train.scheduler.monitor,
        "interval":  cfg.train.scheduler.interval,
        "frequency": cfg.train.scheduler.frequency,
    }


def build_transformation(cfg, split):
    """
    Image transforms. "rad_dino" normalisation reads mean/std from the
    RAD-DINO image processor; ImageNet statistics silently hurt grounding.
    """
    t = []
    if split == "train":
        if cfg.transforms.center_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.center_crop.crop_size))
        elif cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())

    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif cfg.transforms.norm == "rad_dino":
            from transformers import AutoImageProcessor
            hf_name = getattr(cfg.model.vision, "hf_name", "microsoft/rad-dino")
            proc    = AutoImageProcessor.from_pretrained(hf_name)
            t.append(transforms.Normalize(tuple(proc.image_mean), tuple(proc.image_std)))
        else:
            raise NotImplementedError(
                f"normalisation '{cfg.transforms.norm}' not implemented "
                f"(supported: 'imagenet', 'half', 'rad_dino')"
            )

    return transforms.Compose(t)


def build_knowledge_injector(cfg, text_encoder, tokenizer):
    """
    Build the injector named by cfg.knowledge_injection.source (only
    "radgraph"). The RadGraph json is read from cfg.data.radgraph_path.
    Returns None when injection is disabled.
    """
    ki_cfg = getattr(cfg, "knowledge_injection", None)
    if ki_cfg is None or not ki_cfg.get("enabled", False):
        return None

    source = ki_cfg.get("source", "radgraph").lower()
    if source != "radgraph":
        raise ValueError(f"unknown knowledge_injection.source '{source}' (supported: radgraph)")

    from .knowledge.radgraph_injector import RadGraphInjector

    radgraph_path = getattr(cfg.data, "radgraph_path", None)
    if radgraph_path is None:
        raise ValueError("knowledge_injection.enabled=true needs cfg.data.radgraph_path")

    return RadGraphInjector(
        radgraph_path=radgraph_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        max_entities=ki_cfg.get("max_nodes", 64),
    )


def build_dual_stream_encoder(cfg, knowledge_injector):
    """GAT graph encoder on top of the injector's span table (None without one)."""
    if knowledge_injector is None:
        return None

    from .models.dual_stream_text import DualStreamTextEncoder
    ki_cfg = getattr(cfg, "knowledge_injection", {})

    return DualStreamTextEncoder(
        emb_matrix_cpu_fp16=knowledge_injector.emb_matrix,
        max_nodes=ki_cfg.get("max_nodes", 64),
        hidden_dim=cfg.model.text.embedding_dim,
        gat_heads=ki_cfg.get("gat_heads", 4),
        fusion_init=ki_cfg.get("fusion_init", "zero"),
    )


def build_model(cfg, knowledge_injector=None, shared_text_encoder=None):
    """
    Build AFLoc, with the graph encoder when an injector is given. No weights
    are loaded here; see PretrainModel._load_pretrained_weights and
    build_model_from_ckpt.
    """
    from .models.afloc_model import AFLoc

    dual_stream_encoder = None
    if knowledge_injector is not None:
        dual_stream_encoder = build_dual_stream_encoder(cfg, knowledge_injector)

    model = AFLoc(cfg, dual_stream_encoder=dual_stream_encoder)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = getattr(getattr(cfg.model, "vision", None), "model_name", "unknown")
    print(f"[builder] AFLoc({model_name}), graph branch: {knowledge_injector is not None}, "
          f"params: {total:,} total / {trainable:,} trainable before freezing")
    return model
