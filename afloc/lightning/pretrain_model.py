import os

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import builder


class ProgressiveUnfreezeCallback(pl.Callback):
    """
    Freeze/unfreeze schedule for the image encoder.

    Kept from the AFLoc code base for the ResNet-50 baseline. For RAD-DINO the
    backbone forward runs under torch.no_grad (see rad_dino_encoder.py), so the
    backbone stays frozen regardless of this schedule; KAF-Ground sets
    cnn_unfreeze_epoch to 999 to make that explicit.
    """

    def __init__(self, unfreeze_epoch=999, lr_cnn=2e-6):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_cnn         = lr_cnn
        self._unfrozen      = False

    def _cnn_already_in_optimizer(self, optimizer):
        cnn_param_set = set(
            id(p) for p in self._afloc_ref.img_encoder.parameters()
        ) if hasattr(self, "_afloc_ref") else set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if id(p) in cnn_param_set:
                    return True
        return False

    def on_epoch_start(self, trainer, pl_module):
        epoch           = trainer.current_epoch
        self._afloc_ref = pl_module.afloc
        cfg_model       = pl_module.afloc.cfg.model

        model_name = getattr(getattr(cfg_model, "vision", None), "model_name", "resnet_50").lower()
        is_raddino = "rad" in model_name
        encoder    = "RAD-DINO" if is_raddino else "ResNet-50"

        if epoch < self.unfreeze_epoch:
            if not is_raddino:
                for p in pl_module.afloc.img_encoder.parameters():
                    p.requires_grad = False
            if epoch == 0:
                total     = sum(p.numel() for p in pl_module.afloc.parameters())
                trainable = sum(p.numel() for p in pl_module.afloc.parameters() if p.requires_grad)
                print(f"[epoch 0] {encoder} backbone frozen, "
                      f"{trainable:,} trainable / {total - trainable:,} frozen params")
            return

        if not self._unfrozen:
            for p in pl_module.afloc.img_encoder.parameters():
                p.requires_grad = True
            pl_module.afloc.cfg.model.afloc.use_local_word_loss = True

            optimizer = trainer.optimizers[0]
            if not self._cnn_already_in_optimizer(optimizer):
                cnn_params = [p for p in pl_module.afloc.img_encoder.parameters() if p.requires_grad]
                optimizer.add_param_group({"params": cnn_params, "lr": self.lr_cnn})
            self._unfrozen = True
            print(f"[epoch {epoch}] {encoder} unfrozen (lr={self.lr_cnn:.2e})")


class PretrainModel(pl.LightningModule):
    """Lightning wrapper around AFLoc / KAF-Ground pretraining."""

    # set by builder.build_lightning_model before __init__ runs
    _injected_afloc = None

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        if PretrainModel._injected_afloc is not None:
            self.afloc = PretrainModel._injected_afloc
            PretrainModel._injected_afloc = None
            if "load_ckpt" in self.cfg.train and self.cfg.train.load_ckpt:
                self._load_pretrained_weights(self.cfg.train.load_ckpt)
        elif "load_ckpt" in self.cfg.train and self.cfg.train.load_ckpt:
            self.afloc = builder.build_model(self.cfg)
            self._load_pretrained_weights(self.cfg.train.load_ckpt)
        else:
            self.afloc = builder.build_model(cfg)

        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self._apply_initial_freezing()

    # ------------------------------------------------------------------ #
    # optional temperature annealing (ablation; off for KAF-Ground)       #
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self):
        """Linearly decay temp2 to temp2_min when cfg.model.afloc.temp2_anneal is set."""
        afloc_cfg = getattr(self.hparams.model, "afloc", None)
        if not getattr(afloc_cfg, "temp2_anneal", False):
            return

        t_start  = float(getattr(afloc_cfg, "temp2", 1.0))
        t_min    = float(getattr(afloc_cfg, "temp2_min", 0.07))
        max_ep   = int(self.hparams.lightning.trainer.max_epochs)
        frac     = self.current_epoch / max(max_ep - 1, 1)
        new_temp = max(t_min, min(t_start, float(t_start - (t_start - t_min) * frac)))

        if hasattr(self.afloc, "temp2"):
            if isinstance(self.afloc.temp2, torch.Tensor):
                self.afloc.temp2.fill_(new_temp)
            else:
                self.afloc.temp2 = new_temp
        elif afloc_cfg is not None:
            afloc_cfg.temp2 = new_temp

        self.log("train_temp2", new_temp, on_epoch=True, on_step=False, logger=True)

    # ------------------------------------------------------------------ #
    # weight initialisation                                               #
    # ------------------------------------------------------------------ #
    def _load_pretrained_weights(self, ckpt_path):
        """
        Initialise from a checkpoint, controlled by cfg.train.load_ckpt_mode:

        text_only (KAF-Ground): load only text_encoder.* from the released
            AFLoc checkpoint (ResNet-50). The RAD-DINO heads, the text
            projection head and the graph encoder start fresh.
        warm_restart: also load proj_deep, proj_global and text_proj_head from
            an earlier RAD-DINO run (used for the attention-head ablation).
        Otherwise (ResNet-50 runs): load every matching key.
        """
        print(f"[PretrainModel] initialising from {ckpt_path}")
        ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        stripped   = {k.replace("afloc.", "", 1): v for k, v in state_dict.items()}

        model_name = getattr(getattr(self.cfg.model, "vision", None), "model_name", "resnet_50").lower()
        is_raddino = "rad" in model_name
        load_mode  = getattr(self.cfg.train, "load_ckpt_mode", "text_only")

        if is_raddino and load_mode == "warm_restart":
            prefixes = ("text_encoder.", "img_encoder.proj_deep.",
                        "img_encoder.proj_global.", "text_proj_head.")
            keys = {k: v for k, v in stripped.items() if k.startswith(prefixes)}
            missing, unexpected = self.afloc.load_state_dict(keys, strict=False)
            truly_missing = [k for k in missing if "proj_shallow" not in k]
            print(f"  warm_restart: loaded {len(keys)} keys")

        elif is_raddino and load_mode == "text_only":
            keys = {k: v for k, v in stripped.items() if k.startswith("text_encoder.")}
            missing, unexpected = self.afloc.load_state_dict(keys, strict=False)
            # everything outside the text encoder is expected to be new
            expected_new = ("proj", "text_proj", "img_encoder", "dual_stream_encoder")
            truly_missing = [k for k in missing if not any(t in k for t in expected_new)]
            print(f"  text_only: loaded {len(keys)} text-encoder keys")

        else:
            missing, unexpected = self.afloc.load_state_dict(stripped, strict=False)
            expected_new = ("graph_stream", "fusion_proj", "dual_stream", "local2_embedder")
            truly_missing = [k for k in missing if not any(t in k for t in expected_new)]
            print(f"  full load: {len(stripped) - len(unexpected)} keys")

        if truly_missing:
            print(f"  WARNING missing keys: {truly_missing[:5]}")
        if unexpected and load_mode != "text_only":
            print(f"  WARNING unexpected keys: {unexpected[:5]}")

    def _apply_initial_freezing(self):
        model_name = getattr(getattr(self.cfg.model, "vision", None), "model_name", "resnet_50").lower()
        is_raddino = "rad" in model_name
        freeze_cnn = getattr(getattr(self.cfg.model, "vision", None), "freeze_cnn", False)

        # RAD-DINO is frozen inside RadDinoImageEncoder; this only handles ResNet-50.
        if freeze_cnn and not is_raddino:
            for p in self.afloc.img_encoder.parameters():
                p.requires_grad = False

        freeze_bert        = getattr(getattr(self.cfg.model, "text", None), "freeze_bert", False)
        freeze_bert_layers = getattr(getattr(self.cfg.model, "text", None), "freeze_bert_layers", 0)
        if freeze_bert:
            for p in self.afloc.text_encoder.model.parameters():
                p.requires_grad = False
        elif freeze_bert_layers > 0:
            bert = self.afloc.text_encoder.model
            for p in bert.embeddings.parameters():
                p.requires_grad = False
            for i, layer in enumerate(bert.encoder.layer):
                if i < freeze_bert_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        total     = sum(p.numel() for p in self.afloc.parameters())
        trainable = sum(p.numel() for p in self.afloc.parameters() if p.requires_grad)
        lsg_w     = getattr(getattr(self.cfg.model, "afloc", None), "graph_loss_weight", 0.0)
        print(f"[PretrainModel] trainable {trainable:,} / {total:,} params, L_SG weight {lsg_w}")

    # ------------------------------------------------------------------ #
    # optimisation                                                        #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        lr_main        = self.lr
        lr_cnn         = getattr(self.cfg.model.vision, "cnn_lr", 2e-6)
        unfreeze_epoch = getattr(self.cfg.model.vision, "cnn_unfreeze_epoch", 999)

        # When resuming past the unfreeze epoch, rebuild the two param groups.
        resume_ckpt_path      = os.environ.get("PL_RESUME_CHECKPOINT", None)
        resumed_past_unfreeze = False
        if resume_ckpt_path and os.path.exists(resume_ckpt_path):
            try:
                meta = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
                resumed_past_unfreeze = meta.get("epoch", 0) >= unfreeze_epoch
                del meta
            except Exception as e:
                print(f"[Optimizer] could not read checkpoint metadata: {e}")

        if resumed_past_unfreeze:
            for p in self.afloc.img_encoder.parameters():
                p.requires_grad = True
            main_params = [
                p for p in self.afloc.parameters()
                if p.requires_grad and not any(p is cp for cp in self.afloc.img_encoder.parameters())
            ]
            param_groups = [
                {"params": main_params, "lr": lr_main},
                {"params": list(self.afloc.img_encoder.parameters()), "lr": lr_cnn},
            ]
        else:
            param_groups = [{"params": [p for p in self.afloc.parameters() if p.requires_grad],
                             "lr": lr_main}]

        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr_main,
            weight_decay=self.cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )

        sched_dict = builder.build_scheduler(self.cfg, optimizer, self.dm)
        sched_name = getattr(self.cfg.train.scheduler, "name", "step")
        if sched_name == "cos" and hasattr(self.cfg.train.scheduler, "T_max"):
            sched_dict["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.cfg.train.scheduler.T_max),
                eta_min=float(getattr(self.cfg.train.scheduler, "eta_min", 1e-8)),
            )

        return {"optimizer": optimizer, "lr_scheduler": sched_dict}

    # ------------------------------------------------------------------ #
    # steps                                                               #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        self.afloc._current_epoch = self.current_epoch
        res = self.shared_step(batch, "train")

        # tanh(fusion gate): stays at 0 in the released configuration
        # (see dual_stream_text.py).
        if getattr(self.afloc, "dual_stream_encoder", None) is not None and batch_idx % 50 == 0:
            self.log("fusion_gate", self.afloc.dual_stream_encoder.current_gate_value(),
                     on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        return res["loss"]

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")["loss"]

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, split):
        on_step = split == "train"
        output  = self.afloc(batch)
        loss    = self.afloc.calc_loss(output, batch)

        self.log(f"{split}_loss", loss["loss"], on_epoch=True, on_step=on_step,
                 logger=True, prog_bar=True)

        for key, short in (("local_sent_loss", "L_DS"), ("local_word_loss", "L_SW"),
                           ("global_report_loss", "L_GR")):
            val = loss.get(key, 0) or 0
            if hasattr(val, "item"):
                val = val.item()
            self.log(f"{split}_{short}", float(val), on_epoch=True, on_step=on_step,
                     logger=True, prog_bar=True)

        if loss.get("graph_loss", None) is not None:
            self.log(f"{split}_L_SG", loss["graph_loss"], on_epoch=True, on_step=on_step,
                     logger=True, prog_bar=True)

        graph_data = batch.get("graph_data", None)
        if graph_data is not None and "available_mask" in graph_data:
            self.log(f"{split}_graph_coverage",
                     graph_data["available_mask"].float().mean().item(),
                     on_epoch=True, on_step=False, logger=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------ #
    # MS-CXR monitor                                                      #
    # ------------------------------------------------------------------ #
    # A quick grounding check printed during training. It is for monitoring
    # only: checkpoints are saved on the MIMIC-CXR validation loss
    # (lightning.checkpoint_callback.monitor), and all reported numbers come
    # from scripts/dump_heatmaps.py + scripts/score_heatmaps.py.
    # Set `monitor_mscxr: false` in the config to skip it.

    def _compute_sim_maps(self, data, device):
        """Wordmax similarity maps on the shallow head, smoothed with sigma=1.5."""
        from scipy import ndimage
        import cv2

        model = self.afloc
        model.eval()
        sim_maps, n_failed, first_error = [], 0, None

        for i in range(len(data["path"])):
            try:
                pi = model.process_img([data["path"][i]], device, equalize_hist=False)
                with torch.no_grad():
                    iel, _, _, _ = model.image_encoder_forward(pi.to(device))
                iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)      # [37, 37, 768]
                n_h, n_w, feat = iel.shape

                pt = model.process_text(data["label_text"][i], device)
                with torch.no_grad():
                    res = model.text_encoder_forward(
                        pt["caption_ids"].to(device),
                        pt["attention_mask"].to(device),
                        pt["token_type_ids"].to(device),
                    )
                word_emb  = res["word_embeddings"].squeeze(0).transpose(0, 1)  # [seq, 768]
                attn_mask = pt["attention_mask"].to(device)
                if attn_mask.dim() > 1:
                    attn_mask = attn_mask.squeeze(0)
                attn_mask = attn_mask.bool()

                cos_sim = F.normalize(iel.reshape(-1, feat), dim=-1) @ F.normalize(word_emb, dim=-1).t()
                cos_sim = cos_sim.masked_fill(~attn_mask.unsqueeze(0), -1e4)
                patch_sim = cos_sim.max(dim=-1).values.reshape(n_h, n_w).cpu().numpy()
                smoothed  = ndimage.gaussian_filter(patch_sim, sigma=(1.5, 1.5))

                gtmask = data["gtmasks"][i]
                if gtmask.shape[:2] != smoothed.shape:
                    smoothed = cv2.resize(smoothed, (gtmask.shape[1], gtmask.shape[0]))
                sim_maps.append((smoothed, gtmask))
            except Exception as e:
                n_failed += 1
                first_error = first_error or f"{type(e).__name__}: {e}"
                sim_maps.append(None)

        if n_failed:
            print(f"[monitor] {n_failed}/{len(data['path'])} samples failed: {first_error}")
        return sim_maps, n_failed, len(data["path"])

    @staticmethod
    def _iou_at_threshold(sim_maps, threshold):
        from kafground.eval.datasets import norm_heatmap
        from kafground.eval.metrics import compute_iou

        ious = []
        for entry in sim_maps:
            if entry is None:
                ious.append(0.0)
                continue
            smoothed, gtmask = entry
            nan  = np.isnan(smoothed)
            hmap = norm_heatmap(smoothed, nan, mode=0)
            ious.append(compute_iou(gtmask, np.where(hmap > threshold, 1, 0), nan))
        return float(np.nanmean(ious))

    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch
        if epoch == 0 or self.global_rank != 0 or not self.cfg.get("monitor_mscxr", True):
            return
        try:
            from kafground.eval.datasets import load_ms_cxr, set_eval_img_size

            set_eval_img_size(self.cfg.data.image.imsize)
            device = next(self.afloc.parameters()).device
            data   = load_ms_cxr()
            sim_maps, n_failed, n_total = self._compute_sim_maps(data, device)

            if n_failed == n_total:
                self.log("val_miou_quick", float("nan"), on_epoch=True, on_step=False,
                         logger=True, prog_bar=True)
                return

            quick = self._iou_at_threshold(sim_maps, 0.5)
            self.log("val_miou_quick", quick, on_epoch=True, on_step=False,
                     logger=True, prog_bar=True)
            print(f"[monitor] epoch {epoch}: MS-CXR IoU@0.5 = {quick:.4f}")

            if epoch % 10 == 0:
                per_t = [self._iou_at_threshold(sim_maps, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]
                self.log("val_miou", float(np.mean(per_t)), on_epoch=True, on_step=False,
                         logger=True, prog_bar=True)
                print(f"[monitor] epoch {epoch}: MS-CXR mIoU = {np.mean(per_t):.4f}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[monitor] MS-CXR check skipped at epoch {epoch}: {e}")

    # ------------------------------------------------------------------ #
    # resuming                                                            #
    # ------------------------------------------------------------------ #
    def on_load_checkpoint(self, checkpoint):
        """Fill keys the checkpoint predates and start Adam afresh."""
        model_state = self.state_dict()
        ckpt_state  = checkpoint["state_dict"]
        for k, v in model_state.items():
            if k not in ckpt_state:
                ckpt_state[k] = v
        checkpoint["state_dict"] = ckpt_state
        if checkpoint.get("optimizer_states"):
            checkpoint["optimizer_states"] = []
