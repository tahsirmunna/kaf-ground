"""
AFLoc-style vision-language model with a RAD-DINO image encoder and an
optional RadGraph graph branch (KAF-Ground).

Losses (all InfoNCE-style, image <-> text in both directions):
  L_GR  global image embedding  vs. report embedding
  L_DS  shallow patch head (img_emb_l)  vs. sentence embeddings
  L_SW  deep patch head (img_emb_l2)    vs. word embeddings
  L_SG  deep patch head (img_emb_l2)    vs. GAT-encoded RadGraph entities

    L = L_GR + L_DS + L_SW + sigma * L_SG      (sigma = graph_loss_weight)

Grounding maps are read from the shallow head at inference (see
kafground/inference/engine.py).
"""

import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from PIL import Image
from .. import builder
from . import losses
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from skimage import exposure


class AFLoc(nn.Module):
    """
    AFLoc with a frozen RAD-DINO backbone, ClinicalBERT with an optional
    projection head, and an optional DualStreamTextEncoder that enables L_SG.
    Without the graph encoder the model is plain AFLoc with RAD-DINO.
    """

    def __init__(self, cfg, dual_stream_encoder=None):
        super(AFLoc, self).__init__()

        self.cfg = cfg

        # Text encoder (ClinicalBERT)
        self.text_encoder = builder.build_text_model(cfg)

        # optionally freeze BERT
        freeze_bert = getattr(cfg.model.text, "freeze_bert", False)
        if freeze_bert:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
            print(
                f"[AFLoc] ClinicalBERT FROZEN — "
                f"{sum(p.numel() for p in self.text_encoder.parameters()):,} params excluded"
            )
        else:
            print("[AFLoc] ClinicalBERT TRAINABLE")

        # optional trainable projection on top of BERT
        use_text_proj = getattr(cfg.model.text, "use_text_proj_head", False)
        if use_text_proj:
            from .text_proj_head import TextProjHead
            proj_hidden = getattr(cfg.model.text, "text_proj_hidden", 512)
            self.text_proj_head = TextProjHead(
                d=cfg.model.text.embedding_dim,   # 768
                hidden=proj_hidden,
            )
            print(f"[AFLoc] TextProjHead ACTIVE (hidden={proj_hidden})")
        else:
            self.text_proj_head = None
            print("[AFLoc] TextProjHead DISABLED")

        self.img_encoder = builder.build_img_model(cfg)

        self.local_loss  = losses.local_loss
        self.global_loss = losses.global_loss

        self.temp1 = self.cfg.model.afloc.temp1
        self.temp2 = self.cfg.model.afloc.temp2
        self.temp3 = self.cfg.model.afloc.temp3

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # RadGraph graph branch (optional)
        self.dual_stream_encoder = dual_stream_encoder
        self.use_graph_injection  = dual_stream_encoder is not None

        # L_SG weight; 0 disables it
        try:
            self.graph_loss_weight = cfg.model.afloc.graph_loss_weight
        except Exception:
            self.graph_loss_weight = 0.0

        # Which patch head L_SG aligns entities with: "l2" (deep head, shared
        # with L_SW; KAF-Ground) or "l" (shallow head, the one read at
        # inference; the "Route A" ablation).
        target = getattr(cfg.model.afloc, "graph_loss_target", None)   # None on older configs
        self.graph_loss_target = str(target or "l2").lower()
        if self.graph_loss_target not in ("l", "l2"):
            raise ValueError("cfg.model.afloc.graph_loss_target must be 'l' or 'l2'")

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AFLoc] Total params: {total:,}  |  Trainable: {trainable:,}")
        if self.use_graph_injection:
            n_graph = sum(p.numel() for p in self.dual_stream_encoder.parameters())
            print(f"[AFLoc] dual_stream_encoder params: {n_graph:,} "
                  f"(included in totals above)  |  graph_loss_weight={self.graph_loss_weight}")

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        """
        ClinicalBERT, then the projection head (shape preserving) if enabled.

        Inputs:
            caption_ids (torch.Tensor):      tokenized caption  [B, seq_len]
            attention_mask (torch.Tensor):   attention mask     [B, seq_len]
            token_type_ids (torch.Tensor):   token type ids     [B, seq_len]

        Returns:
            res (dict):
                - word_embeddings   [B, 768, seq_len]
                - report_embeddings [B, 768]
                - sent_embeddings   [B, 768, num_sents]
                - report            list
                - sent_units_report list
        """
        res = self.text_encoder(caption_ids, attention_mask, token_type_ids)

        if self.text_proj_head is not None:
            res["word_embeddings"]   = self.text_proj_head(res["word_embeddings"])
            res["sent_embeddings"]   = self.text_proj_head(res["sent_embeddings"])
            res["report_embeddings"] = self.text_proj_head(res["report_embeddings"])

        return res

    def image_encoder_forward(self, imgs):
        """
        Returns:
            img_emb_l  - shallow patch head  [B, 768, H, W]
            img_emb_l2 - deep patch head     [B, 768, H, W]
            img_emb_lf - mean of the two     [B, 768, H, W]
            img_emb_g  - global embedding    [B, 768]

        H = W = 37 for RAD-DINO at 518 px (19 for the ResNet-50 baseline).
        """
        img_feat_g, img_feat_l, img_feat_l2, img_feat_lf = self.img_encoder(
            imgs, get_local=True
        )
        img_emb_g, img_emb_l, img_emb_l2, img_emb_lf = (
            self.img_encoder.generate_embeddings(
                img_feat_g, img_feat_l, img_feat_l2, img_feat_lf
            )
        )
        return img_emb_l, img_emb_l2, img_emb_lf, img_emb_g

    def _get_enhanced_sent_embeddings(self, text_emb_s, graph_data):
        """
        Run the graph branch. Returns the sentence embeddings for L_DS (slot 0
        replaced by the gated fusion output, built with torch.cat rather than
        in place) and the padded node embeddings + mask for L_SG.

        Without graph injection, returns (text_emb_s, None, None).
        """
        if not self.use_graph_injection or graph_data is None:
            return text_emb_s, None, None

        t_s_raw = text_emb_s[:, :, 0]  # [B, 768] — read-only input to the GAT

        t_s_enhanced, node_emb_padded, node_mask = self.dual_stream_encoder(
            t_s_raw, graph_data
        )  # [B,768], [B,max_nodes,768], [B,max_nodes]

        text_emb_s_merged = torch.cat(
            [t_s_enhanced.unsqueeze(-1), text_emb_s[:, :, 1:]], dim=-1
        )  # [B, 768, num_sents]

        return text_emb_s_merged, node_emb_padded, node_mask

    def _calc_local_loss(self, img_emb_l, text_emb_w, report, weight_matrix=None):
        """
        L_DS or L_SW. `report` is a list of token-string lists and is only used
        to count the words of each caption.
        """
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1
            for sent in report
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_w,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            weight_matrix=weight_matrix,
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_r, weight_matrix=None):
        """L_GR."""
        lgr_0, lgr_1, matrix_cosine = self.global_loss(
            img_emb_g, text_emb_r, temp3=self.temp3, weight_matrix=weight_matrix
        )
        return lgr_0, lgr_1, matrix_cosine

    def _calc_graph_loss(self, img_emb_l2, node_emb_padded, node_mask):
        """
        L_SG: entity-level alignment between the GAT node embeddings and a patch
        head (the deep head by default, the image space L_SW also uses). Node
        counts come from node_mask, so local_loss is called directly rather
        than through _calc_local_loss.

        Args:
            img_emb_l2      : [B, 768, H, W]
            node_emb_padded : [B, max_nodes, 768]; transposed to [B, 768, T]
                              because local_loss expects channels first.
            node_mask       : [B, max_nodes] bool, False for padding and for
                              reports without a graph.
        """
        cap_lens = node_mask.sum(dim=1).clamp(min=1).tolist()
        node_emb_for_loss = node_emb_padded.transpose(1, 2)
        lsg_0, lsg_1, attn_maps = self.local_loss(
            img_emb_l2,
            node_emb_for_loss,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            weight_matrix=None,
        )
        return lsg_0, lsg_1, attn_maps

    def calc_loss(self, output, batch):
        """
        Total loss and its parts. L_SG is only added when the graph branch is
        on, its weight is > 0 and the batch has at least one graph.
        """
        img_emb_l         = output["img_emb_l"]
        img_emb_l2        = output["img_emb_l2"]
        img_emb_g         = output["img_emb_g"]
        text_emb_w        = output["text_emb_w"]
        text_emb_r        = output["text_emb_r"]
        text_emb_s        = output["text_emb_s"]
        report            = output["report"]
        sent_units_report = output["sent_units_report"]

        graph_data = batch.get("graph_data", None)

        text_emb_s_for_lds, node_emb_padded, node_mask = self._get_enhanced_sent_embeddings(
            text_emb_s=text_emb_s,
            graph_data=graph_data,
        )

        # L_GR
        if self.cfg.model.afloc.use_global_report_loss:
            lgr_0, lgr_1, matrix_cosine = self._calc_global_loss(
                img_emb_g, text_emb_r
            )
        else:
            lgr_0, lgr_1, matrix_cosine = 0, 0, None

        # L_DS, with the (gated) graph-fused sentence embedding
        if self.cfg.model.afloc.use_local_sent_loss:
            lds_0, lds_1, attn_maps_sent = self._calc_local_loss(
                img_emb_l,
                text_emb_s_for_lds,
                sent_units_report,
            )
        else:
            lds_0, lds_1, attn_maps_sent = 0, 0, None

        # L_SW
        if self.cfg.model.afloc.use_local_word_loss:
            lsw_0, lsw_1, attn_maps2 = self._calc_local_loss(
                img_emb_l2, text_emb_w, report,
            )
        else:
            lsw_0, lsw_1, attn_maps2 = 0, 0, None

        # Weighted total loss
        lgr = (lgr_0 + lgr_1) * self.cfg.model.afloc.global_report_loss_weight
        lds = (lds_0 + lds_1) * self.cfg.model.afloc.local_sent_loss_weight
        lsw = (lsw_0 + lsw_1) * self.cfg.model.afloc.local_word_loss_weight

        # L_SG
        if (
            self.use_graph_injection
            and self.graph_loss_weight > 0
            and node_mask is not None
            and node_mask.any()
        ):
            target = img_emb_l if self.graph_loss_target == "l" else img_emb_l2
            lsg_0, lsg_1, attn_maps_sg = self._calc_graph_loss(
                target, node_emb_padded, node_mask
            )
            lsg = (lsg_0 + lsg_1) * self.graph_loss_weight
        else:
            lsg_0, lsg_1, attn_maps_sg = 0, 0, None
            lsg = 0

        tl = lgr + lds + lsw + lsg

        res = {
            "loss":               tl,
            "global_report_loss": lgr,
            "local_sent_loss":    lds,
            "local_word_loss":    lsw,
            "graph_loss":         lsg,
            "attn_maps_sent":     attn_maps_sent,
            "attn_maps_graph":    attn_maps_sg,
            "report":             report,
            "sent_units_report":  sent_units_report,
            "graph_enhanced":     self.use_graph_injection and graph_data is not None,
        }
        return res

    # ---------------------------------------------------------------------- #
    def forward(self, x):
        """
        Full model forward pass.

        Args:
            x (dict): batch dict, must contain:
                - imgs            (Tensor):  [B, C, H, W]
                - caption_ids     (Tensor):  [B, seq_len]
                - attention_mask  (Tensor):  [B, seq_len]
                - token_type_ids  (Tensor):  [B, seq_len]
                - graph_data      (dict | None): RadGraph batch (optional)

        Returns:
            res (dict): all embeddings
        """
        # Image encoder
        img_emb_l, img_emb_l2, img_emb_lf, img_emb_g = self.image_encoder_forward(
            x["imgs"]
        )

        # Text encoder (with optional proj head applied inside)
        res_text = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        res = {
            # Image embeddings
            "img_emb_l":   img_emb_l,     # [B, 768, H, W] shallow head -> L_DS, inference
            "img_emb_l2":  img_emb_l2,    # [B, 768, H, W] deep head    -> L_SW, L_SG
            "img_emb_lf":  img_emb_lf,    # [B, 768, H, W] mean of both
            "img_emb_g":   img_emb_g,     # [B, 768]        global       -> L_GR

            # Text embeddings (proj_head applied if active)
            "text_emb_w":  res_text["word_embeddings"],    # [B, 768, seq_len]
            "text_emb_r":  res_text["report_embeddings"],  # [B, 768]
            "text_emb_s":  res_text["sent_embeddings"],    # [B, 768, num_sents]
            "report":      res_text["report"],
            "sent_units_report": res_text["sent_units_report"],
        }
        return res

    # inference utilities

    def process_text(self, text, device):
        """Split a phrase into sentences and tokenize it."""
        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            t = t.replace("\n", " ")
            splitter = re.compile(r"[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]
            captions = [sent.strip() for sent in captions if len(sent) > 0]

            tokens = self.tokenizer(
                captions,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            processed_text_tensors.append(tokens)

        caption_ids    = torch.cat([t.input_ids for t in processed_text_tensors]).to(device)
        attention_mask = torch.cat([t.attention_mask for t in processed_text_tensors]).to(device)
        token_type_ids = torch.cat([t.token_type_ids for t in processed_text_tensors]).to(device)

        return {
            "caption_ids":     caption_ids,
            "attention_mask":  attention_mask,
            "token_type_ids":  token_type_ids,
            "cap_lens":        [caption_ids.shape[1]] * caption_ids.shape[0],
        }

    def process_img(self, img_paths, device, equalize_hist=False, transform=None):
        """
        Load image paths into a [B, C, H, W] batch on `device`, using the test
        transform unless one is given.
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        if transform is None:
            transform = builder.build_transformation(self.cfg, "test")

        imgs = []
        for img_path in img_paths:
            if equalize_hist:
                img = self.read_img(img_path)
            else:
                x = cv2.imread(str(img_path), 0)
                img = Image.fromarray(x).convert("RGB")
            img = transform(img)
            imgs.append(img)

        batch = torch.stack(imgs, dim=0).to(device)
        return batch

    # image utilities

    def _resize_img(self, img, scale):
        size    = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)
        if max_ind == 0:
            wpercent     = scale / float(size[0])
            hsize        = int(float(size[1]) * float(wpercent))
            desireable_size = (scale, hsize)
        else:
            hpercent     = scale / float(size[1])
            wsize        = int(float(size[0]) * float(hpercent))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)
        if max_ind == 0:
            pad_size       = scale - resized_img.shape[1]
            left, right    = int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))
            top, bottom    = 0, 0
        else:
            pad_size       = scale - resized_img.shape[0]
            top, bottom    = int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))
            left, right    = 0, 0
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)],
            "constant", constant_values=0,
        )
        return resized_img

    def read_img(self, img_path):
        data = Image.open(img_path)
        img  = self.equalize_hist(data, use_mask=True)
        img  = Image.fromarray(img).convert("RGB")
        return img

    def equalize_hist(self, data, use_mask=True):
        mask = (np.array(data) != 0) if use_mask else None
        img  = np.array(data) / 255.
        img  = exposure.equalize_hist(img, mask=mask)
        img  = (255 * img).astype(np.uint8)
        return img

    def get_imgs(self, img_path, transform=None):
        if "equalize_hist" in self.cfg.data.image and self.cfg.data.image.equalize_hist:
            img = self.read_img(img_path)
        else:
            x   = cv2.imread(str(img_path), 0)
            img = Image.fromarray(x).convert("RGB")
        if transform is not None:
            img = transform(img)
        return img