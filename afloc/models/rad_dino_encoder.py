"""
RAD-DINO image encoder: a frozen microsoft/rad-dino ViT-B/14 plus three small
trainable projection heads.

At 518 x 518 the backbone yields a 37 x 37 patch grid (1,369 tokens). The patch
grid feeds two heads, proj_shallow (img_emb_l: L_DS and inference) and
proj_deep (img_emb_l2: L_SW and L_SG); the CLS token feeds proj_global (L_GR).

cfg.model.vision.proj_type = "attention" swaps proj_shallow for a single
Transformer block over the patch tokens (an ablation; the default "linear" is
a 1x1-conv MLP).
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class _ProjConv(nn.Module):
    """1x1-conv MLP over a feature map: [B, in_dim, H, W] -> [B, out_dim, H, W]."""
    def __init__(self, in_dim: int = 768, out_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=True),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out")
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttentionProjHead(nn.Module):
    """
    Ablation replacement for proj_shallow: one pre-norm Transformer block
    (2-head self-attention + FFN) over the 1,369 patch tokens. Same input and
    output shape as _ProjConv.

    The attention output projection and the second FFN layer start at zero, so
    the block is exactly the identity at initialisation and only departs from
    it as training moves those weights.
    """

    def __init__(self, dim: int = 768, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()

        assert dim % num_heads == 0, \
            f"SpatialAttentionProjHead: dim={dim} must be divisible by num_heads={num_heads}"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),  # index [2] — zero-initialized below
        )

        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, C, H, W]"""
        B, C, H, W = x.shape

        x_seq = x.flatten(2).transpose(1, 2)           # [B, H*W, C]

        x_normed = self.norm1(x_seq)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ffn(self.norm2(x_seq))

        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class _ProjVec(nn.Module):
    """MLP head for the global (CLS) vector: [B, in_dim] -> [B, out_dim]."""
    def __init__(self, in_dim: int = 768, out_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadDinoImageEncoder(nn.Module):
    """
    Frozen RAD-DINO backbone + proj_shallow / proj_deep / proj_global.

        forward(imgs)                    -> (feat_g, feat_l, feat_l2, feat_lf)
        generate_embeddings(g, l, l2, lf) -> (emb_g, emb_l, emb_l2, emb_lf)
    """

    _VALID_PROJ_TYPES = ('linear', 'attention')

    @staticmethod
    def _resolve_proj_type(cfg) -> str:
        """Read cfg.model.vision.proj_type; missing, null or unknown -> "linear"."""
        proj_type_raw = getattr(cfg.model.vision, 'proj_type', None)
        proj_type = str(proj_type_raw or "linear").strip().lower()
        if proj_type not in RadDinoImageEncoder._VALID_PROJ_TYPES:
            print(
                f"[RadDinoImageEncoder] WARNING: unrecognized "
                f"proj_type={proj_type_raw!r} (resolved to {proj_type!r}); "
                f"falling back to 'linear'"
            )
            proj_type = 'linear'

        return proj_type

    def __init__(self, cfg):
        super().__init__()

        hf_name = getattr(cfg.model.vision, "hf_name", "microsoft/rad-dino")
        out_dim  = cfg.model.vision.out_dim    # 768

        # frozen backbone
        print(f"[RadDinoImageEncoder] Loading backbone: {hf_name}")
        self.backbone = AutoModel.from_pretrained(hf_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.num_reg = int(
            getattr(self.backbone.config, "num_register_tokens", 0)
        )
        d = self.backbone.config.hidden_size    # 768 for ViT-B

        print(
            f"[RadDinoImageEncoder] hidden_size={d}, "
            f"num_register_tokens={self.num_reg}, "
            f"backbone params frozen: {sum(p.numel() for p in self.backbone.parameters()):,}"
        )

        proj_type  = self._resolve_proj_type(cfg)
        num_heads  = getattr(cfg.model.vision, 'proj_attn_heads', 2)
        proj_drop  = getattr(cfg.model.vision, 'proj_dropout', 0.0)

        num_heads = num_heads if num_heads is not None else 2
        proj_drop = proj_drop if proj_drop is not None else 0.0

        if proj_type == 'attention':
            self.proj_shallow = SpatialAttentionProjHead(
                dim=d, num_heads=num_heads, dropout=proj_drop
            )
            print(
                f"[RadDinoImageEncoder] proj_shallow -> SpatialAttentionProjHead "
                f"(dim={d}, heads={num_heads}, dropout={proj_drop})"
            )
        else:
            self.proj_shallow = _ProjConv(d, out_dim)
            print("[RadDinoImageEncoder] proj_shallow -> _ProjConv")

        self.proj_deep   = _ProjConv(d, out_dim)
        self.proj_global = _ProjVec(d, out_dim)

        trainable = (
            sum(p.numel() for p in self.proj_shallow.parameters()) +
            sum(p.numel() for p in self.proj_deep.parameters()) +
            sum(p.numel() for p in self.proj_global.parameters())
        )
        print(f"[RadDinoImageEncoder] Trainable head params: {trainable:,}")

    def train(self, mode: bool = True):
        # the backbone always stays in eval mode
        super().train(mode)
        self.backbone.eval()
        return self

    def _patch_grid(self, imgs: torch.Tensor):
        """
        Args:
            imgs: [B, 3, 518, 518]
        Returns:
            cls:  [B, 768]
            grid: [B, 768, 37, 37]
        """
        # The backbone runs under no_grad, so it stays frozen whatever its
        # requires_grad flags say.
        with torch.no_grad():
            seq = self.backbone(pixel_values=imgs).last_hidden_state

        cls     = seq[:, 0]                             # [B, 768]
        patches = seq[:, 1 + self.num_reg:]             # [B, N, 768]

        B, N, C = patches.shape
        side = int(round(N ** 0.5))
        if side * side != N:
            raise ValueError(
                f"[RadDinoImageEncoder] patch count {N} is not square. "
                f"Expected 1369 (37×37). Check input resolution ({imgs.shape}) "
                f"and num_register_tokens ({self.num_reg})."
            )

        grid = patches.transpose(1, 2).reshape(B, C, side, side)
        return cls, grid                                # [B,768], [B,768,37,37]

    def forward(self, imgs: torch.Tensor, get_local: bool = True):
        """
        Frozen backbone features in (g, l, l2, lf) order:
            feat_g  [B, 768] CLS token; feat_l = feat_l2 = feat_lf = [B, 768, 37, 37].
        """
        cls, grid = self._patch_grid(imgs)
        return cls, grid, grid, grid    # (g, l, l2, lf)

    def generate_embeddings(
        self,
        feat_g:  torch.Tensor,         # [B, 768]
        feat_l:  torch.Tensor,         # [B, 768, 37, 37]
        feat_l2: torch.Tensor,         # [B, 768, 37, 37]
        feat_lf: torch.Tensor,         # [B, 768, 37, 37]
    ):
        """
        Apply the trainable heads.

        Returns:
            emb_g:  [B, 768]          global -> L_GR
            emb_l:  [B, 768, 37, 37]  shallow head -> L_DS, inference maps
            emb_l2: [B, 768, 37, 37]  deep head    -> L_SW, L_SG
            emb_lf: [B, 768, 37, 37]  mean of the two
        """
        emb_g  = self.proj_global(feat_g)              # [B, 768]

        emb_l  = self.proj_shallow(feat_l)             # [B, 768, 37, 37]
        emb_l2 = self.proj_deep(feat_l2)               # [B, 768, 37, 37]
        emb_lf = 0.5 * (emb_l + emb_l2)               # [B, 768, 37, 37]
        return emb_g, emb_l, emb_l2, emb_lf