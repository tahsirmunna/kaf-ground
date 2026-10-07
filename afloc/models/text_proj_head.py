"""
Trainable projection applied to ClinicalBERT's word, sentence and report
embeddings (shared weights). The text encoder is initialised from the AFLoc
checkpoint, whose embedding space was aligned with ResNet-50 features; this
small head gives it room to meet the new RAD-DINO feature space.

Enabled with model.text.use_text_proj_head.
"""

import torch
import torch.nn as nn


class TextProjHead(nn.Module):
    """
    Shape-preserving MLP head for text embeddings.

    Handles all three shapes produced by ClinicalBERT:
        [B, D]            → global report embedding
        [B, D, L]         → word or sentence embeddings (L = seq positions)

    The same weights are applied to all three — shared projection
    keeps the embedding spaces consistent across all three losses.

    Args:
        d      (int): input/output dimension (768 to match BERT & image heads)
        hidden (int): bottleneck hidden dimension (512 by default)
    """

    def __init__(self, d: int = 768, hidden: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        # near-identity start, so the AFLoc text space is not disturbed at step 0
        nn.init.eye_(self.proj[0].weight[:hidden, :hidden]
                     if hidden <= d else self.proj[0].weight[:, :d])
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[3].bias)

        params = sum(p.numel() for p in self.parameters())
        print(f"[TextProjHead] d={d}, hidden={hidden}, params={params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, D, L]
        Returns:
            same shape as input
        """
        if x.dim() == 2:
            # [B, D] — report embedding
            return self.proj(x)                        # [B, D]

        if x.dim() == 3:
            # [B, D, L] — word or sentence embeddings
            # transpose → [B, L, D], apply linear, transpose back
            return self.proj(x.transpose(1, 2)).transpose(1, 2)  # [B, D, L]

        raise ValueError(
            f"[TextProjHead] Expected 2D or 3D input, got {x.dim()}D "
            f"with shape {tuple(x.shape)}"
        )