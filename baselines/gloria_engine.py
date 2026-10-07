"""
GLoRIA (Huang et al., ICCV 2021) as a grounding engine.

Requires the official GLoRIA code (https://github.com/marshuang80/gloria) on
disk, pointed to by $GLORIA_CODE, and its released chexpert_resnet50 checkpoint.

Read-out: GLoRIA's own word-level mechanism. Every patch of the local image
embedding is dotted with each word embedding of the phrase (special tokens and
padding removed) and the maximum over words is kept. The grid is smoothed
(sigma = 1.5) and upsampled to GLoRIA's 224 input size; dump_heatmaps.py then
resizes it onto the 518 ground truth.

With RQP, GLoRIA encodes the pruned phrase only (not the three-variant average
used for KAF-Ground and AFLoc). This is how the GLoRIA numbers in the paper
were produced.
"""
import functools
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _compat                                             # noqa: E402
from kafground.inference import rqp as rqp_lib             # noqa: E402
from kafground.inference.engine import smooth_and_upsample  # noqa: E402


class GloriaEngine:
    IMSIZE = 224

    def __init__(self, ckpt, rqp=False, device=None, code_dir=None):
        _compat.install()
        sys.path.insert(0, str(code_dir or os.environ.get("GLORIA_CODE", "third_party/gloria")))
        # the checkpoint pickles Lightning callbacks, so it needs weights_only=False
        _load = torch.load
        torch.load = functools.partial(_load, weights_only=False)
        import gloria
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = gloria.builder.build_gloria_from_ckpt(str(ckpt)).to(self.device).eval()
        torch.load = _load
        self.rqp = rqp

    @torch.no_grad()
    def grid(self, image_path, phrase):
        imgs = self.model.process_img([str(image_path)], self.device)
        img_emb_l, _ = self.model.image_encoder_forward(imgs)
        iel = img_emb_l[0].permute(1, 2, 0).contiguous()            # [H, W, D]

        query = rqp_lib.prune(phrase) if self.rqp else phrase
        t = self.model.process_text([str(query)], self.device)
        text_emb_l, text_emb_g, sents = self.model.text_encoder_forward(
            t["caption_ids"], t["attention_mask"], t["token_type_ids"])
        keep = [i for i, w in enumerate(sents[0]) if not (w.startswith("[") and w.endswith("]"))]

        h, w, d = iel.shape
        if not keep:   # no real word left: fall back to the sentence embedding
            return (iel.view(-1, d) @ text_emb_g.view(-1, d).t()).reshape(h, w).cpu().numpy()
        words = text_emb_l[0].permute(1, 0)[keep]                   # [n_words, D]
        sim = (iel.view(-1, d) @ words.t()).max(dim=1).values.reshape(h, w)
        return sim.cpu().numpy()

    def heatmap(self, image_path, phrase, out_size=None):
        return smooth_and_upsample(self.grid(image_path, phrase), out_size or self.IMSIZE)
