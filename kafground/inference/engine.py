"""
Phrase -> heatmap for KAF-Ground and for the AFLoc baseline.

Both engines follow the same steps after the similarity grid is formed:

    grid (H x W)  ->  Gaussian smoothing (sigma = 1.5)
                  ->  bilinear upsampling to the evaluation size (518)

and, with RQP, the grids of the query variants are min-max normalised and
averaged before smoothing. (For a single variant this normalisation is an affine
map and leaves every metric unchanged.)

Each model is read out the way it was trained:

* KAF-Ground (`KAFGroundEngine`): *word-max*. Every patch of the shallow
  projection head takes its maximum cosine similarity over the phrase's word
  embeddings ([CLS]/[SEP]/padding dropped).
* AFLoc (`AFLocEngine`): *report matching*, i.e. the raw dot product between each
  patch and the phrase's report-level embedding, which is how the released
  ResNet-50 checkpoint was trained. Forcing word-max on it collapses its mIoU.

The margin (pixels outside the central crop) is applied when scoring, not
here; see `set_margin` and scripts/score_heatmaps.py.
"""
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import rqp as rqp_lib

SIGMA = 1.5


# --------------------------------------------------------------------------- #
# shared post-processing                                                       #
# --------------------------------------------------------------------------- #
def norm01(t: torch.Tensor) -> torch.Tensor:
    rng = t.max() - t.min()
    return (t - t.min()) / rng if rng > 0 else torch.zeros_like(t)


def smooth_and_upsample(grid: np.ndarray, size: int) -> np.ndarray:
    """Gaussian smoothing on the patch grid, then bilinear resize to size x size."""
    g = ndimage.gaussian_filter(np.asarray(grid, dtype=np.float32), sigma=(SIGMA, SIGMA), order=0)
    t = torch.tensor(g).reshape(1, 1, *g.shape)
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def set_margin(hmap: np.ndarray, resize_size=512, crop_size=448) -> np.ndarray:
    """
    Mark the border outside the central crop as NaN (AFLoc's `--margin`
    protocol): for a 518 map, a 453 x 453 centre is kept. Scoring ignores NaN
    pixels.
    """
    height, width = hmap.shape
    side = int(crop_size * min(height, width) / resize_size)
    mw, mh = width - side, height - side
    pad = (floor(mw / 2), ceil(mw / 2), floor(mh / 2), ceil(mh / 2))
    nan = torch.isnan(F.pad(torch.zeros(side, side), pad, value=float("nan"))).numpy()
    out = np.array(hmap, dtype=np.float32, copy=True)
    out[nan] = np.nan
    return out


# --------------------------------------------------------------------------- #
# KAF-Ground                                                                   #
# --------------------------------------------------------------------------- #
NATIVE_GRID, PATCH = 37, 14


def parse_views(spec):
    """'id@518,hflip@518' -> [('id', 518), ('hflip', 518)] (sizes multiple of 14)."""
    views = []
    for tok in spec.split(","):
        op, _, size = tok.strip().partition("@")
        if op not in ("id", "hflip") or int(size) % PATCH:
            raise ValueError(f"bad view '{tok}'")
        views.append((op, int(size)))
    return views


class KAFGroundEngine:
    """
    Word-max grounding with the KAF-Ground checkpoint.

    Defaults give the paper's setting. The other options exist for the
    ablations in results/ablations:
        head        "l" shallow head (paper) | "l2" deep head | "lf" their mean
        query_mode  see rqp.query_variants ("rg-ens" when rqp=True)
        views       test-time augmentation, e.g. "id@518,hflip@518"; each view's
                    grid is mapped back to 37 x 37, min-max normalised and averaged
        tokens      "all" (paper) | "rg-mask" (max over anatomy + finding words)
                    | "rg-mean" (mean over them) | "rg-anat" (anatomy words only);
                    needs a role lexicon from build_rqp_lexicon.py --roles
    """

    def __init__(self, ckpt, rqp=False, head="l", device=None, query_mode=None,
                 views="id@518", tokens="all", role_lexicon=None):
        import afloc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = afloc.builder.load_model(
            ckpt_path=str(ckpt),
            device=self.device,
            hf_name_override="microsoft/rad-dino",
            bert_type_override="emilyalsentzer/Bio_ClinicalBERT",
        ).eval()
        self.size = int(self.model.cfg.data.image.imsize)          # 518 -> 37 x 37
        self.transform = afloc.builder.build_transformation(self.model.cfg, "test")
        self.head = head
        self.query_mode = query_mode or ("rg-ens" if rqp else "orig")
        self.views = parse_views(views)
        self.tokens = tokens
        self.roles = rqp_lib.load_roles(role_lexicon) if tokens != "all" else None

    @torch.no_grad()
    def _word_embeddings(self, text):
        pt = self.model.process_text(text, self.device)
        res = self.model.text_encoder_forward(
            pt["caption_ids"].to(self.device),
            pt["attention_mask"].to(self.device),
            pt["token_type_ids"].to(self.device),
        )
        wte, am = res["word_embeddings"], pt["attention_mask"].to(self.device)
        if wte.dim() == 2:
            wte = wte.unsqueeze(0)
        ids = pt["caption_ids"]
        words, roles = [], []
        for b in range(wte.shape[0]):
            keep = am[b].bool()
            seq, sid = wte[b].transpose(0, 1)[keep], ids[b][keep.to(ids.device)]
            if seq.shape[0] > 2:
                seq, sid = seq[1:-1], sid[1:-1]          # drop [CLS], [SEP]
            words.append(seq)
            if self.roles is not None:
                toks = self.model.tokenizer.convert_ids_to_tokens(sid.tolist())
                roles.extend(rqp_lib.subword_roles(self.roles, toks))
        return torch.cat(words, 0), roles                # [n_words, C]

    @torch.no_grad()
    def _view_grid(self, x, words, roles, op, size):
        if size != x.shape[-1]:
            x = F.interpolate(x, size=(size, size), mode="bilinear",
                              align_corners=False, antialias=True)
        if op == "hflip":
            x = torch.flip(x, dims=[3])
        emb_l, emb_l2, emb_lf, _ = self.model.image_encoder_forward(x)
        iel = {"l": emb_l, "l2": emb_l2, "lf": emb_lf}[self.head]
        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)          # [H, W, C]
        H, W, C = iel.shape

        sim = F.normalize(iel.reshape(-1, C), dim=1) @ F.normalize(words, dim=1).t()
        if self.tokens == "all":
            g = sim.max(dim=1).values
        else:
            wanted = ("anat",) if self.tokens == "rg-anat" else ("anat", "finding")
            keep = torch.tensor([r in wanted for r in roles], device=sim.device)
            if not bool(keep.any()):
                keep = torch.ones_like(keep)
            sel = sim[:, keep]
            g = sel.mean(dim=1) if self.tokens == "rg-mean" else sel.max(dim=1).values
        g = torch.nan_to_num(g.reshape(H, W).float(), nan=0.0, posinf=0.0, neginf=0.0)

        if op == "hflip":
            g = torch.flip(g, dims=[1])
        if (H, W) != (NATIVE_GRID, NATIVE_GRID):
            g = F.interpolate(g[None, None], size=(NATIVE_GRID, NATIVE_GRID),
                              mode="bilinear", align_corners=False)[0, 0]
        return norm01(g)

    def grid(self, image_path, phrase):
        """Word-max similarity grid on the 37 x 37 patch lattice (before smoothing)."""
        x = self.model.get_imgs(image_path, self.transform).unsqueeze(0).to(self.device)
        grids = []
        for q in rqp_lib.query_variants(phrase, self.query_mode):
            words, roles = self._word_embeddings(q)
            if words.shape[0] == 0:
                continue
            grids.extend(self._view_grid(x, words, roles, op, s) for op, s in self.views)
        if not grids:
            return np.zeros((NATIVE_GRID, NATIVE_GRID), dtype=np.float32)
        return torch.stack(grids).mean(0).cpu().numpy().astype(np.float32)

    def heatmap(self, image_path, phrase, out_size=None):
        return smooth_and_upsample(self.grid(image_path, phrase), out_size or self.size)


# --------------------------------------------------------------------------- #
# AFLoc baseline (released ResNet-50 checkpoint)                               #
# --------------------------------------------------------------------------- #
class AFLocEngine:
    """
    The released AFLoc checkpoint (ResNet-50) read out with report matching.

    Inputs are prepared exactly as AFLoc's own inference code does:
      image  grayscale read, long side resized to 224 and centre-padded,
             224 centre crop, normalisation to [-1, 1];
      text   lower-cased and split into \\w+ tokens (so "right-sided" becomes
             "right sided"), sentences of a single token dropped.
    """

    IMSIZE = 224

    def __init__(self, ckpt, rqp=False, device=None, out_size=518):
        import afloc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = afloc.builder.load_model(ckpt_path=str(ckpt), device=self.device).eval()
        self.model.cfg.data.image.imsize = self.IMSIZE
        self.transform = afloc.builder.build_transformation(self.model.cfg, "test")
        self.size = out_size
        self.rqp = rqp

    def _image(self, image_path):
        import cv2
        from PIL import Image

        x = cv2.imread(str(image_path), 0)
        x = self.model._resize_img(x, self.IMSIZE)
        return self.transform(Image.fromarray(x).convert("RGB")).unsqueeze(0).to(self.device)

    def _tokenize(self, text):
        import re
        from nltk.tokenize import RegexpTokenizer

        t = text.replace("\n", " ")
        captions = [s for point in re.split(r"[0-9]+\.", t) for s in point.split(".")]
        sents = []
        for c in captions:
            tokens = RegexpTokenizer(r"\w+").tokenize(c.replace("\ufffd\ufffd", " ").lower())
            if len(tokens) <= 1:
                continue
            tokens = [w.encode("ascii", "ignore").decode("ascii") for w in tokens]
            sents.append(" ".join(w for w in tokens if w))
        return self.model.tokenizer(" ".join(sents), return_tensors="pt", truncation=True,
                                    padding="max_length",
                                    max_length=self.model.cfg.data.text.word_num)

    @torch.no_grad()
    def _report_embedding(self, text):
        pt = self._tokenize(text)
        res = self.model.text_encoder_forward(
            pt["input_ids"].to(self.device),
            pt["attention_mask"].to(self.device),
            pt["token_type_ids"].to(self.device),
        )
        teg = res["report_embeddings"]
        if teg.dim() == 1:
            teg = teg.unsqueeze(0)
        elif teg.dim() == 3:
            teg = teg.view(-1, teg.shape[-1])
        if teg.shape[0] > 1:
            teg = teg.mean(dim=0, keepdim=True)
        return teg

    @torch.no_grad()
    def grid(self, image_path, phrase):
        iel, _, _, _ = self.model.image_encoder_forward(self._image(image_path))
        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)
        H, W, C = iel.shape
        P = iel.reshape(-1, C)
        grids = []
        for q in rqp_lib.query_variants(phrase, self.rqp):
            sim = (P @ self._report_embedding(q).t()).reshape(H, W).float()
            grids.append(norm01(torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)))
        return torch.stack(grids).mean(0).cpu().numpy().astype(np.float32)

    def heatmap(self, image_path, phrase, out_size=None):
        return smooth_and_upsample(self.grid(image_path, phrase), out_size or self.size)
