"""
MedKLIP (Wu et al., ICCV 2023) as a grounding engine.

Requires the official MedKLIP code (https://github.com/MediaBrain-SJTU/MedKLIP),
pointed to by $MEDKLIP_CODE (the Sample_Zero-Shot_Grounding_RSNA folder), and
its released checkpoint.

MedKLIP does not read free text: its decoder attends 75 fixed entity queries
over the image, and a heatmap is selected by entity index (as in the official
RSNA grounding demo). Each evaluation category is therefore mapped to one
entity (tables below) and the phrase itself is ignored; RQP is undefined for
MedKLIP. The 14 x 14 attention map (mean of the last four decoder layers) is
smoothed and upsampled to 224, and dump_heatmaps.py resizes it onto the 518
ground truth.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kafground.inference.engine import smooth_and_upsample  # noqa: E402

# the 75 entity queries, in the checkpoint's order
ENTITIES = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
    'process', 'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion',
    'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease',
    'calcification', 'prominence', 'device', 'engorgement', 'picc', 'clip', 'elevation',
    'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking',
    'scar', 'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema',
    'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia', 'drainage',
    'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
    'hardware', 'dilation', 'chf', 'redistribution', 'aspiration', 'tail_abnorm_obs',
    'excluded_obs',
]

ANATOMY = [
    'trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
    'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
    'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
    'upper_left_lobe', 'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe',
    'left_lower_lung', 'left_mid_lung', 'left_upper_lung', 'left_apical_lung',
    'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung',
    'right_apical_lung', 'right_lung_unspec', 'lung_apices', 'lung_bases',
    'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
    'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach',
    'right_atrium', 'right_ventricle', 'aorta', 'svc', 'interstitium', 'parenchymal',
    'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes',
    'unspecified', 'other',
]

# evaluation category -> MedKLIP entity
CATEGORY_TO_ENTITY = {
    "ms-cxr": {
        "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly",
        "Consolidation": "consolidation", "Edema": "edema", "Lung Opacity": "opacity",
        "Pleural Effusion": "effusion", "Pneumonia": "pneumonia",
        "Pneumothorax": "pneumothorax",
    },
    "padchest-gr": {
        "cardiomegaly": "cardiomegaly", "pleural effusion": "effusion",
        "atelectasis": "atelectasis", "alveolar pattern": "consolidation",
        "interstitial pattern": "marking", "nodule": "nodule",
        "pleural thickening": "thicken", "aortic elongation": "tortuous",
    },
}


class MedKLIPEngine:
    IMSIZE, GRID = 224, 14

    def __init__(self, ckpt, dataset="ms-cxr", device=None, code_dir=None):
        src = Path(code_dir or os.environ.get(
            "MEDKLIP_CODE", "third_party/MedKLIP/Sample_Zero-Shot_Grounding_RSNA"))
        # MedKLIP ships a top-level `models` package; make sure it wins.
        for m in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[m]
        sys.path.insert(0, str(src))
        from models.model_MedKLIP import MedKLIP
        from transformers import BertTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cat2ent = CATEGORY_TO_ENTITY[dataset]
        cfg = dict(d_model=256, res_base_model="resnet50", num_queries=75, dropout=0.1,
                   attribute_set_size=2, N=4, H=4,
                   text_encoder="emilyalsentzer/Bio_ClinicalBERT")

        book = json.load(open(src / "observation explanation.json"))
        tok = BertTokenizer.from_pretrained(cfg["text_encoder"])
        enc = lambda texts: tok(list(texts), padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt").to(self.device)
        model = MedKLIP(cfg, enc(["It is located at " + a for a in ANATOMY]),
                        enc([book[k] for k in book]), mode="train")
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["model"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        self.model = model.to(self.device).eval()

    def _image(self, path):
        import cv2
        from PIL import Image
        import torchvision.transforms as T

        x = cv2.imread(str(path), 0)
        t = T.ToTensor()(Image.fromarray(x).convert("RGB").resize((self.IMSIZE, self.IMSIZE)))
        t = (t - t.mean()) / (t.std() + 1e-6)          # per-image z-score, as in the demo
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def grid(self, image_path, category):
        ent = ENTITIES.index(self.cat2ent[category])
        _, ws = self.model(self._image(image_path), None, is_train=False)
        ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4
        ws = ws.reshape(1, ws.shape[1], self.GRID, self.GRID)
        return ws[0, ent].float().cpu().numpy()

    def heatmap(self, image_path, phrase=None, category=None, out_size=None):
        return smooth_and_upsample(self.grid(image_path, category), out_size or self.IMSIZE)
