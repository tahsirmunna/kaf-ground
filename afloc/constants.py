import os
from pathlib import Path

PRETRAIN_VIEW_COL = "view"
PRETRAIN_PATH_COL = "path"
PRETRAIN_SPLIT_COL = "split"
PRETRAIN_REPORT_COL = "report"
PRETRAIN_IMPRESSION_COL = "impression"
PICKLE_SUFFIX = "report"

# MIMIC-CXR locations. Set these through the environment (see README):
#   KAF_MIMIC_IMG_DIR  root that the `path` column of the csv is relative to
#   KAF_MIMIC_CSV      master csv with path / view / split / report columns
MIMIC_IMG_DIR = Path(os.environ.get("KAF_MIMIC_IMG_DIR", "data/mimic-cxr-jpg"))
MIMIC_MASTER_CSV = os.environ.get("KAF_MIMIC_CSV", "data/mimic_master.csv")

MIMIC_VALID_NUM = 5000
MIMIC_VIEW_COL = "view"
MIMIC_PATH_COL = "path"
MIMIC_SPLIT_COL = "split_with_MS"
MIMIC_REPORT_COL = "report"
MIMIC_IMPRESSION_COL = "impression"
MIMIC_FINDINGS_COL = "findings"

MIMIC_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

MIMIC_USED_COLS = [
    MIMIC_PATH_COL,
    MIMIC_VIEW_COL,
    MIMIC_SPLIT_COL,
    MIMIC_REPORT_COL,
    MIMIC_IMPRESSION_COL] + MIMIC_TASKS
