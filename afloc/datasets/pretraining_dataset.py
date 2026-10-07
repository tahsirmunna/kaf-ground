import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from afloc.constants import *

# Optional ablation: train on RadGraph-rewritten report sentences
# (cfg.model.text.use_kg_sentences). Off for KAF-Ground.
try:
    from afloc.knowledge.kg_sentence_builder import RadGraphKGBuilder
    _KG_BUILDER_AVAILABLE = True
except ImportError:
    _KG_BUILDER_AVAILABLE = False
import PIL
from skimage import exposure


class MultimodalPretrainingDataset(data.Dataset):
    """
    MIMIC-CXR image-report pairs for pretraining.

    Each item also carries the RadGraph study key (e.g. 'p10/p10659857/s59206984'),
    which the collate function uses to fetch that report's entity graph for L_SG.
    Reports without a graph are marked unavailable and contribute nothing to L_SG.
    """

    def __init__(self, cfg, split="train", transform=None):
        """
        Args:
            cfg (object):       Configuration object.
            split (str):        Dataset split — 'train', 'val', or 'test'.
            transform (object): Image augmentation transforms.
        """
        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        self.df = pd.DataFrame()
        if 'mimic' in self.cfg.data.dataset:
            self.df_mimic = pd.read_csv(MIMIC_MASTER_CSV)
            self.df_mimic = self.df_mimic.loc[:, MIMIC_USED_COLS]
            self.df_mimic[MIMIC_PATH_COL] = self.df_mimic[MIMIC_PATH_COL].apply(
                lambda x: os.path.join(MIMIC_IMG_DIR, x)
            )
            self.df_mimic.rename(columns={
                MIMIC_REPORT_COL:     PRETRAIN_REPORT_COL,
                MIMIC_SPLIT_COL:      PRETRAIN_SPLIT_COL,
                MIMIC_VIEW_COL:       PRETRAIN_VIEW_COL,
                MIMIC_PATH_COL:       PRETRAIN_PATH_COL,
                MIMIC_IMPRESSION_COL: PRETRAIN_IMPRESSION_COL,
            }, inplace=True)
            self.df = pd.concat([self.df, self.df_mimic])

        self.filenames, self.path2sent, self.impression_num = self.load_text_data(
            split, self.cfg.data.view, "_".join(self.cfg.data.dataset)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.split = split

        # Ablation only: replace the raw report with a RadGraph entity string
        # such as "opacity; right lower lobe; no pleural effusion".
        self.kg_builder = None
        use_kg = getattr(getattr(cfg.model, 'text', None), 'use_kg_sentences', False)
        if use_kg and _KG_BUILDER_AVAILABLE:
            radgraph_path = getattr(cfg.data, 'radgraph_path', None)
            if radgraph_path:
                max_ents = getattr(cfg.model.text, 'kg_max_entities', 64)
                self.kg_builder = RadGraphKGBuilder(
                    radgraph_path=radgraph_path,
                    max_entities=max_ents,
                    fallback_to_raw=True,
                )
                print("[Dataset] KG-sentences enabled")
            else:
                print("[Dataset] WARNING: use_kg_sentences=true but data.radgraph_path not set")
        else:
            print("[Dataset] using raw report sentences")

    @staticmethod
    def extract_study_path(img_path: str) -> str:
        """
        Extract normalised RadGraph study path from a MIMIC image path.

        Confirmed RadGraph JSON key format: 'p10/p10659857/s59206984.txt'
        Normalised (index key):             'p10/p10659857/s59206984'

        MIMIC image path:
            .../files/p10/p10659857/s59206984/abc123.jpg
                         ↑─────────────────────┘
                         extract this segment

        Args:
            img_path (str): Full MIMIC image path from the dataset CSV.

        Returns:
            str: Normalised study path e.g. 'p10/p10659857/s59206984'.
                 Returns img_path unchanged if pattern not found (safe fallback —
                 RadGraphInjector marks the sample as unavailable and uses raw
                 text_emb_s instead).
        """
        # Match: p{digits}/p{digits}/s{digits}
        match = re.search(r'(p\d+/p\d+/s\d+)', img_path)
        if match:
            return match.group(1)
        return img_path

    def precess_missing_values(self, mode=None):
        """
        Impute missing values in the df.

        Args:
            mode (int): Mode for handling missing values.
        """
        if mode == 0:
            print("missing_values mode 0")
            for col in TASKS:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, 1, inplace=True)
                    self.df[col].replace(-2, 0, inplace=True)
                else:
                    self.df[col].replace(-1, 0, inplace=True)
                    self.df[col].replace(-2, 0, inplace=True)
        elif mode == 1:
            print("missing_values mode 1")
            for col in TASKS:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].replace(-2, 0, inplace=True)
        elif mode == 2:
            print("missing_values mode 2")
            for col in TASKS:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].replace(-2, 0, inplace=True)
        else:
            raise NotImplementedError
        self.df[TASKS] = self.df[TASKS].astype(float)

    def load_text_data(self, split, view, dataset):
        """
        Load text data for a given split, view, and dataset.

        Args:
            split (str):   'train', 'val', or 'test'.
            view (str):    'full', 'frontal', or 'lateral'.
            dataset (str): Dataset name.

        Returns:
            filenames (list):       List of image paths.
            path2sent (dict):       Path → list of sentence strings.
            impression_num (dict):  Number of impression sentences per path.
        """
        filepath = os.path.join(
            "./", "captions_{}_{}_{}.pickle".format(PICKLE_SUFFIX, dataset, view)
        )
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exist. Creating captions...")
            path2sent, to_remove, impression_num = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, impression_num], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, impression_num = pickle.load(f)

        if view == "full":
            filenames = self.df[self.df[PRETRAIN_SPLIT_COL] == split][PRETRAIN_PATH_COL]
            print("full view")
        else:
            filenames = self.df[
                (self.df[PRETRAIN_SPLIT_COL] == split)
                & (self.df[PRETRAIN_VIEW_COL] == view)
            ][PRETRAIN_PATH_COL]

        filenames = filenames[~filenames.isin(to_remove)].to_list()
        return filenames, path2sent, impression_num

    def get_caption(self, path, study_id=None):
        """
        Sample and tokenize a caption for one image.

        With full_report + random_combine (the training setting) a random subset
        of the report's sentences is joined in random order.

        Returns:
            tokens (dict): HuggingFace tokenizer output tensors.
            x_len (int):   number of non-padding tokens.
        """
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            if self.cfg.data.text.random_combine is True:
                if len(series_sents) < 2:
                    sent = series_sents[0]
                else:
                    if self.cfg.data.text.random_combine_mode == 1:
                        sample_num = len(series_sents)
                        if self.cfg.data.text.random_num is True:
                            sample_num = random.randint(1, len(series_sents))
                        sent_ix = random.choice(
                            range(len(series_sents)), sample_num, replace=False
                        )
                        sent = ". ".join([series_sents[ix] for ix in sent_ix])
                    else:
                        findings_num = len(series_sents) - self.impression_num[path]
                        if findings_num == 0:
                            sent = ". ".join(series_sents)
                        else:
                            sample_num = findings_num
                            if self.cfg.data.text.random_num is True:
                                sample_num = random.randint(1, findings_num + 1)
                            sent_ix = random.choice(findings_num, sample_num, replace=False)
                            findings = [series_sents[ix] for ix in sent_ix]
                            impression = (
                                series_sents[-self.impression_num[path]:]
                                if self.impression_num[path] != 0
                                else []
                            )
                            sent = ". ".join(findings + impression)
            else:
                sent = ". ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        if self.kg_builder is not None and study_id is not None:
            sent = self.kg_builder.get_kg_sentence(study_id, raw_report=sent)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        return tokens, x_len

    def equalize_hist(self, data, use_mask=True):
        """Apply histogram equalisation to image array."""
        mask = (np.array(data) != 0) if use_mask else None
        img = np.array(data) / 255.
        img = exposure.equalize_hist(img, mask=mask)
        img = (255 * img).astype(np.uint8)
        return img

    def read_img(self, img_path):
        """Read image and apply histogram equalisation."""
        data = PIL.Image.open(img_path)
        img = self.equalize_hist(data, use_mask=True)
        img = PIL.Image.fromarray(img).convert('RGB')
        return img

    def get_imgs(self, img_path, transform=None):
        """
        Load image from path with optional transform.

        Args:
            img_path (str):   Path to image file.
            transform:        torchvision transform pipeline.

        Returns:
            img (Tensor | PIL.Image): Processed image.
        """
        if "equalize_hist" in self.cfg.data.image and self.cfg.data.image.equalize_hist:
            img = self.read_img(img_path)
        else:
            x = cv2.imread(str(img_path), 0)
            img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, index):
        """
        Returns (image, caption tokens, caption length, image path, labels,
        RadGraph study key).
        """
        key = self.filenames[index]

        imgs  = self.get_imgs(key, self.transform)
        label = torch.tensor(
            self.df[self.df[PRETRAIN_PATH_COL] == key][MIMIC_TASKS].values[0]
        )

        study_id = self.extract_study_path(key)
        caps, cap_len = self.get_caption(key, study_id=study_id)

        return imgs, caps, cap_len, key, label, study_id

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):
        """
        Build path → sentences mapping from the dataframe.

        Args:
            df (DataFrame):    Source dataframe.
            max_word_num (int): Max cumulative token count per report.

        Returns:
            path2sent (dict):      Path → list of sentence strings.
            to_remove (list):      Paths with no usable sentences.
            impression_num (dict): Impression sentence count per path.
        """
        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        impression_num = {}

        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            captions = ""
            if type(row[PRETRAIN_REPORT_COL]) == str:
                captions += row[PRETRAIN_REPORT_COL]
            impression = ""
            if type(row[PRETRAIN_IMPRESSION_COL]) == str:
                impression += row[PRETRAIN_IMPRESSION_COL]

            if len(captions) == 0:
                to_remove.append(row[PRETRAIN_PATH_COL])

            captions = captions.replace("\n", " ")

            splitter = re.compile(r"[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]
            impression = impression.rstrip(".")
            impression = splitter.split(impression)
            impression = [point.split(".") for point in impression]
            impression = [sent for point in impression for sent in point]

            cnt = 0
            study_sent = []
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) <= 1:
                    continue
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break
                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            study_impression = []
            for cap in impression:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) <= 1:
                    continue
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_impression.append(" ".join(included_tokens))

            if len(study_sent) > 0:
                path2sent[row[PRETRAIN_PATH_COL]] = study_sent
                if len(study_impression) <= len(study_sent):
                    impression_num[row[PRETRAIN_PATH_COL]] = len(study_impression)
                else:
                    impression_num[row[PRETRAIN_PATH_COL]] = len(study_sent)
            else:
                to_remove.append(row[PRETRAIN_PATH_COL])

        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean():.2f},{sent_lens.max()} "
            f"[{np.percentile(sent_lens, 5):.1f}, {np.percentile(sent_lens, 95):.1f}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean():.2f},{num_sents.max()} "
            f"[{np.percentile(num_sents, 5):.1f}, {np.percentile(num_sents, 95):.1f}]"
        )

        return path2sent, to_remove, impression_num

    def _resize_img(self, img, scale):
        """
        Resize and zero-pad image to scale × scale.

        Args:
            img (np.ndarray): Image loaded via cv2.
            scale (int):      Target side length in pixels.

        Returns:
            resized_img (np.ndarray): Padded image [scale, scale].
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        if max_ind == 0:
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)

        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )

        if max_ind == 0:
            pad_size = scale - resized_img.shape[1]
            left, right = int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))
            top, bottom = 0, 0
        else:
            pad_size = scale - resized_img.shape[0]
            top, bottom = int(np.floor(pad_size / 2)), int(np.ceil(pad_size / 2))
            left, right = 0, 0

        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)],
            "constant", constant_values=0
        )
        return resized_img


# --------------------------------------------------------------------------- #
# Collate function                                                              #
# --------------------------------------------------------------------------- #

def multimodal_collate_fn(batch, knowledge_injector=None):
    """
    Stack a batch, sort it by caption length (as AFLoc does) and, when a
    knowledge injector is given, attach the RadGraph entity graphs for L_SG.

    Returns a dict with caption_ids, token_type_ids, attention_mask, imgs,
    cap_lens, path, labels, study_ids and graph_data (None without injection).
    """
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path, labels, study_ids = [], [], []

    for b in batch:
        img, cap, cap_l, p, label, study_id = b

        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        labels.append(label)
        study_ids.append(study_id)

    imgs      = torch.stack(imgs)
    ids       = torch.stack(ids).squeeze()
    tokens    = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    labels    = torch.stack(labels)

    # sort by caption length, longest first
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True
    )
    # study_ids must be reordered to match sorted batch
    study_ids_sorted = [study_ids[i] for i in sorted_cap_indices.tolist()]

    if knowledge_injector is not None:
        graph_data = knowledge_injector.get_graph_features(
            report_ids=study_ids_sorted,
            device=imgs.device,
        )
        graph_data['study_ids'] = study_ids_sorted
    else:
        graph_data = None

    return {
        "caption_ids":    ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs":           imgs[sorted_cap_indices],
        "cap_lens":       sorted_cap_lens,
        "path":           path,
        "labels":         labels[sorted_cap_indices],
        "study_ids":      study_ids_sorted,
        "graph_data":     graph_data,
    }
