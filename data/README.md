# Data

None of the datasets can be redistributed; all of them need a (free) data-use
agreement. This folder only holds the scripts that turn the official releases
into the layout the code expects.

| dataset | used for | source |
|---|---|---|
| MIMIC-CXR-JPG 2.0.0 + reports | training | PhysioNet |
| RadGraph 1.0.0 (`MIMIC-CXR_graphs.json`) | L_SG graphs, RQP lexicon | PhysioNet |
| MS-CXR 1.1.0 (`MS_CXR_Local_Alignment_v1.1.0.json`) | evaluation (1,162 pairs) | PhysioNet |
| PadChest-GR | transfer evaluation (530 pairs) | BIMCV |

## 1. Images (MIMIC-CXR and MS-CXR)

All images are stored as 518 x 518 grayscale JPEGs: long side resized with
INTER_AREA, short side zero-padded at the centre. MS-CXR images are MIMIC-CXR
images, so one copy serves both.

```bash
python data/resize_pad.py --src <mimic-cxr-jpg>/files --dst data/mimic-cxr-jpg --size 518
export KAF_MIMIC_IMG_DIR=$PWD/data/mimic-cxr-jpg      # training
export KAF_MSCXR_IMG_DIR=$PWD/data/mimic-cxr-jpg      # MS-CXR evaluation
export KAF_MSCXR_JSON=<ms-cxr>/MS_CXR_Local_Alignment_v1.1.0.json
```

## 2. Training table (MIMIC-CXR)

`KAF_MIMIC_CSV` points to one csv with a row per image and these columns:

| column | content |
|---|---|
| `path` | image path relative to `KAF_MIMIC_IMG_DIR`, e.g. `p10/p10000032/s50414267/<dicom_id>.jpg` |
| `view` | `Frontal` or `Lateral` (only frontal images are used) |
| `split_with_MS` | `train` / `validate` / `test`, the official MIMIC-CXR-JPG split |
| `report` | findings and impression sections joined with a space |
| `impression` | impression section |
| 14 CheXpert labels | `No Finding`, `Enlarged Cardiomediastinum`, ..., `Support Devices` (from `mimic-cxr-2.0.0-chexpert.csv`) |

The MS-CXR images themselves are removed from this table; other images of the
same studies stay in it. Sentences are cached in
`captions_report_mimic_Frontal.pickle` on the first run.

## 3. RadGraph

Set `data.radgraph_path` in the config to `MIMIC-CXR_graphs.json`. To
regenerate the RQP lexicon (the shipped `kafground/inference/rqp_modifiers.txt`
was produced this way):

```bash
python scripts/build_rqp_lexicon.py --radgraph <radgraph>/MIMIC-CXR_graphs.json \
    --exclude $KAF_MSCXR_JSON --out kafground/inference/rqp_modifiers.txt \
    --roles runs/rqp_roles.json          # role table for the token ablations
```

## 4. PadChest-GR

```bash
python data/padchest/prepare_padchest_gr.py --src <padchest-gr>/Padchest_GR_files \
    --json <padchest-gr>/grounded_reports_20240819.json --csv <padchest-gr>/master_table.csv \
    --out data/padchest/padchest_gr_prepared --sizes 518
python data/padchest/build_benchmark.py --prepared data/padchest/padchest_gr_prepared \
    --out data/padchest/bench_testsplit.csv
export KAF_PADCHEST_DIR=$PWD/data/padchest/padchest_gr_prepared
export KAF_PADCHEST_BENCH=$PWD/data/padchest/bench_testsplit.csv
```

The benchmark is PadChest-GR's own test split restricted to eight finding
types (530 pairs on 359 images): cardiomegaly, pleural effusion and
atelectasis (shared with MS-CXR), alveolar and interstitial pattern
(near-analogues), and nodule, pleural thickening and aortic elongation (not in
MS-CXR). MedKLIP's entity for each type is listed in
`baselines/medklip_engine.py`.

## Pair ids

Released per-sample results identify a pair by
`sha1("<image file name>|<phrase>")[:16]` (see `kafground.eval.datasets.pair_id`),
so they carry no dataset text but can be joined back to a local copy.
