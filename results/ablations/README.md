# Ablations

All numbers are category-weighted mIoU on MS-CXR (1,162 pairs) without the
inference rules (BEP) unless stated otherwise.

## Training side (`training_ablations.csv`)

| model | graph supervision | mIoU |
|---|---|---|
| AFLoc, released checkpoint (ResNet-50) | none | 0.3232 |
| RAD-DINO, ClinicalBERT frozen | none | 0.334 |
| RAD-DINO, ClinicalBERT trainable (23 epochs) | none | 0.334 |
| **RAD-DINO control**: KAF-Ground recipe, graph branch off | none | 0.3350 |
| KAF-Ground, earlier checkpoint (16 epochs) | L_SG, deep head | 0.338 |
| **KAF-Ground** (reported model) | L_SG, deep head, sigma 1 | **0.3379** |
| Fusion gate able to open (sigma 3, 8 heads) | L_SG + gated fusion | 0.326 |
| L_SG on the shallow head (the one read at inference) | L_SG, shallow head | 0.279 |
| AFLoc + GAT dual-stream, trained end to end | GAT fusion | 0.298 * |
| AFLoc + GAT dual-stream, frozen encoders, slow lr | GAT fusion | 0.317 * |
| AFLoc + GAT + L_SG, CNN frozen, sigma 5 | L_SG | 0.324 * |
| AFLoc + GAT + L_SG, CNN and BERT trainable / frozen | L_SG | 0.321 * |
| AFLoc + KG-rewritten report sentences (with or without raw text) | KG sentences | 0.322 * |
| AFLoc + injected entity definitions at inference | definitions | 0.330 * |
| AFLoc + definition loss (AFLoc-KG) | definition loss | 0.331 * |

\* ResNet-50 runs from an earlier code base; values as reported at the time,
checkpoints not retained, so no confidence intervals.

The RAD-DINO control isolates the graph loss. Paired bootstrap (the same
procedure as the main tables; full rows in `../*/tables/contrasts.csv`):

| contrast (mIoU) | MS-CXR | PadChest-GR |
|---|---|---|
| RAD-DINO control - AFLoc, BEP | +0.0118 [+0.0041, +0.0198] | +0.0233 [+0.0121, +0.0359] |
| KAF-Ground - RAD-DINO control, BEP | +0.0029 [-0.0021, +0.0081] | -0.0078 [-0.0147, -0.0010] |
| KAF-Ground - RAD-DINO control, +RQP+DHS | +0.0030 [-0.0006, +0.0069] | -0.0028 [-0.0081, +0.0022] |

The configurations for the RAD-DINO rows that can be trained with this code are
in `configs/ablations/`:

```bash
python train.py -c configs/ablations/raddino_control.yaml   --train --gpus 0,1 --val_check_interval 0.5
python train.py -c configs/ablations/fusion_gate_open.yaml  --train --gpus 0,1 --val_check_interval 0.5
python train.py -c configs/ablations/lsg_shallow_head.yaml  --train --gpus 0,1 --val_check_interval 0.5
```

The KG-rewritten-sentence ablation is `model.text.use_kg_sentences: true`
(`afloc/knowledge/kg_sentence_builder.py`).

## Inference side (`inference_ablations.csv`)

Every variant is applied to the KAF-Ground checkpoint and compared with the
same checkpoint read out the default way (shallow head, original phrase, one
view: 0.338); the last two rows use the fusion-gate variant and compare with
its own default read-out (0.326). Deltas and 95% CIs come from AFLoc's
evaluation bootstrap (1,000 replicates, the same resamples in every run).

| variant | mIoU | delta | 95% CI |
|---|---|---|---|
| RQP (original + pruned + template), the paper's setting | 0.344 | +0.0057 | [+0.0044, +0.0070] |
| RQP + mean of both heads | 0.344 | +0.0058 | [+0.0036, +0.0080] |
| RQP pruned phrase only | 0.340 | +0.0023 | [+0.0006, +0.0039] |
| hand-written qualifier list, ensemble | 0.342 | +0.0036 | [+0.0025, +0.0048] |
| hand-written qualifier list, pruned only | 0.339 | +0.0010 | [-0.0003, +0.0025] |
| RadGraph tokens: max over anatomy + finding words | 0.338 | -0.0004 | [-0.0012, +0.0004] |
| RadGraph tokens: mean over anatomy + finding words | 0.334 | -0.0042 | [-0.0064, -0.0023] |
| RadGraph tokens: anatomy words only | 0.334 | -0.0042 | [-0.0061, -0.0024] |
| deep head (l2) | 0.328 | -0.0094 | [-0.0124, -0.0063] |
| mean of both heads (lf) | 0.340 | +0.0025 | [+0.0003, +0.0047] |
| TTA horizontal flip | 0.316 | -0.0219 | [-0.0240, -0.0197] |
| TTA multi-scale (448/518/616) | 0.338 | +0.0006 | [+0.0001, +0.0010] |
| TTA flip + multi-scale | 0.317 | -0.0210 | [-0.0232, -0.0187] |

Horizontal flipping hurts because MS-CXR phrases carry laterality.

Each variant is one `dump_heatmaps.py` call; the options are in the last
column of the csv, for example

```bash
python scripts/dump_heatmaps.py --model kaf-ground --ckpt checkpoints/kaf_ground.ckpt \
    --dataset ms-cxr --views id@518,hflip@518 --out runs/heatmaps/tta_hflip.npy
python scripts/score_heatmaps.py --dataset ms-cxr --heatmaps runs/heatmaps/tta_hflip.npy \
    --out runs/scores/tta_hflip.csv
```

The token-role variants need the role table written by
`scripts/build_rqp_lexicon.py --roles`.
