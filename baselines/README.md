# Baselines

All systems are scored by the same code (`scripts/score_heatmaps.py`) against
the same 518 x 518 ground truth. Each one is read out the way it was trained.

| system | read-out | input size | RQP |
|---|---|---|---|
| AFLoc (released checkpoint) | report-level embedding, raw dot product per patch | 224 | original + pruned + template, averaged |
| GLoRIA (chexpert_resnet50) | word-level: max over words of the patch-word dot product | 224 | pruned phrase only |
| MedKLIP | attention map of the entity that matches the finding type | 224 | undefined (no phrase input) |
| KAF-Ground | word-level: max over words of the patch-word cosine | 518 | original + pruned + template, averaged |

AFLoc runs through `kafground.inference.engine.AFLocEngine` with this
repository's `afloc` package (it reproduces the original AFLoc inference code,
including its image preprocessing and tokenisation). GLoRIA and MedKLIP need
their official code bases and checkpoints:

```bash
git clone https://github.com/marshuang80/gloria third_party/gloria
git clone https://github.com/MediaBrain-SJTU/MedKLIP third_party/MedKLIP
export GLORIA_CODE=$PWD/third_party/gloria
export MEDKLIP_CODE=$PWD/third_party/MedKLIP/Sample_Zero-Shot_Grounding_RSNA

python scripts/dump_heatmaps.py --model gloria  --ckpt <gloria>/chexpert_resnet50.ckpt --dataset ms-cxr --rqp off --out runs/heatmaps/gloria_ms-cxr_rqp-off.npy
python scripts/dump_heatmaps.py --model medklip --ckpt <medklip>/checkpoint_final.pth  --dataset ms-cxr --out runs/heatmaps/medklip_ms-cxr_rqp-off.npy
```

`_compat.py` stubs two training-only imports of those packages
(`albumentations`, `segmentation_models_pytorch`) that are never used for
grounding and do not install cleanly next to recent PyTorch.
