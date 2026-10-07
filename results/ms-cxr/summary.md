# MS-CXR: 1162 image-phrase pairs

Category-weighted means with 95% bootstrap intervals. BEP = no inference rule; RQP = RadGraph Query Pruning; DHS = Dynamic Heatmap Sharpening. MedKLIP does not read the phrase, so RQP is undefined for it.

| model | stage | mIoU | Dice | CNR | P@1 |
|---|---|---|---|---|---|
| MedKLIP | BEP | 0.1871 [0.1795, 0.1937] | 0.2935 [0.2833, 0.3025] | 0.952 [0.919, 0.986] | 0.4121 [0.3779, 0.4445] |
| MedKLIP | +DHS | 0.2010 [0.1936, 0.2076] | 0.3128 [0.3026, 0.3221] | 0.639 [0.602, 0.682] | 0.4121 [0.3779, 0.4445] |
| GLoRIA | BEP | 0.2424 [0.2324, 0.2520] | 0.3638 [0.3510, 0.3759] | 1.280 [1.225, 1.341] | 0.6561 [0.6226, 0.6875] |
| GLoRIA | +RQP | 0.2425 [0.2324, 0.2520] | 0.3638 [0.3505, 0.3762] | 1.273 [1.213, 1.333] | 0.6541 [0.6206, 0.6840] |
| GLoRIA | +DHS | 0.2899 [0.2795, 0.3005] | 0.4234 [0.4105, 0.4364] | 1.170 [1.105, 1.237] | 0.6561 [0.6226, 0.6875] |
| GLoRIA | +RQP+DHS | 0.2879 [0.2769, 0.2986] | 0.4208 [0.4068, 0.4339] | 1.153 [1.089, 1.223] | 0.6541 [0.6206, 0.6840] |
| AFLoc | BEP | 0.3232 [0.3136, 0.3323] | 0.4610 [0.4499, 0.4717] | 1.631 [1.578, 1.682] | 0.7843 [0.7581, 0.8059] |
| AFLoc | +RQP | 0.3244 [0.3150, 0.3338] | 0.4626 [0.4514, 0.4733] | 1.640 [1.589, 1.692] | 0.7813 [0.7554, 0.8029] |
| AFLoc | +DHS | 0.3410 [0.3306, 0.3507] | 0.4810 [0.4691, 0.4919] | 1.535 [1.478, 1.596] | 0.7843 [0.7581, 0.8059] |
| AFLoc | +RQP+DHS | 0.3417 [0.3314, 0.3515] | 0.4819 [0.4699, 0.4931] | 1.533 [1.477, 1.597] | 0.7813 [0.7554, 0.8029] |
| RAD-DINO control (no L_SG) | BEP | 0.3350 [0.3258, 0.3449] | 0.4774 [0.4668, 0.4885] | 1.648 [1.605, 1.691] | 0.8242 [0.7992, 0.8449] |
| RAD-DINO control (no L_SG) | +RQP | 0.3375 [0.3287, 0.3471] | 0.4810 [0.4708, 0.4917] | 1.677 [1.635, 1.720] | 0.8238 [0.7982, 0.8449] |
| RAD-DINO control (no L_SG) | +DHS | 0.3616 [0.3509, 0.3720] | 0.5068 [0.4948, 0.5183] | 1.618 [1.559, 1.676] | 0.8242 [0.7992, 0.8449] |
| RAD-DINO control (no L_SG) | +RQP+DHS | 0.3642 [0.3536, 0.3744] | 0.5104 [0.4982, 0.5215] | 1.645 [1.589, 1.703] | 0.8238 [0.7982, 0.8449] |
| KAF-Ground | BEP | 0.3379 [0.3289, 0.3467] | 0.4817 [0.4715, 0.4917] | 1.672 [1.628, 1.714] | 0.8194 [0.7931, 0.8420] |
| KAF-Ground | +RQP | 0.3435 [0.3347, 0.3529] | 0.4883 [0.4781, 0.4989] | 1.703 [1.661, 1.746] | 0.8382 [0.8130, 0.8612] |
| KAF-Ground | +DHS | 0.3637 [0.3538, 0.3733] | 0.5100 [0.4987, 0.5205] | 1.638 [1.579, 1.697] | 0.8194 [0.7931, 0.8420] |
| KAF-Ground | +RQP+DHS | 0.3672 [0.3574, 0.3771] | 0.5142 [0.5029, 0.5249] | 1.667 [1.609, 1.726] | 0.8382 [0.8130, 0.8612] |

## Paired differences in mIoU

| contrast | delta | 95% CI | p |
|---|---|---|---|
| MedKLIP: +DHS vs BEP | +0.0140 | [+0.0108, +0.0173] | 0.001 |
| GLoRIA: +RQP vs BEP | +0.0001 | [-0.0044, +0.0045] | 0.990 |
| GLoRIA: +DHS vs BEP | +0.0475 | [+0.0389, +0.0545] | 0.001 |
| GLoRIA: +RQP+DHS vs BEP | +0.0454 | [+0.0359, +0.0541] | 0.001 |
| GLoRIA: +RQP+DHS vs +RQP | +0.0454 | [+0.0371, +0.0526] | 0.001 |
| AFLoc: +RQP vs BEP | +0.0012 | [-0.0011, +0.0036] | 0.332 |
| AFLoc: +DHS vs BEP | +0.0177 | [+0.0107, +0.0240] | 0.001 |
| AFLoc: +RQP+DHS vs BEP | +0.0185 | [+0.0113, +0.0255] | 0.001 |
| AFLoc: +RQP+DHS vs +RQP | +0.0173 | [+0.0101, +0.0241] | 0.001 |
| RAD-DINO control (no L_SG): +RQP vs BEP | +0.0025 | [-0.0001, +0.0056] | 0.074 |
| RAD-DINO control (no L_SG): +DHS vs BEP | +0.0265 | [+0.0179, +0.0344] | 0.001 |
| RAD-DINO control (no L_SG): +RQP+DHS vs BEP | +0.0292 | [+0.0205, +0.0371] | 0.001 |
| RAD-DINO control (no L_SG): +RQP+DHS vs +RQP | +0.0267 | [+0.0175, +0.0342] | 0.001 |
| KAF-Ground: +RQP vs BEP | +0.0056 | [+0.0033, +0.0083] | 0.001 |
| KAF-Ground: +DHS vs BEP | +0.0258 | [+0.0177, +0.0333] | 0.001 |
| KAF-Ground: +RQP+DHS vs BEP | +0.0293 | [+0.0211, +0.0373] | 0.001 |
| KAF-Ground: +RQP+DHS vs +RQP | +0.0237 | [+0.0155, +0.0312] | 0.001 |
| KAF-Ground vs MedKLIP @ BEP | +0.1508 | [+0.1413, +0.1618] | 0.001 |
| KAF-Ground vs MedKLIP @ +DHS | +0.1626 | [+0.1548, +0.1715] | 0.001 |
| KAF-Ground vs MedKLIP @ best vs best | +0.1662 | [+0.1586, +0.1751] | 0.001 |
| KAF-Ground vs GLoRIA @ BEP | +0.0955 | [+0.0860, +0.1058] | 0.001 |
| KAF-Ground vs GLoRIA @ +RQP | +0.1010 | [+0.0919, +0.1112] | 0.001 |
| KAF-Ground vs GLoRIA @ +DHS | +0.0737 | [+0.0655, +0.0833] | 0.001 |
| KAF-Ground vs GLoRIA @ +RQP+DHS | +0.0794 | [+0.0717, +0.0889] | 0.001 |
| KAF-Ground vs GLoRIA @ best vs best | +0.0794 | [+0.0717, +0.0889] | 0.001 |
| KAF-Ground vs AFLoc @ BEP | +0.0147 | [+0.0072, +0.0231] | 0.001 |
| KAF-Ground vs AFLoc @ +RQP | +0.0191 | [+0.0121, +0.0275] | 0.001 |
| KAF-Ground vs AFLoc @ +DHS | +0.0227 | [+0.0168, +0.0292] | 0.001 |
| KAF-Ground vs AFLoc @ +RQP+DHS | +0.0255 | [+0.0194, +0.0321] | 0.001 |
| KAF-Ground vs AFLoc @ best vs best | +0.0255 | [+0.0194, +0.0321] | 0.001 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ BEP | +0.0029 | [-0.0021, +0.0081] | 0.250 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +RQP | +0.0060 | [+0.0014, +0.0110] | 0.008 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +DHS | +0.0021 | [-0.0017, +0.0063] | 0.288 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +RQP+DHS | +0.0030 | [-0.0006, +0.0069] | 0.096 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ best vs best | +0.0030 | [-0.0006, +0.0069] | 0.096 |
| RAD-DINO control vs AFLoc @ BEP | +0.0118 | [+0.0041, +0.0198] | 0.002 |
| RAD-DINO control vs AFLoc @ +RQP | +0.0131 | [+0.0061, +0.0211] | 0.002 |
| RAD-DINO control vs AFLoc @ +DHS | +0.0206 | [+0.0142, +0.0273] | 0.001 |
| RAD-DINO control vs AFLoc @ +RQP+DHS | +0.0225 | [+0.0160, +0.0290] | 0.001 |
| KAF-Ground: RQP x DHS interaction | -0.0021 | [-0.0042, -0.0000] | 0.048 |

## IoU per finding type (BEP and full protocol)

**BEP**

| finding | n | MedKLIP | GLoRIA | AFLoc | RAD-DINO control (no L_SG) | KAF-Ground |
|---|---|---|---|---|---|---|
| Atelectasis | 61 | 0.1687 | 0.3064 | 0.3816 | 0.3969 | 0.3985 |
| Cardiomegaly | 333 | 0.3458 | 0.2869 | 0.3615 | 0.3235 | 0.3112 |
| Consolidation | 117 | 0.2131 | 0.2625 | 0.3708 | 0.4004 | 0.4016 |
| Edema | 46 | 0.2477 | 0.2367 | 0.2602 | 0.2798 | 0.2912 |
| Lung Opacity | 82 | 0.1578 | 0.2288 | 0.3277 | 0.3228 | 0.3524 |
| Pleural Effusion | 96 | 0.1043 | 0.2176 | 0.3479 | 0.3727 | 0.3577 |
| Pneumonia | 182 | 0.2130 | 0.2890 | 0.4160 | 0.3972 | 0.4160 |
| Pneumothorax | 245 | 0.0460 | 0.1114 | 0.1202 | 0.1867 | 0.1746 |

**+RQP+DHS** (MedKLIP: +DHS)

| finding | n | MedKLIP | GLoRIA | AFLoc | RAD-DINO control (no L_SG) | KAF-Ground |
|---|---|---|---|---|---|---|
| Atelectasis | 61 | 0.1902 | 0.3405 | 0.3954 | 0.4388 | 0.4329 |
| Cardiomegaly | 333 | 0.3846 | 0.3945 | 0.4871 | 0.4247 | 0.4515 |
| Consolidation | 117 | 0.2195 | 0.3061 | 0.3762 | 0.4162 | 0.4086 |
| Edema | 46 | 0.2656 | 0.2870 | 0.2938 | 0.3388 | 0.3455 |
| Lung Opacity | 82 | 0.1608 | 0.2425 | 0.2938 | 0.3050 | 0.3174 |
| Pleural Effusion | 96 | 0.1101 | 0.2375 | 0.3322 | 0.3562 | 0.3396 |
| Pneumonia | 182 | 0.2273 | 0.3439 | 0.4209 | 0.4395 | 0.4549 |
| Pneumothorax | 245 | 0.0502 | 0.1508 | 0.1342 | 0.1948 | 0.1875 |

## Complete failures (IoU = 0) under BEP

AFLoc misses 63 pairs, KAF-Ground 5; 59 recovered, 1 lost (exact McNemar p = 1.1e-16).

| finding | n | AFLoc | KAF-Ground | recovered | lost |
|---|---|---|---|---|---|
| Atelectasis | 61 | 1 | 1 | 0 | 0 |
| Cardiomegaly | 333 | 0 | 0 | 0 | 0 |
| Consolidation | 117 | 1 | 0 | 1 | 0 |
| Edema | 46 | 0 | 0 | 0 | 0 |
| Lung Opacity | 82 | 0 | 0 | 0 | 0 |
| Pleural Effusion | 96 | 0 | 0 | 0 | 0 |
| Pneumonia | 182 | 2 | 2 | 1 | 1 |
| Pneumothorax | 245 | 59 | 2 | 57 | 0 |
