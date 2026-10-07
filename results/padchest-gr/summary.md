# PadChest-GR (test split): 530 image-phrase pairs

Category-weighted means with 95% bootstrap intervals. BEP = no inference rule; RQP = RadGraph Query Pruning; DHS = Dynamic Heatmap Sharpening. MedKLIP does not read the phrase, so RQP is undefined for it.

| model | stage | mIoU | Dice | CNR | P@1 |
|---|---|---|---|---|---|
| MedKLIP | BEP | 0.1847 [0.1749, 0.1949] | 0.2763 [0.2633, 0.2898] | 0.828 [0.772, 0.888] | 0.3921 [0.3560, 0.4294] |
| MedKLIP | +DHS | 0.1959 [0.1854, 0.2066] | 0.2899 [0.2756, 0.3038] | 0.526 [0.469, 0.585] | 0.3921 [0.3560, 0.4294] |
| GLoRIA | BEP | 0.2080 [0.1963, 0.2219] | 0.3108 [0.2949, 0.3298] | 1.220 [1.133, 1.322] | 0.5556 [0.5154, 0.5970] |
| GLoRIA | +RQP | 0.2071 [0.1947, 0.2213] | 0.3089 [0.2926, 0.3270] | 1.201 [1.114, 1.298] | 0.5433 [0.5021, 0.5836] |
| GLoRIA | +DHS | 0.2444 [0.2304, 0.2588] | 0.3531 [0.3353, 0.3723] | 1.105 [1.016, 1.213] | 0.5556 [0.5154, 0.5970] |
| GLoRIA | +RQP+DHS | 0.2434 [0.2299, 0.2577] | 0.3514 [0.3335, 0.3707] | 1.085 [0.993, 1.191] | 0.5433 [0.5021, 0.5836] |
| AFLoc | BEP | 0.2738 [0.2600, 0.2877] | 0.3888 [0.3721, 0.4049] | 1.412 [1.335, 1.491] | 0.6241 [0.5862, 0.6590] |
| AFLoc | +RQP | 0.2822 [0.2684, 0.2959] | 0.3990 [0.3824, 0.4160] | 1.450 [1.371, 1.529] | 0.6479 [0.6113, 0.6828] |
| AFLoc | +DHS | 0.2789 [0.2652, 0.2939] | 0.3924 [0.3754, 0.4107] | 1.296 [1.214, 1.377] | 0.6241 [0.5862, 0.6590] |
| AFLoc | +RQP+DHS | 0.2919 [0.2778, 0.3065] | 0.4075 [0.3909, 0.4250] | 1.339 [1.254, 1.421] | 0.6479 [0.6113, 0.6828] |
| RAD-DINO control (no L_SG) | BEP | 0.2972 [0.2833, 0.3101] | 0.4243 [0.4076, 0.4399] | 1.588 [1.528, 1.654] | 0.7643 [0.7303, 0.7981] |
| RAD-DINO control (no L_SG) | +RQP | 0.2998 [0.2861, 0.3130] | 0.4285 [0.4122, 0.4441] | 1.623 [1.563, 1.686] | 0.7573 [0.7218, 0.7924] |
| RAD-DINO control (no L_SG) | +DHS | 0.3064 [0.2916, 0.3216] | 0.4311 [0.4136, 0.4498] | 1.553 [1.481, 1.630] | 0.7643 [0.7303, 0.7981] |
| RAD-DINO control (no L_SG) | +RQP+DHS | 0.3110 [0.2965, 0.3263] | 0.4371 [0.4201, 0.4558] | 1.586 [1.512, 1.663] | 0.7573 [0.7218, 0.7924] |
| KAF-Ground | BEP | 0.2894 [0.2764, 0.3035] | 0.4181 [0.4030, 0.4347] | 1.572 [1.514, 1.634] | 0.7661 [0.7333, 0.8026] |
| KAF-Ground | +RQP | 0.2908 [0.2775, 0.3044] | 0.4204 [0.4050, 0.4370] | 1.594 [1.540, 1.654] | 0.7800 [0.7439, 0.8172] |
| KAF-Ground | +DHS | 0.3034 [0.2891, 0.3185] | 0.4304 [0.4141, 0.4483] | 1.555 [1.485, 1.635] | 0.7661 [0.7333, 0.8026] |
| KAF-Ground | +RQP+DHS | 0.3082 [0.2939, 0.3239] | 0.4359 [0.4195, 0.4535] | 1.580 [1.511, 1.655] | 0.7800 [0.7439, 0.8172] |

## Paired differences in mIoU

| contrast | delta | 95% CI | p |
|---|---|---|---|
| MedKLIP: +DHS vs BEP | +0.0112 | [+0.0073, +0.0150] | 0.001 |
| GLoRIA: +RQP vs BEP | -0.0009 | [-0.0044, +0.0025] | 0.656 |
| GLoRIA: +DHS vs BEP | +0.0364 | [+0.0274, +0.0453] | 0.001 |
| GLoRIA: +RQP+DHS vs BEP | +0.0354 | [+0.0264, +0.0442] | 0.001 |
| GLoRIA: +RQP+DHS vs +RQP | +0.0363 | [+0.0276, +0.0451] | 0.001 |
| AFLoc: +RQP vs BEP | +0.0084 | [+0.0050, +0.0119] | 0.001 |
| AFLoc: +DHS vs BEP | +0.0050 | [-0.0024, +0.0121] | 0.190 |
| AFLoc: +RQP+DHS vs BEP | +0.0181 | [+0.0105, +0.0256] | 0.001 |
| AFLoc: +RQP+DHS vs +RQP | +0.0097 | [+0.0023, +0.0167] | 0.004 |
| RAD-DINO control (no L_SG): +RQP vs BEP | +0.0026 | [-0.0000, +0.0051] | 0.052 |
| RAD-DINO control (no L_SG): +DHS vs BEP | +0.0092 | [+0.0009, +0.0175] | 0.030 |
| RAD-DINO control (no L_SG): +RQP+DHS vs BEP | +0.0138 | [+0.0051, +0.0221] | 0.004 |
| RAD-DINO control (no L_SG): +RQP+DHS vs +RQP | +0.0113 | [+0.0027, +0.0193] | 0.016 |
| KAF-Ground: +RQP vs BEP | +0.0014 | [-0.0009, +0.0038] | 0.262 |
| KAF-Ground: +DHS vs BEP | +0.0140 | [+0.0054, +0.0227] | 0.002 |
| KAF-Ground: +RQP+DHS vs BEP | +0.0188 | [+0.0095, +0.0280] | 0.001 |
| KAF-Ground: +RQP+DHS vs +RQP | +0.0174 | [+0.0079, +0.0265] | 0.001 |
| KAF-Ground vs MedKLIP @ BEP | +0.1047 | [+0.0910, +0.1185] | 0.001 |
| KAF-Ground vs MedKLIP @ +DHS | +0.1075 | [+0.0947, +0.1203] | 0.001 |
| KAF-Ground vs MedKLIP @ best vs best | +0.1123 | [+0.0997, +0.1250] | 0.001 |
| KAF-Ground vs GLoRIA @ BEP | +0.0814 | [+0.0683, +0.0940] | 0.001 |
| KAF-Ground vs GLoRIA @ +RQP | +0.0837 | [+0.0707, +0.0963] | 0.001 |
| KAF-Ground vs GLoRIA @ +DHS | +0.0590 | [+0.0437, +0.0728] | 0.001 |
| KAF-Ground vs GLoRIA @ +RQP+DHS | +0.0648 | [+0.0491, +0.0785] | 0.001 |
| KAF-Ground vs GLoRIA @ best vs best | +0.0648 | [+0.0491, +0.0785] | 0.001 |
| KAF-Ground vs AFLoc @ BEP | +0.0155 | [+0.0049, +0.0273] | 0.008 |
| KAF-Ground vs AFLoc @ +RQP | +0.0086 | [-0.0020, +0.0207] | 0.110 |
| KAF-Ground vs AFLoc @ +DHS | +0.0245 | [+0.0147, +0.0351] | 0.001 |
| KAF-Ground vs AFLoc @ +RQP+DHS | +0.0163 | [+0.0071, +0.0263] | 0.004 |
| KAF-Ground vs AFLoc @ best vs best | +0.0163 | [+0.0071, +0.0263] | 0.004 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ BEP | -0.0078 | [-0.0147, -0.0010] | 0.032 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +RQP | -0.0090 | [-0.0158, -0.0017] | 0.008 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +DHS | -0.0030 | [-0.0084, +0.0024] | 0.290 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ +RQP+DHS | -0.0028 | [-0.0081, +0.0022] | 0.268 |
| KAF-Ground vs RAD-DINO control (no L_SG) @ best vs best | -0.0028 | [-0.0081, +0.0022] | 0.268 |
| RAD-DINO control vs AFLoc @ BEP | +0.0233 | [+0.0121, +0.0359] | 0.001 |
| RAD-DINO control vs AFLoc @ +RQP | +0.0175 | [+0.0059, +0.0297] | 0.004 |
| RAD-DINO control vs AFLoc @ +DHS | +0.0275 | [+0.0175, +0.0375] | 0.001 |
| RAD-DINO control vs AFLoc @ +RQP+DHS | +0.0191 | [+0.0096, +0.0292] | 0.006 |
| KAF-Ground: RQP x DHS interaction | +0.0034 | [+0.0010, +0.0057] | 0.006 |

## IoU per finding type (BEP and full protocol)

**BEP**

| finding | n | MedKLIP | GLoRIA | AFLoc | RAD-DINO control (no L_SG) | KAF-Ground |
|---|---|---|---|---|---|---|
| alveolar pattern | 39 | 0.2226 | 0.2187 | 0.3425 | 0.3900 | 0.3686 |
| aortic elongation | 107 | 0.3297 | 0.3955 | 0.4685 | 0.4723 | 0.4302 |
| atelectasis | 48 | 0.1020 | 0.2214 | 0.2931 | 0.3036 | 0.3088 |
| cardiomegaly | 99 | 0.4487 | 0.2990 | 0.3019 | 0.2776 | 0.2712 |
| interstitial pattern | 33 | 0.2228 | 0.1886 | 0.3578 | 0.3377 | 0.3226 |
| nodule | 75 | 0.0496 | 0.0911 | 0.0749 | 0.1140 | 0.1178 |
| pleural effusion | 87 | 0.0742 | 0.1668 | 0.2916 | 0.2741 | 0.2768 |
| pleural thickening | 42 | 0.0282 | 0.0829 | 0.0604 | 0.2081 | 0.2189 |

**+RQP+DHS** (MedKLIP: +DHS)

| finding | n | MedKLIP | GLoRIA | AFLoc | RAD-DINO control (no L_SG) | KAF-Ground |
|---|---|---|---|---|---|---|
| alveolar pattern | 39 | 0.2365 | 0.2767 | 0.3455 | 0.3934 | 0.3905 |
| aortic elongation | 107 | 0.3328 | 0.4819 | 0.5000 | 0.5176 | 0.4811 |
| atelectasis | 48 | 0.1019 | 0.1750 | 0.2543 | 0.2392 | 0.2564 |
| cardiomegaly | 99 | 0.4902 | 0.4052 | 0.4292 | 0.3806 | 0.3838 |
| interstitial pattern | 33 | 0.2442 | 0.2465 | 0.3860 | 0.3911 | 0.3862 |
| nodule | 75 | 0.0513 | 0.0819 | 0.0679 | 0.0964 | 0.0933 |
| pleural effusion | 87 | 0.0784 | 0.1834 | 0.2852 | 0.2757 | 0.2750 |
| pleural thickening | 42 | 0.0323 | 0.0967 | 0.0672 | 0.1942 | 0.1991 |

## Complete failures (IoU = 0) under BEP

AFLoc misses 30 pairs, KAF-Ground 5; 28 recovered, 3 lost (exact McNemar p = 4.6e-06).

| finding | n | AFLoc | KAF-Ground | recovered | lost |
|---|---|---|---|---|---|
| alveolar pattern | 39 | 0 | 0 | 0 | 0 |
| aortic elongation | 107 | 0 | 0 | 0 | 0 |
| atelectasis | 48 | 0 | 0 | 0 | 0 |
| cardiomegaly | 99 | 1 | 0 | 1 | 0 |
| interstitial pattern | 33 | 0 | 0 | 0 | 0 |
| nodule | 75 | 11 | 4 | 10 | 3 |
| pleural effusion | 87 | 1 | 1 | 0 | 0 |
| pleural thickening | 42 | 17 | 0 | 17 | 0 |
