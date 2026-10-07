# Results

```
results/
  ms-cxr/          1,162 MS-CXR pairs
  padchest-gr/     530 PadChest-GR test-split pairs (zero-shot transfer)
    persample/     <model>_rqp-<off|on>.csv, one row per pair and DHS setting
    tables/        overall.csv, contrasts.csv, per_category.csv, complete_misses.csv
    summary.md     the main tables in readable form
  ablations/       training- and inference-side ablations (see its README)
```

Models: `kaf-ground`, `afloc`, `gloria`, `medklip`, and `raddino-control`
(the KAF-Ground recipe without the graph loss).

Per-sample columns: `idx` (position in the dataset loader), `rank` (position
in sorted image-path + phrase order, used by the bootstrap), `pair_id` (hash of
image file name and phrase), `category`, `dhs` (off/on), `q` (fraction of
pixels DHS treats as foreground), `point` (P@1), `iou` and `dice` (averaged
over thresholds 0.1 to 0.5) and `cnr`.

Everything in `tables/` and `summary.md` is regenerated with

```bash
python scripts/summarize_results.py
python scripts/check_paper_numbers.py    # 150 numbers stated in the paper
python scripts/make_figures.py           # figures/regenerated/
```
