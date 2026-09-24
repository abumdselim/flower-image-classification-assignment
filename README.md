# FlowerNet — Multi-Model Image Classification with Cross-Validation

Course project: **Computer Vision / Machine Learning** — custom-dataset image
classification comparing **5 models** under **5-fold stratified
cross-validation**, with **data augmentation**, **learning curves** and a full
**evaluation-metric suite** (accuracy, macro precision/recall/F1, ROC-AUC,
out-of-fold confusion matrices).

📄 **Final submission report:** [`report/FlowerNet_Report.pdf`](report/FlowerNet_Report.pdf) (55 pages)

## Dataset

| Property | Value |
|---|---|
| Classes | **3** — daisy, rose, sunflower |
| Images | **100 per class → 300 total** |
| Format | RGB JPEG, resized to 224 × 224 |
| Source | Web-collected via image search, then self-curated & cleaned (`scripts/collect_dataset.py`) |
| Split | No fixed split — evaluated with 5-fold stratified cross-validation over all 300 images |

## Project requirements coverage

| Requirement | Where |
|---|---|
| Custom dataset, ≥ 3 classes | `dataset/` — 3 flower classes × 100 images = 300 |
| Cross-validation | `train_cv.py` — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for every model |
| ≥ 3 models (we use 5) | SimpleCNN (from scratch), VGG16, ResNet50, MobileNetV2, EfficientNet-B0 |
| Data augmentation | `data.py::get_augment_transform` — random resized crop, horizontal flip, rotation, colour jitter (applied to **training folds only**) |
| Proper evaluation metrics | `train_cv.py::evaluate_metrics`, `evaluate.py` — accuracy, macro P/R/F1, ROC-AUC, OOF confusion matrices |
| Learning curves + overfitting check | epoch-wise train/val curves per model (fig 07-*) + accuracy-vs-training-size curve (fig 12) + train–val gap bar chart (fig 13) |

## Final results (mean ± std over 5 folds, seed = 42)

| Model | Accuracy | Macro F1 | ROC-AUC | Params (total / trainable) |
|---|---|---|---|---|
| SimpleCNN (scratch) | 94.7 ± 3.9 % | 0.9463 | 0.9779 | 110,595 / 110,595 |
| VGG16 (frozen) | 97.3 ± 1.3 % | 0.9733 | 0.9965 | 14,780,739 / 66,051 |
| ResNet50 (frozen) | **99.0 ± 0.8 %** | 0.9900 | 0.9974 | 23,770,691 / 262,659 |
| MobileNetV2 (frozen) | **99.0 ± 0.8 %** | 0.9900 | **0.9995** | 2,388,227 / 164,355 |
| EfficientNet-B0 (frozen) | **99.0 ± 1.3 %** | **0.9900** | 0.9991 | 4,171,903 / 164,355 |

- **Best accuracy / F1:** ResNet50, MobileNetV2, EfficientNet-B0 (99.0 %)
- **Best ROC-AUC:** MobileNetV2 (0.9995)
- **Overfitting analysis:** with augmentation the training curves sit *below*
  the validation curves (negative train–val gap) → **no overfitting detected
  for any model** (see `figures/13_overfit_gap.png`)
- **Highest variance:** SimpleCNN (fold std ±3.9 %) — expected for a small
  from-scratch CNN on 300 images

## Project structure

```
cv_assignment/
├── config.py            # all hyper-parameters and paths
├── data.py              # dataset loading, eval/augment transforms
├── models.py            # SimpleCNN + 4 frozen ImageNet backbones + MLP heads
├── prepare_dataset.py   # raw images -> uniform 224x224 dataset
├── extract_features.py  # frozen backbones -> cached features (clean + augmented)
├── train_cv.py          # 5-fold stratified CV, metrics, learning curves
├── evaluate.py          # all figures + comparison CSV + summary table
├── main.py              # one-command driver
├── scripts/
│   ├── collect_dataset.py   # dataset collection helper (image search + cleaning)
│   ├── make_code_shots.py   # code screenshots for the report
│   ├── make_report_pdf.py   # builds report/FlowerNet_Report.pdf
│   └── make_summary.py      # results_summary.json for the web showcase
├── dataset/             # FINAL dataset: dataset/<class>/<class>_XXXX.jpg (300 imgs)
├── results/             # metrics.json, comparison_table.csv, per-run fragments
├── figures/             # all 20 evaluation figures (PNG)
├── report/              # FlowerNet_Report.pdf + code screenshots
└── logs/                # run logs (regenerated; not tracked)
```

> `dataset_raw/` (original downloads before resizing, ~192 MB) and
> `results/features/` (cached `.npy` backbone features) are local working
> directories and are **not tracked** in the repository. Both are reproducible:
> `python scripts/collect_dataset.py` → `python prepare_dataset.py` →
> `python extract_features.py`.

## How to run

```bash
pip install -r requirements.txt        # torch CPU wheels are fine
python main.py                          # full pipeline (extraction is cached)
```

Individual stages:

```bash
python prepare_dataset.py               # rebuild dataset/ from dataset_raw/
python extract_features.py              # (re)extract frozen backbone features
python train_cv.py                      # cross-validation training only
python evaluate.py                      # figures + tables only
```

## Models

| Model | Type | Key idea | Trainable part |
|---|---|---|---|
| SimpleCNN | from scratch | 3×(Conv-BN-ReLU-MaxPool) baseline | whole network |
| VGG16 | transfer (frozen) | uniform 3×3 conv stacks | MLP head on GAP features |
| ResNet50 | transfer (frozen) | residual/skip connections | MLP head on GAP features |
| MobileNetV2 | transfer (frozen) | depthwise-separable inverted residuals | MLP head on GAP features |
| EfficientNet-B0 | transfer (frozen) | compound scaling | MLP head on GAP features |

**Why these models?** They span the classic design spectrum: a small
hand-designed CNN baseline, a depth-stacked network (VGG16), a
residual network (ResNet50), a lightweight mobile architecture
(MobileNetV2) and a compound-scaled architecture (EfficientNet-B0) —
so the comparison covers capacity vs. efficiency trade-offs under
identical data, folds and augmentation.

All experiments use a fixed seed (42) and are fully reproducible on CPU.
