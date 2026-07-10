# GoogLeNet-22 + Squeeze-and-Excitation — 3-Class Image Classifier (from scratch)

A custom **GoogLeNet-22 (BN-Inception)** convolutional network with **Squeeze-and-Excitation (SE) attention**, trained **entirely from scratch** (no ImageNet pretrained weights) to classify texture-based images into **3 classes**. The pipeline is deliberately built to survive **colour-shifted / noisy test images** through an aggressive colour + noise augmentation strategy.

> Author: Minh-Phuong Luong
> Task: design → train → evaluate → predict a 3-class classifier on the `vir_data_exam` dataset (1,580 labelled train images + 300 unlabelled test images).

---

## Highlights

| Metric | GoogLeNet-22 + SE | GoogLeNet-22 (no SE) |
|---|---|---|
| Validation accuracy | **96.84 %** | 95.57 % |
| Macro F1 | **0.9600** | 0.9524 |
| Best epoch | 49 | 46 |
| Parameters | 10,794,313 | 10,320,041 |
| Model size | 43.18 MB | 41.28 MB |
| Inference latency | 28.85 ms (~34.7 FPS) | ~20.7 ms |
| Train/Val accuracy gap | 0.95 pp (low overfitting) | −0.24 pp |

SE attention adds ~474 K parameters (~4.6 %) and lifts validation accuracy by **+1.27 pp** while keeping inference cost effectively unchanged.

---

## Project structure

```
02_GoogLeNET_SE_fromScratch/
│
├── config3.py            # Central config: seeds, paths, hyperparameters, feature flags
├── data_setup3.py        # (1) Scan folders → stratified 80/20 split → train mean/std stats
├── dataset3.py           # Dataset class + augmentation pipeline + WeightedRandomSampler
├── model3.py             # SE block, Inception module, GoogLeNet-22 backbone, aux heads
├── trainex3.py           # (2) Training loop, evaluation, SE-vs-noSE comparison, benchmark
├── visualize3.py         # (3) Confusion matrices, ROC, per-class, Grad-CAM, saliency, feat maps
├── visualize3_old.py     # Earlier version of the visualisation script (kept for reference)
├── run3.py               # Reload a saved checkpoint and re-run full evaluation
├── testrun3.py           # (4) Inference on the unlabelled test set → clsn3_ans.csv
├── pipeline.md           # Short run-order note
├── util3.txt             # Frozen environment (pip freeze output)
│
├── outputs/              # ← Generated artifacts (splits, stats, history, checkpoint)
│   ├── train_split.csv       # 1,264 training rows (filepath, label, class_name, filename)
│   ├── val_split.csv         # 316 validation rows
│   ├── train_stats.json      # Per-channel mean/std computed on the TRAIN split only
│   ├── training_history.csv  # 60 epochs: loss/acc/lr/time per epoch
│   ├── se_comparison.json    # SE vs noSE metrics side by side
│   ├── report3.pth           # Trained checkpoint (NOT committed — see .gitignore)
│   └── figures/              # All PNG plots
│       ├── cm_val_SE.png / cmnorm_val_SE.png ...      # Confusion matrices (raw + normalised)
│       ├── roc_SE.png / roc_noSE.png / roc_reload.png # ROC curves
│       ├── per_class_*.png                            # Per-class precision/recall/F1
│       ├── learning_curves_*.png                      # Loss/accuracy curves
│       ├── featmaps_class*.png                        # Feature-map visualisations
│       ├── gradcam_class*.png / gradcam_class_panel.png
│       ├── saliency_class*.png / saliency_class_panel.png
│       └── class_comparison_grid.png
│
├── clsn3_ans.csv         # Final predictions on the 300 test images (filename, prediction)
└── 02_GoogLeNET_SE_fromScratch.pdf   # Full written report
```

> **Note on paths:** the exact filenames above (e.g. `cm_val_SE.png`) currently sit at the top level of this folder. When you commit, you can either keep them flat or move the CSV/JSON/PNG artifacts into an `outputs/` (and `outputs/figures/`) directory to match `config3.py`. The README describes the intended layout.

---

## Dataset

The dataset is **not photographic** — the images are structured, texture-like patterns.

| Class | Images | Proportion | Role |
|---|---|---|---|
| class 1 | 600 | 38.0 % | Second largest |
| class 2 | 670 | 42.4 % | Majority class |
| class 3 | 310 | 19.6 % | Minority class |
| **Total** | **1,580** | 100 % | — |

- All images are **432 × 288 px, RGBA** (alpha discarded → RGB).
- Imbalance ratio ≈ **2.16 : 1** (class 2 vs class 3), handled with a `WeightedRandomSampler`.
- Train mean/std (computed on the train split only, leak-safe):
  `μ ≈ [0.798, 0.798, 0.798]`, `σ ≈ [0.258, 0.258, 0.258]` — near-identical across channels (effectively greyscale content).

The raw `vir_data_exam/` folder (train images + `test1_vir/` test images) is **not included** in this repo. Place it beside the scripts before running.

---

## Method summary

**Backbone** — GoogLeNet-22 (BN-Inception): stem → 9 inception modules in 3 depth stages → global average pooling → dropout(0.4) → 3-class linear head. Batch norm after every conv; Kaiming/He init for convs.

**SE attention** — one SE block after every inception concatenation: squeeze (global avg pool → C-vector) → excitation (FC → ReLU → FC → sigmoid, reduction ratio 16) → per-channel rescaling. A single `use_se` flag toggles the SE / no-SE variants for the ablation.

**Auxiliary classifiers** — two aux heads (from Inception 4a and 4d) during training, each weighted 0.3, for deep-network gradient signal.

**Augmentation (train only)** — RGBA→RGB, RandomResizedCrop(224, scale 0.7–1.0), horizontal flip, ±15° rotation, ColorJitter(0.3, 0.3, 0.3, 0.10), **RandomChannelScale(±0.15)** (per-channel RGB gain — the key colour-robustness transform), Normalize(train μ/σ), Gaussian noise (σ=0.04), RandomErasing(p=0.25). Val/test transforms are clean (resize + centre crop + normalise only).

**Training** — AdamW, lr 3e-4, weight decay 1e-4, batch size 32, 60 epochs (max), 5-epoch linear warmup + cosine annealing, label smoothing 0.1, early stopping patience 12, mixed precision (AMP), seed 42.

Full details and ablations are in `02_GoogLeNET_SE_fromScratch.pdf`.

---

## Setup

Requires **Python 3.11+** and a CUDA-capable GPU (CPU works but is slow). The frozen environment (`util3.txt`) uses PyTorch 2.12 + CUDA 12.6.

```bash
# clone your repo first, then:
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas pillow scikit-learn matplotlib seaborn
```

> `util3.txt` is a `pip freeze` with conda `@ file://` paths — it documents the exact versions but is not directly installable. Use the command above (or pin versions from `util3.txt` if you want an exact match).

---

## How to run

Order comes from `pipeline.md`:

```bash
python data_setup3.py    # (1) build stratified split + compute train stats
python trainex3.py       # (2) train SE + noSE, save report3.pth, history, comparison
python visualize3.py     # (3) generate all figures (confusion, ROC, Grad-CAM, ...)
python testrun3.py       # (4) predict the 300 test images -> clsn3_ans.csv
```

- `run3.py` reloads `report3.pth` and re-runs the full evaluation on the val/train splits (handy for regenerating metrics without retraining).
- `testrun3.py` looks for `report3.pth` at the project root first, then falls back to `outputs/report3.pth`.

---

## Outputs you'll get

- **`outputs/train_split.csv` / `val_split.csv`** — reproducible 1,264 / 316 stratified split (seed 42).
- **`outputs/train_stats.json`** — leak-safe normalisation stats.
- **`outputs/training_history.csv`** — 60 epochs of loss/accuracy/lr/time.
- **`outputs/se_comparison.json`** — SE vs no-SE metrics.
- **`clsn3_ans.csv`** — final predictions: columns `filename, prediction` (e.g. `image_001, class 1`).
- **`outputs/figures/*.png`** — confusion matrices, ROC curves, per-class metrics, learning curves, Grad-CAM, saliency maps, feature maps, and a class comparison grid.

---

## Reproducibility

`seed_everything(42)` locks Python / NumPy / PyTorch RNGs and sets cuDNN deterministic mode. The train/val split is persisted as CSV so any re-run produces identical partitions.

---

## License

Add a license of your choice (e.g. MIT) as a `LICENSE` file if you intend to share this publicly.
