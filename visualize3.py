
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config3 import (seed_everything, SEED, DEVICE, OUTPUT_DIR, FIGURES_DIR,
                     CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, MODEL_SAVE_PATH)
from dataset3 import val_transform, MEAN, STD
from model3 import build_model


# ─── helpers ──────────────────────────────────────────────────────────────────

def denorm(t):
    """Tensor (C,H,W) -> numpy (H,W,3) in [0,1] for display."""
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def load_model():
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(use_se=ckpt.get("use_se", True), use_aux=False).to(DEVICE)
    state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("aux")}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def pick_samples(n_per_class=3):
    """Return n_per_class random images per class, reproducibly."""
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(OUTPUT_DIR / "train_split.csv")
    rows = []
    for cls_idx in range(NUM_CLASSES):
        subset = df[df.label == cls_idx]
        chosen = subset.sample(n=min(n_per_class, len(subset)), random_state=rng)
        rows.append(chosen)
    df_sel = pd.concat(rows, ignore_index=True)

    tensors, labels, names = [], [], []
    for _, r in df_sel.iterrows():
        img = Image.open(r.filepath).convert("RGB")
        tensors.append(val_transform(img))
        labels.append(int(r.label))
        names.append(r.class_name)
    return torch.stack(tensors).to(DEVICE), labels, names


# ─── feature map extraction ──────────────────────────────────────────────────

def extract_feature_maps(model, x):
    """Run forward once, capture activations at three depths via hooks.
    Returns dict[layer_name] = tensor (B, C, H, W)."""
    targets = {"stem": model.stem, "inc3b": model.inc3b, "inc5b": model.inc5b}
    acts = {}
    hooks = []
    for name, module in targets.items():
        hooks.append(
            module.register_forward_hook(
                lambda _m, _i, o, n=name: acts.__setitem__(n, o.detach())
            )
        )
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return acts


# ─── Grad-CAM (on inc5b — the deepest inception before GAP) ──────────────────

def compute_gradcam(model, x_single):
    """Grad-CAM for a single image tensor (1,3,H,W). Returns (H,W) heatmap [0,1]."""
    feats, grads = {}, {}
    target = model.inc5b
    fh = target.register_forward_hook(lambda _m, _i, o: feats.__setitem__("v", o))
    bh = target.register_full_backward_hook(lambda _m, gi, go: grads.__setitem__("v", go[0]))

    model.zero_grad()
    out = model(x_single)
    cls = out.argmax(1)
    out[0, cls].backward()

    w = grads["v"].mean(dim=(2, 3), keepdim=True)              # channel weights
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fh.remove(); bh.remove()
    return cam


# ─── saliency map ─────────────────────────────────────────────────────────────

def compute_saliency(model, x_single):
    """Vanilla gradient saliency for a single image. Returns (H,W) map [0,1]."""
    inp = x_single.clone().requires_grad_(True)
    model.zero_grad()
    out = model(inp)
    cls = out.argmax(1)
    out[0, cls].backward()
    sal = inp.grad.abs().max(dim=1)[0][0].cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal


# ─── THE BIG GRID (9 rows × 6 cols) ──────────────────────────────────────────

def make_comparison_grid(model, x, labels, names, n_per_class=3):
    """
    Grid layout:
        Col 0: Input (original RGB)
        Col 1: Stem feature map (channel-mean)
        Col 2: Inc3b feature map
        Col 3: Inc5b feature map
        Col 4: Grad-CAM overlay
        Col 5: Saliency map
    Row grouping: class 1 (3 rows) | class 2 (3 rows) | class 3 (3 rows)
    """
    col_titles = ["Input", "Stem", "Inc3b (mid)", "Inc5b (deep)", "Grad-CAM", "Saliency"]
    n_rows = NUM_CLASSES * n_per_class
    n_cols = len(col_titles)

    # pre-compute feature maps (batched, fast)
    acts = extract_feature_maps(model, x)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.8))

    for row in range(n_rows):
        cls_idx = labels[row]
        cname   = names[row]

        # col 0 — input
        axes[row, 0].imshow(denorm(x[row]))

        # col 1,2,3 — feature maps (channel-mean heatmap)
        for ci, layer_name in enumerate(["stem", "inc3b", "inc5b"]):
            fm = acts[layer_name][row].mean(0).cpu().numpy()
            axes[row, ci + 1].imshow(fm, cmap="viridis")

        # col 4 — Grad-CAM overlay
        cam = compute_gradcam(model, x[row:row + 1])
        axes[row, 4].imshow(denorm(x[row]))
        axes[row, 4].imshow(cam, cmap="jet", alpha=0.5)

        # col 5 — saliency
        sal = compute_saliency(model, x[row:row + 1])
        axes[row, 5].imshow(sal, cmap="hot")

        # row label on the left
        axes[row, 0].set_ylabel(cname, fontsize=11, fontweight="bold", rotation=0,
                                labelpad=60, va="center")

    # column headers
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=12, fontweight="bold")

    # class separators (horizontal lines between groups)
    for cls_i in range(1, NUM_CLASSES):
        row_idx = cls_i * n_per_class
        for ci in range(n_cols):
            for spine in axes[row_idx, ci].spines.values():
                spine.set_visible(True); spine.set_edgecolor("red"); spine.set_linewidth(2)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Class Comparison: Input → Feature Maps → Grad-CAM → Saliency",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = FIGURES_DIR / "class_comparison_grid.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ok {p.name}  ({n_rows} rows x {n_cols} cols)")


# ─── compact Grad-CAM panel (3 classes × 3 samples) ──────────────────────────

def make_gradcam_panel(model, x, labels, names, n_per_class=3):
    """3 rows (one per class), each row = 3 Grad-CAM overlays side by side."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class, figsize=(n_per_class * 4, NUM_CLASSES * 3.5))
    for row in range(NUM_CLASSES):
        for col in range(n_per_class):
            idx = row * n_per_class + col
            cam = compute_gradcam(model, x[idx:idx + 1])
            axes[row, col].imshow(denorm(x[idx]))
            axes[row, col].imshow(cam, cmap="jet", alpha=0.45)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(CLASS_NAMES[row], fontsize=12,
                                          fontweight="bold", rotation=0, labelpad=55, va="center")
    plt.suptitle("Grad-CAM: Where does the network focus for each class?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "gradcam_class_panel.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ok {p.name}")


# ─── compact Saliency panel ──────────────────────────────────────────────────

def make_saliency_panel(model, x, labels, names, n_per_class=3):
    """3 rows (one per class), each row = 3 saliency maps."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class * 2,
                             figsize=(n_per_class * 6, NUM_CLASSES * 3.2))
    for row in range(NUM_CLASSES):
        for col in range(n_per_class):
            idx = row * n_per_class + col
            sal = compute_saliency(model, x[idx:idx + 1])
            ci_img = col * 2       # input column
            ci_sal = col * 2 + 1   # saliency column
            axes[row, ci_img].imshow(denorm(x[idx]))
            axes[row, ci_img].set_xticks([]); axes[row, ci_img].set_yticks([])
            axes[row, ci_sal].imshow(sal, cmap="hot")
            axes[row, ci_sal].set_xticks([]); axes[row, ci_sal].set_yticks([])
            if col == 0:
                axes[row, ci_img].set_ylabel(CLASS_NAMES[row], fontsize=12,
                                             fontweight="bold", rotation=0, labelpad=55, va="center")
    plt.suptitle("Saliency: Which pixels matter most for each class prediction?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "saliency_class_panel.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ok {p.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    seed_everything(SEED)
    print("=" * 70)
    print("  VISUALIZATION — 3 classes × 3 samples comparison grid")
    print("=" * 70)

    model = load_model()
    N_PER_CLASS = 3
    x, labels, names = pick_samples(n_per_class=N_PER_CLASS)
    print(f"  Loaded {len(x)} images ({N_PER_CLASS} per class)")

    print("\n  [1/3] Building full comparison grid (9 rows × 6 cols)...")
    make_comparison_grid(model, x, labels, names, N_PER_CLASS)

    print("\n  [2/3] Grad-CAM class panel...")
    make_gradcam_panel(model, x, labels, names, N_PER_CLASS)

    print("\n  [3/3] Saliency class panel...")
    make_saliency_panel(model, x, labels, names, N_PER_CLASS)

    print(f"\n  All saved to {FIGURES_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()