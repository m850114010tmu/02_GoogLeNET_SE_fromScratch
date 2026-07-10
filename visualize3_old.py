import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config3 import (seed_everything, SEED, DEVICE, OUTPUT_DIR, FIGURES_DIR,
                     CLASS_NAMES, IMAGE_SIZE, MODEL_SAVE_PATH)
from dataset3 import val_transform, MEAN, STD
from model3 import build_model


def denorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def load_model():
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(use_se=ckpt.get("use_se", True), use_aux=False).to(DEVICE)
    # filter aux weights (aux disabled here) then load the rest
    state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("aux")}
    model.load_state_dict(state, strict=False); model.eval()
    return model


def one_image_per_class():
    df = pd.read_csv(OUTPUT_DIR / "train_split.csv")
    rows = [df[df.label == i].iloc[0] for i in range(len(CLASS_NAMES))]
    imgs = [val_transform(Image.open(r.filepath).convert("RGB")) for r in rows]
    return torch.stack(imgs).to(DEVICE), [r.class_name for r in rows]


# ─── (1) feature maps ─────────────────────────────────────────────────────────
def feature_maps(model, x, names):
    acts = {}
    layers = {"stem": model.stem, "inc3b": model.inc3b, "inc4e": model.inc4e, "inc5b": model.inc5b}
    hooks = [m.register_forward_hook(lambda _m, _i, o, n=n: acts.__setitem__(n, o.detach()))
             for n, m in layers.items()]
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    for bi, cname in enumerate(names):
        fig, axes = plt.subplots(1, len(layers) + 1, figsize=(4 * (len(layers) + 1), 4))
        axes[0].imshow(denorm(x[bi])); axes[0].set_title(f"input: {cname}"); axes[0].axis("off")
        for j, (lname, fm) in enumerate(acts.items()):
            m = fm[bi].mean(0).cpu().numpy()                 # avg across channels
            axes[j + 1].imshow(m, cmap="viridis"); axes[j + 1].set_title(lname); axes[j + 1].axis("off")
        plt.tight_layout(); p = FIGURES_DIR / f"featmaps_{cname.replace(' ', '')}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")


# ─── (2) Grad-CAM on the last inception inc5b ───────────────────────────────
def grad_cam(model, x, names):
    feats, grads = {}, {}
    target = model.inc5b
    fh = target.register_forward_hook(lambda _m, _i, o: feats.__setitem__("v", o))
    bh = target.register_full_backward_hook(lambda _m, gi, go: grads.__setitem__("v", go[0]))
    for bi, cname in enumerate(names):
        model.zero_grad()
        out = model(x[bi:bi + 1])
        cls = out.argmax(1)
        out[0, cls].backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True)        # GAP over gradients
        cam = F.relu((w * feats["v"]).sum(1, keepdim=True))  # weighted sum -> ReLU
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(denorm(x[bi])); ax[0].set_title(f"input: {cname}"); ax[0].axis("off")
        ax[1].imshow(denorm(x[bi])); ax[1].imshow(cam, cmap="jet", alpha=0.5)
        ax[1].set_title(f"Grad-CAM (pred {CLASS_NAMES[cls.item()]})"); ax[1].axis("off")
        plt.tight_layout(); p = FIGURES_DIR / f"gradcam_{cname.replace(' ', '')}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")
    fh.remove(); bh.remove()


# ─── (3) saliency map ─────────────────────────────────────────────────────────
def saliency(model, x, names):
    for bi, cname in enumerate(names):
        inp = x[bi:bi + 1].clone().requires_grad_(True)
        model.zero_grad()
        out = model(inp); cls = out.argmax(1); out[0, cls].backward()
        sal = inp.grad.abs().max(dim=1)[0][0].cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(denorm(x[bi])); ax[0].set_title(f"input: {cname}"); ax[0].axis("off")
        ax[1].imshow(sal, cmap="hot"); ax[1].set_title("Saliency |dScore/dInput|"); ax[1].axis("off")
        plt.tight_layout(); p = FIGURES_DIR / f"saliency_{cname.replace(' ', '')}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")


def main():
    seed_everything(SEED)
    model = load_model()
    x, names = one_image_per_class()
    print("  (1) feature maps...");  feature_maps(model, x, names)
    print("  (2) Grad-CAM...");      grad_cam(model, x, names)
    print("  (3) saliency...");      saliency(model, x, names)
    print(f"\n  All visualizations saved to {FIGURES_DIR}\n")


if __name__ == "__main__":
    main()