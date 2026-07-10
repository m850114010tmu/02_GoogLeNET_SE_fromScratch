import os
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from PIL import Image

from config3 import seed_everything, SEED, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE
from dataset3 import val_transform
from model3 import build_model


# ─── Paths (relative to this script) ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "report3.pth"                   # root-level copy
if not MODEL_PATH.exists():
    MODEL_PATH = SCRIPT_DIR / "outputs" / "report3.pth"    # fallback to outputs/
DATA_DIR   = SCRIPT_DIR / "vir_data_exam"
TEST_DIR   = DATA_DIR / "test1_vir"
OUTPUT_CSV = SCRIPT_DIR / "clsn3_ans.csv"


def main():
    seed_everything(SEED)

    # ── 1. Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── 2. Load model from checkpoint ────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"  [ERROR] report3.pth not found at {MODEL_PATH}")
        print(f"          Run trainex3.py first, or copy report3.pth here.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    use_se  = checkpoint.get("use_se", True)
    use_aux = checkpoint.get("use_aux", True)

    # Build model with aux=False for inference (aux heads not needed)
    model = build_model(use_se=use_se, use_aux=False, num_classes=NUM_CLASSES).to(device)

    # Filter out auxiliary classifier weights from checkpoint if present
    state = {k: v for k, v in checkpoint["model_state"].items()
             if not k.startswith("aux")}
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"  Model  : SamGoogLeNetEx3 (SE={use_se})")
    print(f"  Loaded : {MODEL_PATH}")
    print(f"  Epoch  : {checkpoint.get('epoch', '?')}")
    print(f"  Val acc: {checkpoint.get('val_acc', '?')}")

    # ── 3. Collect test images ───────────────────────────────────────────
    if not TEST_DIR.exists():
        print(f"  [ERROR] Test folder not found: {TEST_DIR}")
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"  Test images found : {len(image_files)}")

    # ── 4. Inference ─────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    with torch.no_grad():
        for fname in image_files:
            img_path = TEST_DIR / fname

            # RGBA → RGB, apply val transform (clean: resize + crop + norm)
            image = Image.open(img_path).convert("RGB")
            tensor = val_transform(image).unsqueeze(0).to(device)

            logits = model(tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = CLASS_NAMES[pred_idx]

            # File name without extension (e.g. "image_001")
            file_stem = Path(fname).stem

            results.append({
                "filename":   file_stem,
                "prediction": pred_class,
            })

    elapsed = time.time() - t_start
    fps = len(image_files) / elapsed if elapsed > 0 else 0

    # ── 5. Save CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Predictions saved  : {OUTPUT_CSV}")
    print(f"  Total predictions  : {len(df)}")
    print(f"  Inference time     : {elapsed:.2f}s ({fps:.1f} images/sec)")

    # Distribution check
    print("\n  Prediction distribution:")
    for cls_name in CLASS_NAMES:
        count = len(df[df["prediction"] == cls_name])
        pct = count / len(df) * 100
        print(f"    {cls_name} : {count:>3d}  ({pct:.1f}%)")

    print(f"\n  Done\n")


if __name__ == "__main__":
    main()