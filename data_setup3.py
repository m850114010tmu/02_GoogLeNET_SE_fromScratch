import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from config3 import (
    seed_everything, SEED, TRAIN_DIR, OUTPUT_DIR, STATS_PATH,
    CLASS_NAMES, CLASS_TO_IDX, VAL_SPLIT, IMAGE_SIZE,
)


def build_dataframe() -> pd.DataFrame:
    records = []
    for class_name in CLASS_NAMES:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"  [WARNING] Folder not found: {class_dir}")
            continue
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        for fname in files:
            records.append({
                "filepath":   str(class_dir / fname),
                "label":      CLASS_TO_IDX[class_name],
                "class_name": class_name,
                "filename":   fname,
            })
    return pd.DataFrame(records)


def compute_train_stats(train_df: pd.DataFrame) -> dict:
    """Mean/std per RGB channel over the TRAIN split only )."""
    print("\n  Computing channel mean/std on TRAIN split ...")
    psum    = np.zeros(3, dtype=np.float64)
    psum_sq = np.zeros(3, dtype=np.float64)
    n_pix   = 0
    for fp in train_df["filepath"]:
        img = Image.open(fp).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float64) / 255.0      # H,W,3 in [0,1]
        psum    += arr.sum(axis=(0, 1))
        psum_sq += (arr ** 2).sum(axis=(0, 1))
        n_pix   += arr.shape[0] * arr.shape[1]
    mean = psum / n_pix
    std  = np.sqrt(psum_sq / n_pix - mean ** 2)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    print(f"  mean = {np.round(mean, 4).tolist()}")
    print(f"  std  = {np.round(std, 4).tolist()}")
    return stats


def main():
    seed_everything(SEED)
    print("=" * 70)
    print("  DATA SETUP scan folders, stratified split")
    print("=" * 70)

    df = build_dataframe()
    print(f"\n  Total images : {len(df)}")
    print("  Class distribution:")
    for cls in CLASS_NAMES:
        c = int((df['class_name'] == cls).sum())
        print(f"    {cls} : {c:>4d}  ({c / len(df) * 100:.1f} %)")

    # Stratified split — keeps the 600/670/310 ratio inside both sets.
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    print(f"\n  Train : {len(train_df)}   Val : {len(val_df)}")
    print("  Stratification check (val):")
    for cls in CLASS_NAMES:
        tot = int((df['class_name'] == cls).sum())
        v   = int((val_df['class_name'] == cls).sum())
        print(f"    {cls} : {v}/{tot}  ({v / tot * 100:.1f} % in val)")

    stats = compute_train_stats(train_df)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved: train_split.csv, val_split.csv, train_stats.json -> {OUTPUT_DIR}")
    print("  Data setup complete.\n")


if __name__ == "__main__":
    main()