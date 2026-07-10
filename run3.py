import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config3 import (seed_everything, SEED, DEVICE, OUTPUT_DIR, BATCH_SIZE,
                     HISTORY_SAVE_PATH, MODEL_SAVE_PATH)
from dataset3 import get_dataloaders, val_transform, VirusDataset
from model3 import build_model
from trainex3 import full_evaluation, benchmark


def main():
    seed_everything(SEED)
    if not MODEL_SAVE_PATH.exists():
        print("  [ERROR] report3.pth not found. Run trainex3.py first."); sys.exit(1)

    train_df = pd.read_csv(OUTPUT_DIR / "train_split.csv")
    val_df   = pd.read_csv(OUTPUT_DIR / "val_split.csv")
    _, val_loader = get_dataloaders(train_df, val_df, BATCH_SIZE)
    train_eval_loader = DataLoader(VirusDataset(train_df, transform=val_transform),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(use_se=ckpt.get("use_se", True), use_aux=ckpt.get("use_aux", True)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"  Loaded report3.pth | SE={ckpt.get('use_se')} | saved Val acc={ckpt.get('val_acc'):.2f}% "
          f"@ epoch {ckpt.get('epoch')}")

    hist = pd.read_csv(HISTORY_SAVE_PATH)
    full_evaluation(model, train_eval_loader, val_loader, hist, tag="reload")
    print("\n  Stored benchmark:", ckpt.get("benchmark"))
    print("  Live benchmark   :", benchmark(model))


if __name__ == "__main__":
    main()