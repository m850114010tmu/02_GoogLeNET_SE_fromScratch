
import sys, time, json, subprocess, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, precision_recall_fscore_support, f1_score)
from sklearn.preprocessing import label_binarize

from config3 import (
    seed_everything, SEED, DEVICE, OUTPUT_DIR, FIGURES_DIR,
    NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, PATIENCE, LABEL_SMOOTHING, GRAD_ACCUM_STEPS, IMAGE_SIZE,
    USE_AUX, AUX_WEIGHT, USE_MIXUP, MIXUP_ALPHA, COMPARE_SE,
    MODEL_SAVE_PATH, MODEL_SAVE_PATH_ROOT, HISTORY_SAVE_PATH, UTIL_SAVE_PATH,
)
from dataset3 import get_dataloaders, val_transform, VirusDataset
from model3 import build_model
from torch.utils.data import DataLoader


def fmt_time(s): m, s = divmod(int(s), 60); return f"{m}m {s:02d}s" if m else f"{s}s"


# ─── mixup helper ─────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─── one training epoch (handles AMP, aux heads, mixup, grad-accum) ───────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if USE_MIXUP:
            images, ya, yb, lam = mixup_batch(images, labels, MIXUP_ALPHA)

        with torch.amp.autocast(DEVICE.type, enabled=use_amp):
            out = model(images)
            heads = out if isinstance(out, tuple) else (out,)
            weights = [1.0] + [AUX_WEIGHT] * (len(heads) - 1)
            if USE_MIXUP:
                loss = sum(w * (lam * criterion(h, ya) + (1 - lam) * criterion(h, yb))
                           for w, h in zip(weights, heads))
            else:
                loss = sum(w * criterion(h, labels) for w, h in zip(weights, heads))
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        main = heads[0]
        run_loss += loss.item() * images.size(0) * GRAD_ACCUM_STEPS
        pred = main.argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()      # approx acc when mixup is on
    return run_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, use_amp):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    L, P, PR = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(DEVICE.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), 1)
        pred  = logits.argmax(1)
        run_loss += loss.item() * images.size(0)
        total += labels.size(0); correct += pred.eq(labels).sum().item()
        L.append(labels.cpu().numpy()); P.append(pred.cpu().numpy()); PR.append(probs.cpu().numpy())
    return (run_loss / total, 100.0 * correct / total,
            np.concatenate(L), np.concatenate(P), np.vstack(PR))


# ─── training loop for ONE variant use_se True/False ────────────────────────
def train_variant(use_se, train_loader, val_loader, train_df, val_df, tag):
    seed_everything(SEED)
    model = build_model(use_se=use_se, use_aux=USE_AUX).to(DEVICE)
    total_p, train_p = model.count_parameters()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                               total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    use_amp = (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)

    gpu = torch.cuda.get_device_name(0) if use_amp else "CPU"
    print("\n" + "=" * 92)
    print(f"  TRAINING [{tag}] | SE={use_se} | {len(train_df)} train / {len(val_df)} val")
    print("=" * 92)
    print(f"  Device {gpu} | Params {total_p:,} | LR {LEARNING_RATE} | Batch {BATCH_SIZE}"
          f" | Epochs {EPOCHS} | Mixup {USE_MIXUP}")
    print("=" * 92)

    best_acc, patience_ctr, history = 0.0, 0, []
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        te = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        va_loss, va_acc, *_ = validate(model, val_loader, criterion, use_amp)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]; et = time.time() - te
        marker = ""
        if va_acc > best_acc:
            best_acc, patience_ctr = va_acc, 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep; marker = " * Best"
        else:
            patience_ctr += 1
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr, "time_s": et})
        print(f"  Ep {ep:>3}/{EPOCHS} | Train L {tr_loss:.4f} A {tr_acc:6.2f}% | "
              f"Val L {va_loss:.4f} A {va_acc:6.2f}% | LR {lr:.1e} | "
              f"{fmt_time(et)} ETA {fmt_time(et*(EPOCHS-ep))}{marker}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {ep} (no val gain for {PATIENCE}).")
            break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    print(f"  [{tag}] done in {fmt_time(train_time)} | best Val {best_acc:.2f}% @ ep {best_epoch}")
    return model, pd.DataFrame(history), {
        "tag": tag, "use_se": use_se, "best_val_acc": best_acc,
        "params": total_p, "best_epoch": best_epoch, "train_time_s": train_time,
    }


# ─── speed + memory benchmark ─────────────────────────────────────────────────
@torch.no_grad()
def benchmark(model):
    model.eval()
    x1 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for _ in range(10): _ = model(x1)                     # warmup
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(50): _ = model(x1)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    lat_ms = (time.time() - t) / 50 * 1000
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else 0.0
    total_p, _ = model.count_parameters()
    size_mb = total_p * 4 / 1e6
    return {"latency_ms": lat_ms, "fps": 1000.0 / lat_ms,
            "peak_vram_mb": peak_mb, "model_size_mb": size_mb, "params": total_p}


# ─── plots (learning curve / confusion / ROC / per-class) ─────────────────────
def plot_learning_curves(hist, tag):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    a.plot(hist.epoch, hist.train_loss, "o-", label="train"); a.plot(hist.epoch, hist.val_loss, "o-", label="val")
    a.set_title(f"Loss [{tag}]"); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    b.plot(hist.epoch, hist.train_acc, "o-", label="train"); b.plot(hist.epoch, hist.val_acc, "o-", label="val", color="g")
    b.set_title(f"Accuracy [{tag}]"); b.set_xlabel("epoch"); b.legend(); b.grid(alpha=.3)
    plt.tight_layout(); p = FIGURES_DIR / f"learning_curves_{tag}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")

def plot_confusion(y, yp, tag, split):
    cm = confusion_matrix(y, yp)
    for norm, cmap, fmt, name in [(False, "Blues", "d", "cm"), (True, "Oranges", ".1f", "cmnorm")]:
        M = cm.astype(float) / cm.sum(1, keepdims=True) * 100 if norm else cm
        plt.figure(figsize=(7, 6))
        sns.heatmap(M, annot=True, fmt=fmt, cmap=cmap, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("Predicted"); plt.ylabel("Truth"); plt.title(f"{name} {split} [{tag}]")
        plt.tight_layout(); p = FIGURES_DIR / f"{name}_{split}_{tag}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")

def plot_roc(y, prob, tag):
    yb = label_binarize(y, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(7, 6))
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(yb[:, i], prob[:, i]); plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC {auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.5); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC OvR [{tag}]"); plt.legend(loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); p = FIGURES_DIR / f"roc_{tag}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")

def plot_per_class(y, yp, tag):
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, labels=list(range(NUM_CLASSES)), zero_division=0)
    x = np.arange(NUM_CLASSES); w = .25
    plt.figure(figsize=(10, 5))
    plt.bar(x - w, pr, w, label="Precision"); plt.bar(x, rc, w, label="Recall"); plt.bar(x + w, f1, w, label="F1")
    plt.xticks(x, CLASS_NAMES); plt.ylim(0, 1.15); plt.title(f"Per-class [{tag}]"); plt.legend(); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); p = FIGURES_DIR / f"per_class_{tag}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"  ok {p.name}")


def full_evaluation(model, train_eval_loader, val_loader, hist, tag):
    use_amp = (DEVICE.type == "cuda"); crit = nn.CrossEntropyLoss()
    print("\n" + "=" * 92); print(f"  EVALUATION [{tag}]"); print("=" * 92)
    plot_learning_curves(hist, tag)
    vl, va, yl, yp, pr = validate(model, val_loader, crit, use_amp)
    f1m = f1_score(yl, yp, average="macro", zero_division=0)
    print(f"  Val acc {va:.2f}% | Val F1(macro) {f1m:.4f}")
    print("\n" + classification_report(yl, yp, target_names=CLASS_NAMES, zero_division=0))
    plot_confusion(yl, yp, tag, "val"); plot_roc(yl, pr, tag); plot_per_class(yl, yp, tag)
    tl, ta, *_ = validate(model, train_eval_loader, crit, use_amp)
    gap = ta - va
    verdict = "OK (low overfit)" if gap < 5 else ("watch (moderate)" if gap < 10 else "OVERFIT")
    print(f"  Overfit check -> Train {ta:.2f}% / Val {va:.2f}% / Gap {gap:.2f}% -> {verdict}")
    return va, f1m


def export_util():
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        UTIL_SAVE_PATH.write_text(r.stdout, encoding="utf-8"); print(f"  ok util3.txt -> {UTIL_SAVE_PATH}")
    except Exception as e:
        print(f"  [warn] util3.txt failed: {e}  (run: pip freeze > util3.txt)")


def save_checkpoint(model, info, bench):
    ckpt = {"model_state": model.state_dict(), "class_names": CLASS_NAMES,
            "use_se": info["use_se"], "use_aux": USE_AUX, "seed": SEED,
            "val_acc": info["best_val_acc"], "epoch": info["best_epoch"], "benchmark": bench}
    torch.save(ckpt, MODEL_SAVE_PATH)
    torch.save(ckpt, MODEL_SAVE_PATH_ROOT)
    print(f"  ok report3.pth -> {MODEL_SAVE_PATH}")
    print(f"  ok report3.pth -> {MODEL_SAVE_PATH_ROOT}")


def main():
    seed_everything(SEED)
    tr_csv, va_csv = OUTPUT_DIR / "train_split.csv", OUTPUT_DIR / "val_split.csv"
    if not tr_csv.exists():
        print("  [ERROR] run  python data_setup3.py  first."); sys.exit(1)
    train_df, val_df = pd.read_csv(tr_csv), pd.read_csv(va_csv)
    train_loader, val_loader = get_dataloaders(train_df, val_df, BATCH_SIZE)
    # clean (no-aug) train loader for an honest overfitting check
    train_eval_loader = DataLoader(VirusDataset(train_df, transform=val_transform),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    variants = [True, False] if COMPARE_SE else [True]
    results, se_model, se_hist, se_info = [], None, None, None
    for use_se in variants:
        tag = "SE" if use_se else "noSE"
        model, hist, info = train_variant(use_se, train_loader, val_loader, train_df, val_df, tag)
        va, f1m = full_evaluation(model, train_eval_loader, val_loader, hist, tag)
        bench = benchmark(model)
        info.update({"val_acc": va, "f1_macro": f1m, **bench})
        results.append(info)
        if use_se:
            se_model, se_hist, se_info, se_bench = model, hist, info, bench

    # save the SE model as the deliverable report3.pth + history
    se_hist.to_csv(HISTORY_SAVE_PATH, index=False)
    save_checkpoint(se_model, se_info, se_bench)
    export_util()

    # comparison table (the exam requirement: with vs without attention)
    print("\n" + "=" * 92); print("  COMPARISON — WITH vs WITHOUT SE attention"); print("=" * 92)
    print(f"  {'variant':<8}{'val_acc%':>10}{'F1_macro':>10}{'params':>12}"
          f"{'lat_ms':>10}{'FPS':>9}{'VRAM_MB':>10}{'size_MB':>9}")
    for r in results:
        print(f"  {r['tag']:<8}{r['val_acc']:>10.2f}{r['f1_macro']:>10.4f}{r['params']:>12,}"
              f"{r['latency_ms']:>10.2f}{r['fps']:>9.1f}{r['peak_vram_mb']:>10.1f}{r['model_size_mb']:>9.1f}")
    with open(OUTPUT_DIR / "se_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved se_comparison.json. ALL DONE.\n")


if __name__ == "__main__":
    main()