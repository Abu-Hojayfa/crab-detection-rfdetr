import os, sys, glob, shutil, subprocess, threading, time
import torch

# =========================================================
# CONFIG — edit before running
# =========================================================
DATASET_DIR = None   

# =========================================================
# Step 1: Install
# =========================================================
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rfdetr[train]"])

# =========================================================
# Step 2: GPU check
# =========================================================
assert torch.cuda.is_available(), "Enable GPU in notebook settings."
print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================================================
# Step 3: Locate Roboflow COCO dataset
# =========================================================
def find_dataset_dir(root="/kaggle/input"):
    for dirpath, _, filenames in os.walk(root):
        if "_annotations.coco.json" in filenames:
            return os.path.dirname(dirpath)
    return None

if DATASET_DIR:
    dataset_dir = DATASET_DIR
    assert os.path.isdir(dataset_dir), f"DATASET_DIR does not exist: {dataset_dir}"
    assert glob.glob(os.path.join(dataset_dir, "**", "_annotations.coco.json"), recursive=True), \
        f"No _annotations.coco.json found under {dataset_dir}"
else:
    dataset_dir = find_dataset_dir()
    assert dataset_dir, "Could not find _annotations.coco.json under /kaggle/input"
print(f"Dataset dir: {dataset_dir}")

# =========================================================
# Step 4: Output dir — always wiped clean, no restore/resume.
# RF-DETR writes checkpoints + metrics.csv here as it trains, and Kaggle
# persists /kaggle/working on its own, so no separate backup step is needed.
# =========================================================
output_dir = "/kaggle/working/output"

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"Cleared {output_dir} — starting from a clean output directory")

# =========================================================
# Step 5: Per-epoch metrics printer
# RF-DETR always writes {output_dir}/metrics.csv via its built-in CSV logger.
# This thread tails that file and prints each new epoch's validation results
# plus the best mAP seen so far, without needing to hook into RF-DETR internals.
# =========================================================
def metrics_watcher():
    import pandas as pd
    metrics_path = os.path.join(output_dir, "metrics.csv")
    seen_rows = 0
    best_map = -1.0
    best_epoch = None
    print("[metrics] watcher started, waiting for metrics.csv ...")
    while True:
        time.sleep(15)
        if not os.path.isfile(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if len(df) <= seen_rows:
            continue
        new_rows = df.iloc[seen_rows:]
        seen_rows = len(df)

        val_col = "val/mAP_50_95" if "val/mAP_50_95" in df.columns else None
        if val_col is None:
            continue  # column name may differ slightly by rfdetr version — check metrics.csv headers if this never prints

        for _, row in new_rows.iterrows():
            val = row.get(val_col)
            if pd.isna(val):
                continue  # this row was a train-step log, not a validation epoch
            epoch = row.get("epoch")
            epoch = int(epoch) if pd.notna(epoch) else "?"
            map50 = row.get("val/mAP_50", float("nan"))
            mar = row.get("val/mAR", float("nan"))
            ema = row.get("val/ema_mAP_50_95", float("nan"))

            improved = val > best_map
            if improved:
                best_map = val
                best_epoch = epoch

            line = f"[epoch {epoch}] val mAP50-95: {val:.4f}"
            if pd.notna(map50):
                line += f" | mAP50: {map50:.4f}"
            if pd.notna(mar):
                line += f" | mAR: {mar:.4f}"
            if pd.notna(ema):
                line += f" | EMA mAP50-95: {ema:.4f}"
            line += "  <-- NEW BEST" if improved else ""
            line += f"  (best so far: {best_map:.4f} @ epoch {best_epoch})"
            print(line)

threading.Thread(target=metrics_watcher, daemon=True).start()

# =========================================================
# Step 6: Train — settings chosen to mirror your Roboflow-hosted run
# =========================================================
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir=dataset_dir,
    output_dir=output_dir,
    epochs=100,                       # RF-DETR/Roboflow default cap — early stopping ends it sooner, same as your Roboflow run
    batch_size="auto",                # let RF-DETR pick the largest safe batch size for this GPU
    lr=1e-4,                          # RF-DETR default
    resolution=576,                   # matches the resolution used on Roboflow's hosted training run
    early_stopping=True,
    early_stopping_patience=10,       # RF-DETR default — matches the plateau-then-stop pattern seen on Roboflow
    early_stopping_min_delta=0.001,   # RF-DETR default
)

print("Training complete.")