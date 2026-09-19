# RF-DETR Medium: full evaluation for a Kaggle notebook
# Paste each "# %%" block into its own cell (or run the file top to bottom).
#
# Produces, in /kaggle/working/eval_results:
#   training_curves.png            loss, mAP, precision/recall/F1, learning rate
#   checkpoint_comparison.csv      COCO mAP for every checkpoint on valid and test
#   per_class_results.csv          per-class AP, recall and precision
#   cm_<checkpoint>_<split>.png    confusion matrix (normalized)
#   cm_<checkpoint>_<split>_counts.png   confusion matrix (raw counts)

# %% Cell 1: install (uncomment in Kaggle if the packages are missing)
# !pip -q install -U supervision rfdetr pycocotools

# %% Cell 2: config
import os, glob, gc, io, json, hashlib, contextlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import supervision as sv
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRMedium

W = "/kaggle/working/rfdetr_deploy"          # training output folder
OUT = "/kaggle/working/eval_results"
os.makedirs(OUT, exist_ok=True)

SPLITS = ["valid", "test"]
CONF = 0.3          # threshold used for the confusion matrix only
IOU = 0.5

# Dataset root: auto-detected. Set DATA by hand if this finds nothing or several.
hits = glob.glob("/kaggle/input/**/valid/_annotations.coco.json", recursive=True)
print("annotation files found:", hits)
DATA = os.path.dirname(os.path.dirname(hits[0]))   # e.g. /kaggle/input/x5-ds-combined
# DATA = "/kaggle/input/x5-ds-combined"
print("DATA =", DATA)

cfg = json.load(open(f"{W}/training_config.json"))
CLASSES = cfg["class_names"]
NUM_CLASSES = len(CLASSES)
print("classes:", CLASSES)

CKPTS = {
    "best_ema":   f"{W}/checkpoint_best_ema.pth",
    "best_total": f"{W}/checkpoint_best_total.pth",
    "last_ema":   f"{W}/last_ema.pth",
}

# %% Cell 3: training curves from metrics.csv
df = pd.read_csv(f"{W}/metrics.csv")
ep = df.groupby("epoch").mean(numeric_only=True)     # step rows and epoch rows are separate

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].plot(ep["train/loss"].dropna(), marker="o"); ax[0, 0].set_title("Total train loss")
for c, l in [("train/loss_bbox", "bbox (L1)"), ("train/loss_giou", "GIoU"), ("train/loss_ce", "classification")]:
    if c in ep: ax[0, 1].plot(ep[c].dropna(), marker="o", label=l)
ax[0, 1].set_title("Train loss components"); ax[0, 1].legend()
if "train/class_error" in ep:
    ax[0, 2].plot(ep["train/class_error"].dropna(), marker="o", color="tab:red")
ax[0, 2].set_title("Train class error (%)")

val = ep[ep["val/mAP_50"].notna()]
for c, l in [("val/mAP_50", "mAP@50"), ("val/mAP_50_95", "mAP@50:95"), ("val/mAP_75", "mAP@75")]:
    ax[1, 0].plot(val[c], marker="o", label=l)
best = val["val/mAP_50_95"].idxmax()
ax[1, 0].axvline(best, color="gray", ls="--", alpha=.6)
ax[1, 0].set_title(f"Validation mAP (best mAP@50:95 at epoch {int(best)})"); ax[1, 0].legend()
for c, l in [("val/precision", "precision"), ("val/recall", "recall"), ("val/F1", "F1")]:
    ax[1, 1].plot(val[c], marker="o", label=l)
ax[1, 1].set_title("Validation precision / recall / F1"); ax[1, 1].legend()
lr = df[df["train/lr"].notna()]
ax[1, 2].plot(lr["step"], lr["train/lr"]); ax[1, 2].set_title("Learning rate"); ax[1, 2].set_xlabel("step")
for a in ax.flat:
    a.grid(alpha=.3)
    if a is not ax[1, 2]: a.set_xlabel("epoch")
plt.tight_layout()
plt.savefig(f"{OUT}/training_curves.png", dpi=140)
plt.show()

# %% Cell 4: helpers
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def quiet():
    return contextlib.redirect_stdout(io.StringIO())

def load_gt(split):
    """Ground truth straight from the COCO json. Class ids follow CLASSES (0..N-1),
    so the dummy category 0 that Roboflow adds never enters the picture."""
    with quiet():
        coco = COCO(f"{DATA}/{split}/_annotations.coco.json")
    cat2idx = {c["id"]: CLASSES.index(c["name"])
               for c in coco.dataset["categories"] if c["name"] in CLASSES}
    targets = {}
    for img_id in coco.getImgIds():
        anns = [a for a in coco.imgToAnns[img_id] if a["category_id"] in cat2idx]
        if anns:
            xyxy = np.array([[a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                             for a in anns], dtype=float)
            cid = np.array([cat2idx[a["category_id"]] for a in anns], dtype=int)
            targets[img_id] = sv.Detections(xyxy=xyxy, class_id=cid)
        else:
            targets[img_id] = sv.Detections.empty()
    return coco, targets

def predict_split(model, coco, split):
    """One prediction pass per checkpoint and split, reused for mAP and confusion matrix."""
    preds = {}
    for img_id, info in coco.imgs.items():
        im = Image.open(f"{DATA}/{split}/{info['file_name']}").convert("RGB")
        d = model.predict(im, threshold=0.01)          # low threshold so mAP sees everything
        if len(d):
            d = d[(d.class_id >= 0) & (d.class_id < NUM_CLASSES)]   # drop the stray extra class id
        preds[img_id] = d
    return preds

def coco_metrics(coco, preds):
    name2cat = {c["name"]: c["id"] for c in coco.dataset["categories"]}
    results = []
    for img_id, d in preds.items():
        if len(d) == 0:
            continue
        for (x1, y1, x2, y2), s, c in zip(d.xyxy, d.confidence, d.class_id):
            results.append({"image_id": img_id,
                            "category_id": name2cat[CLASSES[int(c)]],
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(s)})
    if not results:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "mAP75": 0.0}, {n: (0.0, 0.0) for n in CLASSES}

    with quiet():
        E = COCOeval(coco, coco.loadRes(results), "bbox")
        E.params.catIds = [name2cat[n] for n in CLASSES]   # so index k matches CLASSES[k]
        E.evaluate(); E.accumulate(); E.summarize()

    overall = {"mAP50": float(E.stats[1]), "mAP50_95": float(E.stats[0]), "mAP75": float(E.stats[2])}
    prec = E.eval["precision"]                              # [iou, recall, class, area, maxDets]
    per_class = {}
    for k, name in enumerate(CLASSES):
        p = prec[:, :, k, 0, 2];  p = p[p > -1]
        p50 = prec[0, :, k, 0, 2]; p50 = p50[p50 > -1]
        per_class[name] = (float(p.mean()) if p.size else float("nan"),
                           float(p50.mean()) if p50.size else float("nan"))
    return overall, per_class

# %% Cell 5: evaluate every checkpoint on every split
hashes = {n: md5(p) for n, p in CKPTS.items()}
unique = {}
for n, h in hashes.items():
    unique.setdefault(h, n)
for n, h in hashes.items():
    if unique[h] != n:
        print(f"{n} is byte-identical to {unique[h]}; evaluating once and reusing the result")

GT = {s: load_gt(s) for s in SPLITS}
rows, class_rows = [], []

for h, name in unique.items():
    same = [n for n, hh in hashes.items() if hh == h]
    label = "+".join(same)
    model = RFDETRMedium(pretrain_weights=CKPTS[name], num_classes=NUM_CLASSES)
    names = model.class_names
    names = list(names.values()) if isinstance(names, dict) else list(names)
    assert names == CLASSES, f"class mismatch: model {names} vs config {CLASSES}"

    for split in SPLITS:
        coco, targets = GT[split]
        preds = predict_split(model, coco, split)
        overall, per_class = coco_metrics(coco, preds)

        ids = list(preds)
        cm = sv.ConfusionMatrix.from_detections(
            predictions=[preds[i] for i in ids],
            targets=[targets[i] for i in ids],
            classes=CLASSES, conf_threshold=CONF, iou_threshold=IOU,
        )
        cm.plot(save_path=f"{OUT}/cm_{label}_{split}.png", normalize=True,
                title=f"{label} / {split} (normalized)")
        cm.plot(save_path=f"{OUT}/cm_{label}_{split}_counts.png", normalize=False,
                title=f"{label} / {split} (counts)")
        plt.close("all")

        m = cm.matrix.astype(float)          # rows = true, cols = predicted, last row = FP, last col = FN
        for i, cname in enumerate(CLASSES):
            tp = m[i, i]
            for n in same:
                class_rows.append({
                    "checkpoint": n, "split": split, "class": cname,
                    "AP50_95": round(per_class[cname][0], 4),
                    "AP50": round(per_class[cname][1], 4),
                    "recall": round(tp / m[i].sum(), 4) if m[i].sum() else np.nan,
                    "precision": round(tp / m[:, i].sum(), 4) if m[:, i].sum() else np.nan,
                    "n_real": int(m[i].sum()),
                })
        for n in same:
            rows.append({"checkpoint": n, "split": split, **{k: round(v, 4) for k, v in overall.items()}})
        print(f"done: {label} / {split}")

    del model; gc.collect(); torch.cuda.empty_cache()

# %% Cell 6: results
res = pd.DataFrame(rows).sort_values(["checkpoint", "split"])
res.to_csv(f"{OUT}/checkpoint_comparison.csv", index=False)
print(res.pivot(index="checkpoint", columns="split", values=["mAP50", "mAP50_95", "mAP75"]))

pc = pd.DataFrame(class_rows).sort_values(["checkpoint", "split", "class"])
pc.to_csv(f"{OUT}/per_class_results.csv", index=False)
print(pc.to_string(index=False))

# Pick on valid, report on test
best_ckpt = res[res.split == "valid"].sort_values("mAP50_95", ascending=False).iloc[0]["checkpoint"]
print("\nbest checkpoint on valid (mAP@50:95):", best_ckpt)
print(res[(res.checkpoint == best_ckpt) & (res.split == "test")].to_string(index=False))

# %% Cell 7 (optional): zip everything for download
# !cd /kaggle/working && zip -r eval_results.zip eval_results