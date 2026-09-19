import supervision as sv
from rfdetr import RFDETRMedium
from IPython.display import Image, display

# 1. Define paths and classes
CHECKPOINT_PATH = ""
BASE_DATASET_PATH = ""  # Root folder containing /valid and /test
CLASSES = ["European Green Crab", "Jonah Crab", "Rock Crab"]

# 2. Load model once to save time
model = RFDETRMedium(pretrain_weights=CHECKPOINT_PATH, num_classes=len(CLASSES))

# 3. Inference callback (converts BGR to RGB)
def callback(image):
    detections = model.predict(image[:, :, ::-1].copy(), threshold=0.3)
    return detections[detections.class_id < len(CLASSES)]

# 4. Loop through both splits
for split in ["valid", "test"]:
    print(f"\n{'='*40}")
    print(f"Generating Confusion Matrix for: {split.upper()}")
    print(f"{'='*40}")
    
    # Load dataset for the current split
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{BASE_DATASET_PATH}/{split}",
        annotations_path=f"{BASE_DATASET_PATH}/{split}/_annotations.coco.json",
    )
    
    # Compute confusion matrix
    cm = sv.ConfusionMatrix.benchmark(
        dataset=ds,
        callback=callback,
        conf_threshold=0.3,
        iou_threshold=0.5,
    )
    
    # Save with normalize=False to get raw counts (not column-wise percentages)
    save_path = f"/kaggle/working/cm_best_total_{split}_raw.png"
    cm.plot(save_path=save_path, normalize=False)
    
    # Display the image inline
    display(Image(filename=save_path))
    
    # Optional: Print mAP metrics for the split as well
    map_metrics = sv.MeanAveragePrecision.benchmark(dataset=ds, callback=callback)
    print(f"{split.upper()} mAP@50: {map_metrics.map50:.4f}")