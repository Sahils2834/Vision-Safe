
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"A:\PPE detection system")
DATA = ROOT / "data" / "shwd_helmet_eval" / "data.yaml"
WEIGHTS = ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))

metrics = model.val(
    data=str(DATA),
    split="test",
    imgsz=640,
    batch=4,
    conf=0.001,
    iou=0.7,
    max_det=300,
    classes=[1],
    device=0,
    workers=0,
    plots=True,
    save_json=False,
    verbose=False,
)

print("\nMODEL NAMES:")
print(model.names)

print("\nEVALUATION METRICS:")
print(f"Precision: {metrics.box.mp:.6f}")
print(f"Recall: {metrics.box.mr:.6f}")
print(f"mAP50: {metrics.box.map50:.6f}")
print(f"mAP50-95: {metrics.box.map:.6f}")

print("\nCONFUSION MATRIX:")
cm = metrics.confusion_matrix.matrix
print(cm)

print("\nMATRIX SHAPE:", cm.shape)
print("COLUMN TOTALS:", cm.sum(axis=0))
print("ROW TOTALS:", cm.sum(axis=1))
print("GRAND TOTAL:", cm.sum())

print("\nCONFUSION MATRIX CLASS NAMES:")
print(metrics.confusion_matrix.names)
