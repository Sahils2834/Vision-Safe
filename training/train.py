"""Fine-tune the included YOLO11 base weights on the converted helmet dataset."""
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data" / "ppe_yolo" / "images" / "train"
    if not dataset.exists():
        raise SystemExit("Dataset not prepared. Run: python prepare_voc.py")
    model = YOLO(str(root / "yolo11n.pt"))
    model.train(data=str(Path(__file__).with_name("data.yaml")), epochs=80, imgsz=640, batch=8, project="runs", name="ppe-yolo11")

