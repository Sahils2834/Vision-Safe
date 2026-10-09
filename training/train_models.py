"""Training script for fine-tuning YOLO11n, YOLO12n, and YOLO26n on VOC2028 PPE dataset."""
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = str(ROOT / "training" / "data.yaml")
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

def train_model(base_weights: str, name: str, epochs: int = 25, batch: int = 8, imgsz: int = 640):
    print(f"\n==================================================")
    print(f" Starting Fine-Tuning: {name} (base: {base_weights})")
    print(f"==================================================")
    
    model = YOLO(base_weights)
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        device=0,
        project="runs",
        name=name,
        patience=10,
        save=True,
        plots=True,
        verbose=True
    )
    
    # Save best model to weights/
    best_weights = Path("runs") / "detect" / name / "weights" / "best.pt"
    target_weights = WEIGHTS_DIR / f"{name}.pt"
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, target_weights)
        print(f"Saved best weights to {target_weights}")
    return target_weights

if __name__ == "__main__":
    train_model("yolo11n.pt", "ppe_yolo11n", epochs=12, batch=8)
    train_model("yolo12n.pt", "ppe_yolo12n", epochs=12, batch=8)
    train_model("yolo26n.pt", "ppe_yolo26n", epochs=12, batch=8)
