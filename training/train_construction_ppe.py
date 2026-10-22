"""Training pipeline for Fine-Tuning YOLO on Construction PPE Kit & Gloves dataset."""
import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "construction_ppe_subset" / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
BACKEND_MODELS = ROOT / "backend" / "models"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_MODELS.mkdir(parents=True, exist_ok=True)

def train_construction_ppe(
    base_model: str = "yolo11n.pt",
    data_yaml: Path = DEFAULT_DATA,
    epochs: int = 25,
    batch: int = 8,
    imgsz: int = 640,
    device: str = None,
    run_name: str = "ppe_kit_gloves_yolo11n"
):
    print("=" * 65)
    print(f" VisionSafe - Fine-Tuning on Construction PPE Kit & Gloves")
    print(f" Base Model : {base_model}")
    print(f" Dataset    : {data_yaml}")
    print(f" Epochs     : {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    print("=" * 65)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run build_dataset.py first.")

    # Base model weight lookup
    base_path = Path(base_model)
    if not base_path.exists():
        if (ROOT / base_model).exists():
            base_path = ROOT / base_model
        elif (WEIGHTS_DIR / base_model).exists():
            base_path = WEIGHTS_DIR / base_model

    model = YOLO(str(base_path))

    if device is None:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ROOT / "runs"),
        name=run_name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )

    best_weights = ROOT / "runs" / "detect" / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = ROOT / "runs" / run_name / "weights" / "best.pt"

    if best_weights.exists():
        target_name = f"{run_name}.pt"
        saved_weights = WEIGHTS_DIR / target_name
        backend_weights = BACKEND_MODELS / target_name
        
        shutil.copy2(best_weights, saved_weights)
        shutil.copy2(best_weights, backend_weights)
        print(f"\n✓ Saved fine-tuned weights:")
        print(f"  -> {saved_weights}")
        print(f"  -> {backend_weights}")

        # Validation
        val_model = YOLO(str(saved_weights))
        print(f"Model validation loaded successfully. Classes: {val_model.names}")
        return saved_weights
    else:
        print(f"✗ Best weights not found at {best_weights}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Construction PPE & Gloves")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights (yolo11n.pt, yolo12n.pt, yolo26n.pt)")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default=None, help="Device (0 or cpu)")
    parser.add_argument("--name", default="ppe_kit_gloves_yolo11n", help="Run identifier name")
    args = parser.parse_args()

    train_construction_ppe(
        base_model=args.model,
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.name
    )

if __name__ == "__main__":
    main()
