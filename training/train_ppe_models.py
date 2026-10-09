"""Training script for fine-tuning YOLO11n, YOLO12n, and YOLO26n on VOC2028 PPE dataset."""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = str(ROOT / "training" / "data.yaml")
WEIGHTS_DIR = ROOT / "weights"
BACKEND_MODELS = ROOT / "backend" / "models"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_MODELS.mkdir(parents=True, exist_ok=True)

def train_model(base_weights: str, name: str, output_name: str, epochs: int = 15, batch: int = 16, imgsz: int = 640):
    print(f"\n{'='*60}")
    print(f" Starting Fine-Tuning: {name} (base: {base_weights}) -> {output_name}")
    print(f"{'='*60}")
    
    # Check if base weights exist in root, weights, or download
    base_path = Path(base_weights)
    if not base_path.exists():
        if (ROOT / base_weights).exists():
            base_path = ROOT / base_weights
        elif (WEIGHTS_DIR / base_weights).exists():
            base_path = WEIGHTS_DIR / base_weights
            
    model = YOLO(str(base_path))
    
    # Determine device (cuda if available else cpu)
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        device=device,
        project=str(ROOT / "runs"),
        name=name,
        patience=8,
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )
    
    # Copy best weights to weights/ and backend/models/
    best_weights = ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = ROOT / "runs" / name / "weights" / "best.pt"
        
    if best_weights.exists():
        target_weights = WEIGHTS_DIR / output_name
        shutil.copy2(best_weights, target_weights)
        
        backend_target = BACKEND_MODELS / output_name
        shutil.copy2(best_weights, backend_target)
        
        # Also copy to root for compatibility
        root_target = ROOT / output_name
        shutil.copy2(best_weights, root_target)
        
        print(f"✓ Saved best weights to {target_weights} and {backend_target}")
        
        # Verify
        verify_model = YOLO(str(target_weights))
        print(f"  Task: {verify_model.task}")
        print(f"  Classes: {verify_model.names}")
        return target_weights
    else:
        print(f"✗ Best weights not found at {best_weights}")
        return None

def main():
    print("=" * 60)
    print("PPE DETECTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Train 1: YOLO11n on PPE Dataset
    train_model("yolo11n.pt", "ppe_yolo11n", "yolo11n.pt", epochs=15, batch=16)
    
    # Train 2: YOLO12n on PPE Dataset
    train_model("yolo12n.pt", "ppe_yolo12n", "yolo12n.pt", epochs=15, batch=16)
    
    # Train 3: YOLO26n on PPE Dataset
    train_model("yolo26n.pt", "ppe_yolo26n", "yolo26n.pt", epochs=15, batch=16)
    
    print("\n" + "=" * 60)
    print("ALL 3 PPE DETECTION MODELS TRAINED & SAVED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
