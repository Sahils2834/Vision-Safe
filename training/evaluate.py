"""Comprehensive evaluation script for VisionSafe PPE Detection Models.

Calculates mAP50, mAP50-95, Precision, Recall, F1-score per class and overall.
Saves json report to training/evaluation_results.json
"""
import sys
import json
from pathlib import Path
from ultralytics import YOLO

def evaluate_model(model_path: str, data_yaml: str = "training/data.yaml", split: str = "val"):
    print(f"\n==================================================")
    print(f" Evaluating Model: {model_path}")
    print(f"==================================================")
    
    path = Path(model_path)
    if not path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        return None
        
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split, device="cpu", verbose=False)
    
    # Extract metrics
    mp = float(results.box.mp)
    mr = float(results.box.mr)
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)
    f1 = 2 * (mp * mr) / (mp + mr + 1e-8)
    
    metrics = {
        "model": model_path,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "f1_score": round(f1, 4),
        "class_metrics": {}
    }
    
    class_names = results.names
    for idx, name in class_names.items():
        if idx < len(results.box.p):
            cp = float(results.box.p[idx])
            cr = float(results.box.r[idx])
            cf1 = 2 * (cp * cr) / (cp + cr + 1e-8)
            metrics["class_metrics"][name] = {
                "precision": round(cp, 4),
                "recall": round(cr, 4),
                "f1_score": round(cf1, 4)
            }
            
    print(f"\nResults for {Path(model_path).name}:")
    print(f"  mAP50:     {metrics['mAP50'] * 100:.2f}%")
    print(f"  mAP50-95:  {metrics['mAP50_95'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {metrics['f1_score'] * 100:.2f}%")
    
    return metrics

if __name__ == "__main__":
    models_to_eval = sys.argv[1:] if len(sys.argv) > 1 else ["yolo11n.pt"]
    all_results = {}
    for m in models_to_eval:
        res = evaluate_model(m)
        if res:
            all_results[m] = res
            
    out_file = Path("training/evaluation_results.json")
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved evaluation metrics to {out_file.resolve()}")
