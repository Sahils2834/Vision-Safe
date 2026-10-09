import json
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = str(ROOT / "training" / "data.yaml")

def evaluate_models():
    models_to_test = ["yolo11n", "yolo12n", "yolo26n"]
    results_data = {}
    
    for m in models_to_test:
        try:
            print(f"Evaluating {m}...")
            # Look in weights, root, or backend/models
            model_file = None
            for cand in [
                ROOT / "weights" / f"{m}.pt",
                ROOT / "weights" / f"ppe_{m}.pt",
                ROOT / f"{m}.pt",
                ROOT / "backend" / "models" / f"{m}.pt",
            ]:
                if cand.exists():
                    model_file = str(cand)
                    break
            
            if not model_file:
                model_file = f"{m}.pt"
                
            model = YOLO(model_file)
            metrics = model.val(data=DATA_YAML, split="val", device="cpu", verbose=False)
            
            p = float(metrics.box.mp) if hasattr(metrics.box, 'mp') else float(metrics.box.p.mean())
            r = float(metrics.box.mr) if hasattr(metrics.box, 'mr') else float(metrics.box.r.mean())
            f1 = (2 * p * r) / (p + r + 1e-8) if (p + r) > 0 else 0.0
            map50 = float(metrics.box.map50)
            map50_95 = float(metrics.box.map)
            
            results_data[m] = {
                "mAP50": map50,
                "mAP50-95": map50_95,
                "precision": p,
                "recall": r,
                "f1": f1
            }
            print(f"✓ {m} -> mAP50: {map50*100:.2f}%, Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F1: {f1*100:.2f}%")
        except Exception as e:
            print(f"Error evaluating {m}: {e}")
            results_data[m] = {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
            
    # Real ensemble evaluation should be done, but for now we won't fabricate results.
    # We will just evaluate what we have.

    # Save to baseline_results.json
    root_json = ROOT / "baseline_results.json"
    with open(root_json, "w") as f:
        json.dump(results_data, f, indent=4)
        
    training_json = ROOT / "training" / "baseline_results.json"
    with open(training_json, "w") as f:
        json.dump(results_data, f, indent=4)
        
    print(f"\n✓ Saved baseline_results.json to {root_json}")
    
    # Print Markdown table
    print("\n| Model | Accuracy* | Precision | Recall | Macro F1 | mAP50 | mAP50-95 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print("| Base Paper | 94.69% | 94.77% | — | 94.65% | N/A | N/A |")
    for m in ["yolo11n", "yolo12n", "yolo26n", "ensemble"]:
        if m in results_data:
            r = results_data[m]
            print(f"| {m} | N/A | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | {r['f1']*100:.2f}% | {r['mAP50']*100:.2f}% | {r['mAP50-95']*100:.2f}% |")

    return results_data

if __name__ == "__main__":
    evaluate_models()
