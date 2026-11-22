
from pathlib import Path
import csv
import torch
from ultralytics import YOLO

ROOT = Path(r"A:\PPE detection system")

DATA = ROOT / "data" / "shwd_helmet_eval" / "data.yaml"

MODELS = {
    "YOLO11n": ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt",
    "YOLO12n": ROOT / "runs" / "ppe_yolo12n" / "weights" / "best.pt",
}

OUTPUT = ROOT / "runs" / "shwd_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300

def main():

    print("=" * 65)
    print("VISIONSAFE - SHWD INDEPENDENT TEST EVALUATION")
    print("=" * 65)

    print("Ultralytics evaluation version:")
    import ultralytics
    print(ultralytics.__version__)

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    results_table = []

    for model_name, checkpoint in MODELS.items():

        print("\n" + "=" * 65)
        print("Evaluating:", model_name)
        print("Checkpoint:", checkpoint)
        print("=" * 65)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))

        print("Model classes:", model.names)

        if model.names.get(1) != "helmet":
            raise ValueError(
                f"Expected class 1 to be helmet, got {model.names}"
            )

        metrics = model.val(
            data=str(DATA),
            split="test",
            imgsz=IMGSZ,
            batch=4,
            conf=CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            classes=[1],
            device=0,
            workers=0,
            plots=True,
            save_json=True,
            project=str(OUTPUT),
            name=model_name.lower(),
            exist_ok=True,
            verbose=True
        )

        row = {
            "Model": model_name,
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }

        results_table.append(row)

        print("\nRESULTS:", model_name)

        for key, value in row.items():
            if key != "Model":
                print(f"{key}: {value:.6f}")

    csv_path = OUTPUT / "shwd_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95"
            ]
        )
        writer.writeheader()
        writer.writerows(results_table)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for row in results_table:
        print("\n", row["Model"])
        for key in ["Precision", "Recall", "mAP50", "mAP50-95"]:
            print(f"{key}: {row[key] * 100:.2f}%")

    print("\nComparison saved to:", csv_path)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
