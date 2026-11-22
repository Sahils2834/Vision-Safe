
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from ultralytics import YOLO


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(r"A:\PPE detection system")

IMAGE_DIR = ROOT / "data" / "shwd_helmet_eval" / "images" / "test"
LABEL_DIR = ROOT / "data" / "shwd_helmet_eval" / "labels" / "test"

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "shwd_comparison"
    / "helmet_only_diagnostic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "YOLO11n": ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt",
    "YOLO12n": ROOT / "runs" / "ppe_yolo12n" / "weights" / "best.pt",
}


# --------------------------------------------------
# EVALUATION SETTINGS
# --------------------------------------------------

IMGSZ = 640
CONF = 0.25
NMS_IOU = 0.70
MATCH_IOU = 0.45
DEVICE = 0


# --------------------------------------------------
# LOAD GROUND-TRUTH HELMET BOXES
# --------------------------------------------------

def load_ground_truth(label_path, image_width, image_height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            # Only helmet class
            if class_id != 1:
                continue

            x_center, y_center, width, height = map(
                float, parts[1:]
            )

            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            boxes.append([x1, y1, x2, y2])

    return boxes


# --------------------------------------------------
# CALCULATE IOU
# --------------------------------------------------

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# GREEDY ONE-TO-ONE MATCHING
# --------------------------------------------------

def match_boxes(ground_truth, predictions):

    candidates = []

    for gt_index, gt_box in enumerate(ground_truth):

        for pred_index, pred_box in enumerate(predictions):

            iou = calculate_iou(gt_box, pred_box)

            if iou >= MATCH_IOU:

                candidates.append(
                    (iou, gt_index, pred_index)
                )

    # Highest IoU candidates matched first
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

    tp = len(matched_gt)
    fn = len(ground_truth) - tp
    fp = len(predictions) - len(matched_predictions)

    return tp, fp, fn


# --------------------------------------------------
# DRAW DIAGNOSTIC MATRIX
# --------------------------------------------------

def save_matrix(model_name, tp, fp, fn):

    # TN is intentionally unavailable.
    matrix = np.array([
        [tp, fn],
        [fp, np.nan]
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))

    masked_matrix = np.ma.masked_invalid(matrix)

    image = ax.imshow(
        masked_matrix,
        cmap="Blues",
        interpolation="nearest"
    )

    labels = [
        ["TP", "FN"],
        ["FP", "TN"]
    ]

    for row in range(2):
        for col in range(2):

            value = matrix[row, col]

            if np.isnan(value):
                display_text = "N/A\nNot measured"
            else:
                display_text = f"{labels[row][col]}\n{int(value):,}"

            ax.text(
                col,
                row,
                display_text,
                ha="center",
                va="center",
                fontsize=13,
                color="black"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted helmet", "Predicted background"])
    ax.set_yticklabels(["Actual helmet", "No annotated helmet"])

    ax.set_title(
        f"{model_name} - Helmet-only diagnostic\n"
        f"Confidence={CONF}, Matching IoU={MATCH_IOU}"
    )

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")

    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{model_name.lower()}_diagnostic.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved matrix:", output_path)


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("VISIONSAFE - HELMET-ONLY DIAGNOSTIC")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(DEVICE))

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in {IMAGE_DIR}"
        )

    results = []

    for model_name, checkpoint in MODELS.items():

        print("\n" + "=" * 65)
        print("Evaluating:", model_name)
        print("=" * 65)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))

        if model.names.get(1) != "helmet":
            raise ValueError(
                f"Expected class 1 to be helmet. Got: {model.names}"
            )

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_predictions = 0

        for index, image_path in enumerate(image_paths, start=1):

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            ground_truth = load_ground_truth(
                label_path,
                image_width,
                image_height
            )

            prediction_result = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=NMS_IOU,
                classes=[1],
                device=DEVICE,
                verbose=False
            )[0]

            predictions = []

            if prediction_result.boxes is not None:
                predictions = (
                    prediction_result.boxes.xyxy
                    .cpu()
                    .numpy()
                    .tolist()
                )

            tp, fp, fn = match_boxes(
                ground_truth,
                predictions
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            total_gt += len(ground_truth)
            total_predictions += len(predictions)

            if index % 100 == 0:
                print(
                    f"Processed {index}/{len(image_paths)} images"
                )

        precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 0.0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        row = {
            "Model": model_name,
            "Images": len(image_paths),
            "Ground_Truth_Helmets": total_gt,
            "Predictions": total_predictions,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": "Not measured",
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Confidence": CONF,
            "NMS_IoU": NMS_IOU,
            "Matching_IoU": MATCH_IOU
        }

        results.append(row)

        print("\nRESULT:", model_name)

        for key, value in row.items():
            print(f"{key}: {value}")

        save_matrix(
            model_name,
            total_tp,
            total_fp,
            total_fn
        )

    csv_path = OUTPUT_DIR / "helmet_only_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print("CSV:", csv_path)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
