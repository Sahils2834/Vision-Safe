
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(r"A:\PPE detection system")
IMAGE_DIR = ROOT / "data" / "shwd_helmet_eval" / "images" / "test"
LABEL_DIR = ROOT / "data" / "shwd_helmet_eval" / "labels" / "test"
OUTPUT_DIR = ROOT / "runs" / "shwd_comparison" / "binary_confusion"

MODELS = {
    "YOLO11n": ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt",
    "YOLO12n": ROOT / "runs" / "ppe_yolo12n" / "weights" / "best.pt",
}

# Prediction confidence and NMS IoU
CONF = 0.25
NMS_IOU = 0.7

# IoU required to match a predicted helmet with ground truth
MATCH_IOU = 0.45

IMGSZ = 640


def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return np.empty((0, 4))

    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, values)

        if int(cls) != 1:
            continue

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=float).reshape(-1, 4)


def calculate_iou(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    gt_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0])
        * (gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    pred_area = (
        (pred_boxes[:, 2] - pred_boxes[:, 0])
        * (pred_boxes[:, 3] - pred_boxes[:, 1])
    )

    union = gt_area[:, None] + pred_area[None, :] - intersection

    return intersection / np.maximum(union, 1e-9)


def match_boxes(gt_boxes, pred_boxes):
    ious = calculate_iou(gt_boxes, pred_boxes)

    candidates = np.argwhere(ious >= MATCH_IOU)
    candidates = sorted(
        candidates,
        key=lambda pair: ious[pair[0], pair[1]],
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()

    for gt_idx, pred_idx in candidates:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_gt)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(matched_pred)

    return tp, fp, fn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No JPG images found in {IMAGE_DIR}")

    rows = []

    for model_name, weights in MODELS.items():
        print(f"\nEvaluating {model_name}...")

        model = YOLO(str(weights))

        tp_total = 0
        fp_total = 0
        fn_total = 0

        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

            height, width = image.shape[:2]

            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            gt_boxes = load_ground_truth(label_path, width, height)

            result = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=NMS_IOU,
                classes=[1],
                device=0,
                verbose=False,
            )[0]

            if result.boxes is not None and len(result.boxes):
                pred_boxes = (
                    result.boxes.xyxy.cpu().numpy().astype(float)
                )
            else:
                pred_boxes = np.empty((0, 4))

            tp, fp, fn = match_boxes(gt_boxes, pred_boxes)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            if index % 100 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)}")

        precision = tp_total / max(tp_total + fp_total, 1)
        recall = tp_total / max(tp_total + fn_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        rows.append({
            "Model": model_name,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        # Binary confusion matrix: rows = predicted, columns = actual
        matrix = np.array([
            [tp_total, fp_total],
            [fn_total, 0],
        ])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Helmet", "Background"])
        ax.set_yticklabels(["Helmet", "Background"])

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(
            f"{model_name} Binary Helmet Confusion Matrix\n"
            f"Confidence={CONF}, Matching IoU={MATCH_IOU}"
        )

        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j, i, f"{value:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{model_name.lower()}_binary_cm.png",
            dpi=300,
        )
        plt.close()

        print(
            f"{model_name}: TP={tp_total}, FP={fp_total}, "
            f"FN={fn_total}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}"
        )

    csv_path = OUTPUT_DIR / "binary_confusion_results.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nFINAL RESULTS")
    for row in rows:
        print(row)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
