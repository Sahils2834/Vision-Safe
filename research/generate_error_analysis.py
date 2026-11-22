
from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"A:\PPE detection system")

IMAGE_DIR = ROOT / "data" / "shwd_helmet_eval" / "images" / "test"
LABEL_DIR = ROOT / "data" / "shwd_helmet_eval" / "labels" / "test"

OUTPUT_DIR = ROOT / "runs" / "shwd_comparison" / "error_analysis"

MODELS = {
    "YOLO11n": ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt",
    "YOLO12n": ROOT / "runs" / "ppe_yolo12n" / "weights" / "best.pt",
}

CONF = 0.25
NMS_IOU = 0.70
MATCH_IOU = 0.45
IMGSZ = 640
MAX_EXAMPLES = 8


def load_gt(path, width, height):
    boxes = []

    if not path.exists():
        return np.empty((0, 4))

    for line in path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, values)

        if int(cls) != 1:
            continue

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])

    return np.array(boxes, dtype=float).reshape(-1, 4)


def calculate_iou(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))

    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    gt_area = (
        (gt[:, 2] - gt[:, 0]) *
        (gt[:, 3] - gt[:, 1])
    )

    pred_area = (
        (pred[:, 2] - pred[:, 0]) *
        (pred[:, 3] - pred[:, 1])
    )

    union = gt_area[:, None] + pred_area[None, :] - intersection

    return intersection / np.maximum(union, 1e-9)


def match_boxes(gt, pred):
    ious = calculate_iou(gt, pred)

    candidates = np.argwhere(ious >= MATCH_IOU)
    candidates = sorted(
        candidates,
        key=lambda pair: ious[pair[0], pair[1]],
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()
    pairs = []

    for gt_idx, pred_idx in candidates:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            pairs.append((gt_idx, pred_idx))

    false_gt = [i for i in range(len(gt)) if i not in matched_gt]
    false_pred = [i for i in range(len(pred)) if i not in matched_pred]

    return pairs, false_gt, false_pred


def draw_box(image, box, color, text):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    label_y = max(20, y1 - 7)

    cv2.putText(
        image,
        text,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def create_annotated_image(image_path, gt, pred, confs, pairs):
    image = cv2.imread(str(image_path))

    matched_gt = {g for g, p in pairs}
    matched_pred = {p for g, p in pairs}

    # Ground-truth boxes: green
    for i, box in enumerate(gt):
        draw_box(image, box, (0, 200, 0), "GT HELMET")

    # Matched predictions: cyan
    for i, box in enumerate(pred):
        if i in matched_pred:
            draw_box(
                image,
                box,
                (255, 200, 0),
                f"TP {confs[i]:.2f}",
            )
        else:
            draw_box(
                image,
                box,
                (0, 0, 255),
                f"FP {confs[i]:.2f}",
            )

    # Missed ground-truth boxes: magenta outline
    for i in range(len(gt)):
        if i not in matched_gt:
            draw_box(image, gt[i], (255, 0, 255), "FN MISSED")

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No JPG images found in {IMAGE_DIR}")

    all_csv_rows = []

    for model_name, weights in MODELS.items():
        print(f"\nProcessing {model_name}...")

        model = YOLO(str(weights))

        model_dir = OUTPUT_DIR / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        records = []

        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(f"Cannot read {image_path}")

            height, width = image.shape[:2]

            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            gt = load_gt(label_path, width, height)

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
                pred = result.boxes.xyxy.cpu().numpy().astype(float)
                confs = result.boxes.conf.cpu().numpy()
            else:
                pred = np.empty((0, 4))
                confs = np.empty((0,))

            pairs, missed_gt, false_pred = match_boxes(gt, pred)

            record = {
                "image": image_path.name,
                "path": image_path,
                "gt": gt,
                "pred": pred,
                "confs": confs,
                "pairs": pairs,
                "tp": len(pairs),
                "fp": len(false_pred),
                "fn": len(missed_gt),
                "max_fp_conf": max(
                    [float(confs[i]) for i in false_pred],
                    default=0.0,
                ),
            }

            records.append(record)

            all_csv_rows.append({
                "Model": model_name,
                "Image": image_path.name,
                "TP": record["tp"],
                "FP": record["fp"],
                "FN": record["fn"],
                "FP_max_confidence": record["max_fp_conf"],
            })

            if index % 200 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)}")

        categories = {
            "false_positives": [
                r for r in records if r["fp"] > 0
            ],
            "false_negatives": [
                r for r in records if r["fn"] > 0
            ],
            "correct_detections": [
                r for r in records if r["tp"] > 0
            ],
        }

        for category, candidates in categories.items():
            if category == "false_positives":
                candidates.sort(
                    key=lambda r: (r["max_fp_conf"], r["fp"]),
                    reverse=True,
                )
            elif category == "false_negatives":
                candidates.sort(
                    key=lambda r: (r["fn"], r["fp"]),
                    reverse=True,
                )
            else:
                candidates.sort(
                    key=lambda r: (r["tp"], -r["fp"]),
                    reverse=True,
                )

            category_dir = model_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for record in candidates[:MAX_EXAMPLES]:
                annotated = create_annotated_image(
                    record["path"],
                    record["gt"],
                    record["pred"],
                    record["confs"],
                    record["pairs"],
                )

                output_path = category_dir / record["image"]
                cv2.imwrite(str(output_path), annotated)

            print(
                f"{category}: saved "
                f"{min(len(candidates), MAX_EXAMPLES)} examples"
            )

    csv_path = OUTPUT_DIR / "image_error_analysis.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Image",
                "TP",
                "FP",
                "FN",
                "FP_max_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print("\nERROR ANALYSIS COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
