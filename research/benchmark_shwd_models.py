
from pathlib import Path
import csv
import time
import torch
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"A:\PPE detection system")

MODELS = {
    "YOLO11n": ROOT / "runs" / "ppe_yolo11n" / "weights" / "best.pt",
    "YOLO12n": ROOT / "runs" / "ppe_yolo12n" / "weights" / "best.pt",
}

IMAGE_DIR = ROOT / "data" / "shwd_helmet_eval" / "images" / "test"
OUTPUT_DIR = ROOT / "runs" / "shwd_comparison"

IMG_SIZE = 640
WARMUP = 30
ITERATIONS = 200

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))

    images = sorted(IMAGE_DIR.glob("*.jpg"))

    if not images:
        raise FileNotFoundError("No test images found.")

    image_path = str(images[0])
    print("Benchmark image:", image_path)

    results = []

    for name, checkpoint in MODELS.items():
        print("\n" + "=" * 55)
        print("Benchmarking:", name)
        print("=" * 55)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))

        model.model.to(device)
        model.model.eval()

        # Model information
        parameters = sum(
            p.numel() for p in model.model.parameters()
        )

        model_size_mb = checkpoint.stat().st_size / (1024 ** 2)

        # Warm-up
        print("Running warm-up...")
        for _ in range(WARMUP):
            model.predict(
                source=image_path,
                imgsz=IMG_SIZE,
                device=0,
                conf=0.25,
                verbose=False
            )

        torch.cuda.synchronize()

        # Timed inference
        print("Running", ITERATIONS, "iterations...")

        timings = []

        for _ in range(ITERATIONS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            model.predict(
                source=image_path,
                imgsz=IMG_SIZE,
                device=0,
                conf=0.25,
                verbose=False
            )

            torch.cuda.synchronize()
            end = time.perf_counter()

            timings.append((end - start) * 1000)

        avg_ms = float(np.mean(timings))
        median_ms = float(np.median(timings))
        fps = 1000.0 / avg_ms

        row = {
            "Model": name,
            "Parameters": parameters,
            "Model Size (MB)": round(model_size_mb, 3),
            "Average Inference (ms)": round(avg_ms, 3),
            "Median Inference (ms)": round(median_ms, 3),
            "Approx FPS": round(fps, 2),
        }

        results.append(row)

        for key, value in row.items():
            print(f"{key}: {value}")

        del model
        torch.cuda.empty_cache()

    output_file = OUTPUT_DIR / "computational_comparison.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("FINAL COMPUTATIONAL COMPARISON")
    print("=" * 55)

    for row in results:
        print("\n", row)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
