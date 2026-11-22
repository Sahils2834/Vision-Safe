
import torch
from ultralytics import YOLO


def main():
    print("=" * 50)
    print("YOLO12n TRAINING RESUMPTION")
    print("=" * 50)

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = r"runs\ppe_yolo12n\weights\last.pt"

    print("\nLoading checkpoint:", checkpoint)

    model = YOLO(checkpoint)

    print("\nResuming previous training...")
    print("Workers: 0 (Windows compatibility)")

    results = model.train(
        resume=True,
        workers=0
    )

    print("\nTraining operation finished.")
    print("Results:", results)


if __name__ == "__main__":
    main()
