
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(r"A:\PPE detection system")

SOURCE = ROOT / "runs" / "shwd_comparison" / "error_analysis"
OUTPUT = ROOT / "runs" / "shwd_comparison" / "comparison_figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "false_positives": "False Positive Comparison",
    "false_negatives": "False Negative Comparison",
    "correct_detections": "Correct Detection Comparison",
}

MODEL_DIRS = {
    "YOLO11n": SOURCE / "yolo11n",
    "YOLO12n": SOURCE / "yolo12n",
}

IMAGE_WIDTH = 550
IMAGE_HEIGHT = 360
HEADER_HEIGHT = 45
ROW_GAP = 24


def load_font(size=22):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_category_figure(folder, title):
    model_a = MODEL_DIRS["YOLO11n"] / folder
    model_b = MODEL_DIRS["YOLO12n"] / folder

    common = sorted(
        set(p.name for p in model_a.glob("*.jpg"))
        & set(p.name for p in model_b.glob("*.jpg"))
    )

    if not common:
        print(f"No common images for {folder}")
        return

    title_font = load_font(25)
    header_font = load_font(20)
    caption_font = load_font(15)

    total_height = (
        HEADER_HEIGHT
        + len(common) * (IMAGE_HEIGHT + ROW_GAP)
        + 30
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_WIDTH * 2 + 30, total_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (15, 8),
        title,
        fill="black",
        font=title_font
    )

    y = HEADER_HEIGHT

    for filename in common:
        img_a = Image.open(model_a / filename).convert("RGB")
        img_b = Image.open(model_b / filename).convert("RGB")

        img_a = ImageOps.fit(
            img_a,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        img_b = ImageOps.fit(
            img_b,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        draw.text(
            (10, y),
            f"YOLO11n | {filename}",
            fill="black",
            font=header_font
        )

        draw.text(
            (IMAGE_WIDTH + 25, y),
            f"YOLO12n | {filename}",
            fill="black",
            font=header_font
        )

        y += 28

        canvas.paste(img_a, (10, y))
        canvas.paste(img_b, (IMAGE_WIDTH + 25, y))

        y += IMAGE_HEIGHT + ROW_GAP

    output_path = OUTPUT / f"{folder}_comparison.png"

    canvas.save(output_path, quality=95)

    print(f"Saved: {output_path}")
    print(f"Images compared: {len(common)}")


def main():
    for folder, title in CATEGORIES.items():
        print(f"\nGenerating {title}...")
        create_category_figure(folder, title)

    print("\nCOMPARISON FIGURES GENERATED")
    print(f"Output folder: {OUTPUT}")


if __name__ == "__main__":
    main()
