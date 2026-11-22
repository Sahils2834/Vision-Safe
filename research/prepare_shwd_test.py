
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

SOURCE = Path(
    r"Safety-Helmet-Wearing-Dataset-master\VOC2028"
)

OUTPUT = Path(r"data\shwd_helmet_eval")

IMAGE_SOURCE = SOURCE / "JPEGImages"
ANNOTATION_SOURCE = SOURCE / "Annotations"
TEST_FILE = SOURCE / "ImageSets" / "Main" / "test.txt"

IMAGE_OUTPUT = OUTPUT / "images" / "test"
LABEL_OUTPUT = OUTPUT / "labels" / "test"

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
LABEL_OUTPUT.mkdir(parents=True, exist_ok=True)

if not TEST_FILE.exists():
    raise FileNotFoundError(TEST_FILE)

test_ids = [
    line.strip()
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("SHWD TEST DATASET PREPARATION")
print("=" * 50)
print("Test IDs:", len(test_ids))

counts = Counter()
missing = []
unexpected = []
total_helmet_boxes = 0

for index, image_id in enumerate(test_ids, start=1):

    image_path = IMAGE_SOURCE / f"{image_id}.jpg"
    xml_path = ANNOTATION_SOURCE / f"{image_id}.xml"

    if not image_path.exists() or not xml_path.exists():
        missing.append(image_id)
        continue

    root = ET.parse(xml_path).getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing image dimensions: {image_id}")

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {image_id}")

    labels = []

    for obj in root.findall("object"):

        class_name = obj.findtext("name", "").strip().lower()

        if class_name == "person":
            counts["unhelmeted_head_regions"] += 1
            continue

        if class_name == "hat":
            box = obj.find("bndbox")

            if box is None:
                raise ValueError(f"Missing box: {image_id}")

            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))
            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"Invalid helmet box in {image_id}"
                )

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Original model class mapping:
            # 0 = person
            # 1 = helmet
            labels.append(
                f"1 {x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )

            counts["hat"] += 1
            total_helmet_boxes += 1

        else:
            unexpected.append((image_id, class_name))

    destination_image = IMAGE_OUTPUT / image_path.name
    destination_label = LABEL_OUTPUT / f"{image_id}.txt"

    shutil.copy2(image_path, destination_image)

    # Empty label files are intentionally retained
    # for test images without helmet ground truth.
    destination_label.write_text(
        "\n".join(labels),
        encoding="utf-8"
    )

    counts["images_processed"] += 1

    if index % 100 == 0:
        print(f"Processed {index}/{len(test_ids)}")

if missing:
    print("\nMissing files:", missing[:20])
    raise RuntimeError("Some test images or annotations are missing.")

if unexpected:
    print("\nUnexpected classes:", unexpected[:20])
    raise RuntimeError("Unexpected test annotation classes found.")

yaml_content = """path: data/shwd_helmet_eval
train: images/test
val: images/test
test: images/test

names:
  0: person
  1: helmet
"""

(OUTPUT / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)

print("\n" + "=" * 50)
print("DATASET PREPARATION COMPLETE")
print("Images processed:", counts["images_processed"])
print("Helmet ground-truth boxes:", counts["hat"])
print("Unhelmeted-head negative regions:", counts["unhelmeted_head_regions"])
print("Unexpected classes:", len(unexpected))
print("Missing files:", len(missing))
print("Output:", OUTPUT.resolve())
print("=" * 50)
