"""Convert the full Safety-Helmet-Wearing Pascal VOC download into YOLO labels.

The repository placed in the project root contains only demo images. Download and
extract the full VOC2028 dataset first; it must contain Annotations, JPEGImages,
and ImageSets/Main. Then run this script from the training directory.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "ppe_yolo"
CLASSES = {"person": 0, "hat": 1, "helmet": 1}


def source_directory() -> Path | None:
    for candidate in (ROOT / "VOC2028", ROOT / "Safety-Helmet-Wearing-Dataset-master" / "VOC2028"):
        if (candidate / "Annotations").is_dir() and (candidate / "JPEGImages").is_dir(): return candidate
    return None


def convert(source: Path, image_id: str, split: str) -> bool:
    xml_path, image_path = source / "Annotations" / f"{image_id}.xml", source / "JPEGImages" / f"{image_id}.jpg"
    if not xml_path.exists() or not image_path.exists(): return False
    root = ET.parse(xml_path).getroot(); size = root.find("size")
    width, height = float(size.findtext("width", "0")), float(size.findtext("height", "0"))
    if not width or not height: return False
    lines = []
    for obj in root.findall("object"):
        label = CLASSES.get(obj.findtext("name", "").lower())
        box = obj.find("bndbox")
        if label is None or box is None: continue
        xmin, ymin = float(box.findtext("xmin", "0")), float(box.findtext("ymin", "0"))
        xmax, ymax = float(box.findtext("xmax", "0")), float(box.findtext("ymax", "0"))
        x, y = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
        w, h = (xmax - xmin) / width, (ymax - ymin) / height
        lines.append(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    destination = TARGET / "images" / split / image_path.name
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(image_path, destination)
    label_path = TARGET / "labels" / split / f"{image_id}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True); label_path.write_text("\n".join(lines), encoding="utf-8")
    return True


if __name__ == "__main__":
    source = source_directory()
    if source is None:
        raise SystemExit("Full VOC data is not present. Extract the folder containing Annotations, JPEGImages, and ImageSets as A:\\PPE detection system\\VOC2028 (or inside Safety-Helmet-Wearing-Dataset-master).")
    ids_file = source / "ImageSets" / "Main" / "trainval.txt"
    ids = ids_file.read_text().split() if ids_file.exists() else [p.stem for p in (source / "Annotations").glob("*.xml")]
    random.Random(42).shuffle(ids); cut = int(len(ids) * .9); count = 0
    for index, image_id in enumerate(ids): count += convert(source, image_id, "train" if index < cut else "val")
    print(f"Prepared {count} labelled images in {TARGET}")
