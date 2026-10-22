#!/usr/bin/env python3
"""VisionSafe PPE Dataset Builder.

Assembles and curates a balanced training subset focusing on standard PPE:
- Person
- Helmet
- Gloves (Construction Gloves)
- Vest (Safety Vest)
- Boots
- Goggles

Remaps labels from the Ultralytics Construction-PPE dataset into standard classes:
0: person
1: helmet
2: gloves
3: vest
4: boots
5: goggles
"""
from pathlib import Path
import argparse
import shutil
import random
import yaml
from collections import Counter

# Standard Target PPE Classes
TARGET_NAMES = {
    0: "person",
    1: "helmet",
    2: "gloves",
    3: "vest",
    4: "boots",
    5: "goggles"
}

# Source Ultralytics Construction-PPE class IDs to Target standard classes
# Source IDs: 0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none,
# 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots
SOURCE_TO_TARGET = {
    6: 0,  # Person -> person
    0: 1,  # helmet -> helmet
    1: 2,  # gloves -> gloves
    2: 3,  # vest -> vest
    3: 4,  # boots -> boots
    4: 5   # goggles -> goggles
}

def parse_and_remap_labels(label_path: Path):
    """Parses a source label file and remaps boxes to standard target classes."""
    valid_boxes = []
    classes_found = set()
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    src_cls = int(float(parts[0]))
                    if src_cls in SOURCE_TO_TARGET:
                        tgt_cls = SOURCE_TO_TARGET[src_cls]
                        coords = [float(p) for p in parts[1:5]]
                        valid_boxes.append((tgt_cls, coords))
                        classes_found.add(tgt_cls)
                except (ValueError, IndexError):
                    pass
    return valid_boxes, classes_found

def locate_pairs(root: Path):
    """Recursively locates image and label pairs from the source directory."""
    pairs = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        parts = list(img.parts)
        if "images" not in parts:
            continue
        idx = len(parts) - 1 - parts[::-1].index("images")
        lp = parts[:]
        lp[idx] = "labels"
        label = Path(*lp).with_suffix(".txt")
        if label.exists():
            boxes, classes = parse_and_remap_labels(label)
            if boxes:  # only include images with at least one valid target PPE box
                pairs.append((img, label, boxes, classes))
    return pairs

def find_source_dir(cli_source: str = None) -> Path:
    """Finds the source dataset directory across common workspace paths."""
    candidates = []
    if cli_source:
        candidates.append(Path(cli_source))
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    
    candidates.extend([
        cwd / "_construction_ppe_source",
        script_dir.parent / "_construction_ppe_source",
        repo_root / "Safety-Helmet-Wearing-Dataset-master" / "_construction_ppe_source",
        repo_root / "_construction_ppe_source",
        Path("A:/PPE detection system/Safety-Helmet-Wearing-Dataset-master/_construction_ppe_source"),
        Path("A:/PPE detection system/_construction_ppe_source")
    ])

    for c in candidates:
        if c.exists() and (c / "images").exists() and (c / "labels").exists():
            return c.resolve()
            
    raise FileNotFoundError(
        f"Could not locate _construction_ppe_source. Checked candidates:\n" +
        "\n".join(str(c) for c in candidates)
    )

def main():
    parser = argparse.ArgumentParser(description="Curate balanced PPE and Gloves dataset")
    parser.add_argument("--source", type=str, default=None, help="Path to _construction_ppe_source")
    parser.add_argument("--gloves-limit", type=int, default=50, help="Max images sampled for Gloves")
    parser.add_argument("--ppe-limit", type=int, default=50, help="Max images sampled for PPE Kit (helmet, vest, boots, goggles)")
    parser.add_argument("--all", action="store_true", help="Include all available source images without limit")
    parser.add_argument("--out", type=str, default="A:/PPE detection system/data/construction_ppe_subset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    args = parser.parse_args()

    source_dir = find_source_dir(args.source)
    print(f"Source dataset found: {source_dir}")

    pairs = locate_pairs(source_dir)
    if not pairs:
        raise SystemExit(f"No valid image/label pairs with target PPE found in {source_dir}")

    # Separate candidates: gloves vs PPE kit
    gloves_candidates = [p for p in pairs if 2 in p[3]]  # class 2 = gloves
    ppe_kit_candidates = [p for p in pairs if any(c in p[3] for c in (1, 3, 4, 5))]  # helmet, vest, boots, goggles

    print(f"Candidate images available with Gloves: {len(gloves_candidates)}")
    print(f"Candidate images available with PPE Kit items: {len(ppe_kit_candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(gloves_candidates)
    rng.shuffle(ppe_kit_candidates)

    chosen = {}
    if args.all:
        for p in pairs:
            chosen[str(p[0].resolve())] = p
    else:
        for img, lbl, boxes, classes in gloves_candidates[:args.gloves_limit]:
            chosen[str(img.resolve())] = (img, lbl, boxes, classes)
        for img, lbl, boxes, classes in ppe_kit_candidates[:args.ppe_limit]:
            chosen[str(img.resolve())] = (img, lbl, boxes, classes)

    records = list(chosen.values())
    rng.shuffle(records)
    total_imgs = len(records)
    print(f"Selected {total_imgs} unique curated images (deduplicated).")

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # 80% train, 10% val, 10% test split
    n_val = max(1, round(total_imgs * 0.10)) if total_imgs >= 8 else 0
    n_test = max(1, round(total_imgs * 0.10)) if total_imgs >= 8 else 0
    n_train = total_imgs - n_val - n_test

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(splits)

    class_stats = Counter()
    for i, ((img_path, lbl_path, boxes, classes), split) in enumerate(zip(records, splits), 1):
        target_stem = f"ppe_{i:04d}_{img_path.stem}"
        target_img = out_dir / "images" / split / f"{target_stem}{img_path.suffix}"
        target_lbl = out_dir / "labels" / split / f"{target_stem}.txt"

        shutil.copy2(img_path, target_img)
        lines = [f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}" for cls_id, coords in boxes]
        target_lbl.write_text("\n".join(lines), encoding="utf-8")

        for cls_id, _ in boxes:
            class_stats[cls_id] += 1

    # Generate data.yaml
    yaml_data = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": TARGET_NAMES
    }
    data_yaml_path = out_dir / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("DATASET CURATION SUMMARY")
    print("=" * 60)
    print(f"Output Directory : {out_dir.resolve()}")
    print(f"Total Images     : {total_imgs} (train: {n_train}, val: {n_val}, test: {n_test})")
    print(f"Classes (6 Target):")
    for k in sorted(TARGET_NAMES.keys()):
        print(f"  Class {k} ({TARGET_NAMES[k]:10s}): {class_stats[k]} annotations")
    print(f"Config YAML      : {data_yaml_path.resolve()}")
    print("\nReady to train with YOLO:")
    print(f'yolo detect train data="{data_yaml_path.resolve()}" model=yolo11n.pt epochs=30 imgsz=640 batch=8')
    print("=" * 60)

if __name__ == "__main__":
    main()
