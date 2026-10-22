# VisionSafe PPE Dataset Builder

**Important:** this ZIP contains the dataset builder, not the 40–60 photo files. The current runtime could not download the source dataset, so I can't honestly claim that the requested image dataset is already included. Run the script below on a computer with internet access to download the public source and assemble the sample dataset.

## Build the dataset
1. Extract this ZIP.
2. Install Python 3.10+.
3. In a terminal in the extracted folder, run:
   ```bash
   pip install PyYAML
   python scripts/build_dataset.py --limit 30
   ```
4. It downloads the Ultralytics Construction-PPE archive, selects up to 30 images containing glove/no-glove annotations and up to 30 containing other PPE annotations, preserves YOLO bounding-box labels, and creates `VisionSafe_PPE_Dataset/`.

## Train
```bash
pip install ultralytics
yolo detect train data="VisionSafe_PPE_Dataset/data.yaml" model=yolo11n.pt epochs=50 imgsz=640
```

Classes include helmet, gloves, vest, boots, goggles, person, and missing-PPE classes. The glove and other-PPE groups may overlap because one image can contain both; the final training set deduplicates files.

Source documentation: https://docs.ultralytics.com/datasets/detect/construction-ppe/

Review the source dataset's current license and attribution terms before using or redistributing it. This small sample is useful for pipeline testing; use the full dataset and a separate test set for serious evaluation.
