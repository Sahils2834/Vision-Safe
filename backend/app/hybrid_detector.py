
"""Hybrid PPE Detection Engine.

Combines YOLO object detection with:
1. Spatial PPE-to-worker association using box overlap and centre distance.
2. Helmet, gloves, vest, boots and goggles detection.
3. Person fallback detection from available models.
4. PPE enrichment from the fine-tuned PPE model.
5. Class-aware ensemble Non-Maximum Suppression.

Important:
The detector reports detected PPE. The monitor is responsible for
deciding compliance according to the configured site requirements.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class HybridPPEDetector:

    def __init__(self, head_ratio: float = 0.38):
        self.head_ratio = head_ratio

        self.ppe_labels = {
            "person": "person",
            "worker": "person",

            "hat": "helmet",
            "helmet": "helmet",
            "hardhat": "helmet",
            "safety_helmet": "helmet",

            "vest": "vest",
            "safety_vest": "vest",
            "ppe_kit": "vest",

            "glove": "gloves",
            "gloves": "gloves",
            "construction_gloves": "gloves",

            "boot": "boots",
            "boots": "boots",
            "shoe": "boots",
            "shoes": "boots",
            "safety_boots": "boots",

            "goggle": "goggles",
            "goggles": "goggles",
            "glasses": "goggles",
            "safety_glasses": "goggles",
        }

        self.ppe_categories = (
            "helmet",
            "gloves",
            "vest",
            "boots",
            "goggles",
        )

    # ---------------------------------------------------------
    # LABEL NORMALIZATION
    # ---------------------------------------------------------

    def standardize_label(self, label: str) -> str:
        normalized = (
            str(label)
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return self.ppe_labels.get(normalized, normalized)

    # ---------------------------------------------------------
    # GEOMETRY HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _valid_box(box) -> bool:
        if box is None:
            return False

        if len(box) != 4:
            return False

        x1, y1, x2, y2 = map(float, box)

        return (
            np.isfinite([x1, y1, x2, y2]).all()
            and x2 > x1
            and y2 > y1
        )

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = map(float, box)

        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        return (ix2 - ix1) * (iy2 - iy1)

    @classmethod
    def _overlap_ratio(cls, item_box, region_box) -> float:
        """Fraction of the PPE box overlapping a worker region."""

        item_area = cls._area(item_box)

        if item_area <= 0:
            return 0.0

        return cls._intersection(item_box, region_box) / item_area

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = map(float, box)

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @classmethod
    def _matches_region(
        cls,
        item_box,
        region_box,
        minimum_overlap: float = 0.10,
        center_margin_x: float = 0.10,
        center_margin_y: float = 0.10,
    ) -> bool:
        """
        Match an item to a worker region using:
        - PPE-box overlap with the region.
        - A tolerant centre-point fallback.

        This avoids rejecting valid PPE boxes simply because their
        centres fall slightly outside a strict geometric boundary.
        """

        if not cls._valid_box(item_box):
            return False

        if not cls._valid_box(region_box):
            return False

        overlap = cls._overlap_ratio(item_box, region_box)

        if overlap >= minimum_overlap:
            return True

        cx, cy = cls._center(item_box)

        rx1, ry1, rx2, ry2 = map(float, region_box)

        rw = rx2 - rx1
        rh = ry2 - ry1

        return (
            rx1 - rw * center_margin_x
            <= cx
            <= rx2 + rw * center_margin_x
            and
            ry1 - rh * center_margin_y
            <= cy
            <= ry2 + rh * center_margin_y
        )

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "workers": [],
            "helmets": [],
            "gloves": [],
            "vests": [],
            "boots": [],
            "goggles": [],
            "all_ppe": [],
            "safe_workers_count": 0,
            "total_workers_count": 0,
            "violation_count": 0,
            "compliance_rate": 100.0,
            "latency_ms": 0.0,
            "safety_index": "SAFE",
        }

    # ---------------------------------------------------------
    # MODEL INFERENCE
    # ---------------------------------------------------------

    def _run_model(self, model, frame, confidence):
        result = model(
            frame,
            conf=confidence,
            verbose=False,
        )[0]

        if len(result.boxes) == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=float),
                [],
            )

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        names = result.names

        labels = [
            self.standardize_label(
                names.get(int(class_id), str(class_id))
            )
            for class_id in class_ids
        ]

        return boxes, confidences, labels

    # ---------------------------------------------------------
    # ENSEMBLE INFERENCE
    # ---------------------------------------------------------

    def _run_ensemble(
        self,
        all_models: Dict[str, YOLO],
        frame: np.ndarray,
        conf_thresh: float,
    ):
        """Combine model detections using class-aware NMS."""

        all_boxes = []
        all_confs = []
        all_classes = []

        for model_name, model_instance in all_models.items():

            try:
                boxes, confidences, labels = self._run_model(
                    model_instance,
                    frame,
                    conf_thresh,
                )

            except Exception as exc:
                print(
                    f"[Ensemble] Model {model_name} failed: {exc}"
                )
                continue

            for box, confidence, label in zip(
                boxes,
                confidences,
                labels,
            ):

                if label not in (
                    "person",
                    "helmet",
                    "gloves",
                    "vest",
                    "boots",
                    "goggles",
                ):
                    continue

                all_boxes.append(box)
                all_confs.append(float(confidence))
                all_classes.append(label)

        if not all_boxes:
            return (
                np.empty((0, 4)),
                np.empty((0,)),
                [],
            )

        boxes_array = np.asarray(all_boxes, dtype=float)
        confs_array = np.asarray(all_confs, dtype=float)

        selected_indices = []

        # NMS must be performed separately for each class.
        # A helmet must never suppress a person or a vest.
        for category in sorted(set(all_classes)):

            category_indices = [
                i
                for i, label in enumerate(all_classes)
                if label == category
            ]

            category_boxes = boxes_array[category_indices]
            category_scores = confs_array[category_indices]

            xywh_boxes = [
                [
                    int(box[0]),
                    int(box[1]),
                    max(1, int(box[2] - box[0])),
                    max(1, int(box[3] - box[1])),
                ]
                for box in category_boxes
            ]

            kept = cv2.dnn.NMSBoxes(
                bboxes=xywh_boxes,
                scores=category_scores.tolist(),
                score_threshold=conf_thresh,
                nms_threshold=0.55,
            )

            if kept is None or len(kept) == 0:
                continue

            kept = np.asarray(kept).reshape(-1)

            selected_indices.extend(
                category_indices[int(index)]
                for index in kept
            )

        selected_indices = sorted(
            set(selected_indices),
            key=lambda index: all_confs[index],
            reverse=True,
        )

        return (
            boxes_array[selected_indices],
            confs_array[selected_indices],
            [all_classes[i] for i in selected_indices],
        )

    # ---------------------------------------------------------
    # PERSON DETECTION FALLBACK
    # ---------------------------------------------------------

    def _add_person_fallback(
        self,
        frame,
        raw_boxes,
        raw_confs,
        raw_classes,
        all_models,
        active_model,
        confidence,
    ):
        if not all_models:
            return raw_boxes, raw_confs, raw_classes

        if any(label == "person" for label in raw_classes):
            return raw_boxes, raw_confs, raw_classes

        fallback_order = [
            "yolo11n",
            "yolo12n",
            "yolo26n",
            "ppe_kit_gloves_yolo11n",
        ]

        for model_key in fallback_order:

            fallback_model = all_models.get(model_key)

            if fallback_model is None:
                continue

            if fallback_model is active_model:
                continue

            try:
                boxes, confs, labels = self._run_model(
                    fallback_model,
                    frame,
                    confidence,
                )

            except Exception as exc:
                print(
                    f"[Person fallback] {model_key} failed: {exc}"
                )
                continue

            person_indices = [
                i
                for i, label in enumerate(labels)
                if label == "person"
            ]

            if not person_indices:
                continue

            raw_boxes = np.vstack(
                [raw_boxes, boxes[person_indices]]
            ) if len(raw_boxes) else boxes[person_indices]

            raw_confs = np.concatenate(
                [raw_confs, confs[person_indices]]
            )

            raw_classes.extend(
                ["person"] * len(person_indices)
            )

            break

        return raw_boxes, raw_confs, raw_classes

    # ---------------------------------------------------------
    # PPE ENRICHMENT
    # ---------------------------------------------------------

    def _add_ppe_enrichment(
        self,
        frame,
        raw_boxes,
        raw_confs,
        raw_classes,
        all_models,
        active_model,
        confidence,
    ):
        if not all_models:
            return raw_boxes, raw_confs, raw_classes

        ppe_model = all_models.get(
            "ppe_kit_gloves_yolo11n"
        )

        if ppe_model is None or ppe_model is active_model:
            return raw_boxes, raw_confs, raw_classes

        try:
            boxes, confs, labels = self._run_model(
                ppe_model,
                frame,
                confidence,
            )

        except Exception as exc:
            print(f"[PPE enrichment] Failed: {exc}")
            return raw_boxes, raw_confs, raw_classes

        wanted = {
            "helmet",
            "gloves",
            "vest",
            "boots",
            "goggles",
        }

        indices = [
            i
            for i, label in enumerate(labels)
            if label in wanted
        ]

        if not indices:
            return raw_boxes, raw_confs, raw_classes

        extra_boxes = boxes[indices]
        extra_confs = confs[indices]
        extra_labels = [labels[i] for i in indices]

        raw_boxes = (
            np.vstack([raw_boxes, extra_boxes])
            if len(raw_boxes)
            else extra_boxes
        )

        raw_confs = np.concatenate(
            [raw_confs, extra_confs]
        )

        raw_classes.extend(extra_labels)

        return raw_boxes, raw_confs, raw_classes

    # ---------------------------------------------------------
    # DUPLICATE PERSON FILTERING
    # ---------------------------------------------------------

    def _filter_person_boxes(
        self,
        boxes,
        confidences,
        classes,
    ):
        """
        Remove heavily overlapping duplicate person detections.

        Does not remove a smaller person merely because they are
        standing near or partially in front of another person.
        """

        person_indices = [
            i
            for i, label in enumerate(classes)
            if label == "person"
        ]

        if len(person_indices) <= 1:
            return person_indices

        ordered = sorted(
            person_indices,
            key=lambda i: float(confidences[i]),
            reverse=True,
        )

        accepted = []

        for index in ordered:

            candidate = boxes[index]
            duplicate = False

            for accepted_index in accepted:

                existing = boxes[accepted_index]

                intersection = self._intersection(
                    candidate,
                    existing,
                )

                candidate_area = self._area(candidate)
                existing_area = self._area(existing)

                if candidate_area <= 0 or existing_area <= 0:
                    continue

                smaller_area = min(
                    candidate_area,
                    existing_area,
                )

                overlap = intersection / smaller_area

                # Suppress only near-duplicate person boxes.
                if overlap >= 0.78:
                    duplicate = True
                    break

            if not duplicate:
                accepted.append(index)

        return accepted

    # ---------------------------------------------------------
    # PPE ITEM CREATION
    # ---------------------------------------------------------

    def _build_ppe_items(
        self,
        boxes,
        confidences,
        classes,
    ):
        category_map = {
            "helmet": [],
            "gloves": [],
            "vest": [],
            "boots": [],
            "goggles": [],
        }

        all_ppe = []

        for index, label in enumerate(classes):

            if label not in category_map:
                continue

            box = boxes[index]

            if not self._valid_box(box):
                continue

            item = {
                "label": label,
                "box": [
                    float(value)
                    for value in box
                ],
                "confidence": round(
                    float(confidences[index]),
                    3,
                ),
                "matched_worker_id": None,
            }

            category_map[label].append(item)
            all_ppe.append(item)

        return category_map, all_ppe

    # ---------------------------------------------------------
    # PPE ASSOCIATION
    # ---------------------------------------------------------

    def _associate_ppe(
        self,
        worker_box,
        ppe_items,
        category,
    ):
        """
        Find the best PPE association for one worker.

        Uses category-specific body regions and overlap-based
        matching instead of relying only on box centres.
        """

        x1, y1, x2, y2 = map(float, worker_box)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False, 0.0, None, None

        regions = {
            "helmet": [
                x1 - width * 0.15,
                y1 - height * 0.08,
                x2 + width * 0.15,
                y1 + height * 0.45,
            ],

            "gloves": [
                x1 - width * 0.40,
                y1 + height * 0.12,
                x2 + width * 0.40,
                y2 + height * 0.18,
            ],

            "vest": [
                x1 - width * 0.18,
                y1 + height * 0.10,
                x2 + width * 0.18,
                y1 + height * 0.85,
            ],

            "boots": [
                x1 - width * 0.20,
                y1 + height * 0.50,
                x2 + width * 0.20,
                y2 + height * 0.25,
            ],

            "goggles": [
                x1 - width * 0.12,
                y1,
                x2 + width * 0.12,
                y1 + height * 0.42,
            ],
        }

        region = regions[category]

        best_item = None
        best_score = 0.0

        for item in ppe_items:

            item_box = item.get("box")

            if not self._valid_box(item_box):
                continue

            if not self._matches_region(
                item_box,
                region,
                minimum_overlap=0.08,
                center_margin_x=0.08,
                center_margin_y=0.08,
            ):
                continue

            overlap = self._overlap_ratio(
                item_box,
                region,
            )

            confidence = float(
                item.get("confidence", 0.0)
            )

            # Prefer detections that overlap the region and have
            # stronger model confidence.
            score = (
                overlap * 0.65
                + confidence * 0.35
            )

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return False, 0.0, None, None

        return (
            True,
            float(best_item.get("confidence", 0.0)),
            best_item.get("box"),
            best_item,
        )

    # ---------------------------------------------------------
    # MAIN DETECTION PIPELINE
    # ---------------------------------------------------------

    def detect_and_validate(
        self,
        model: YOLO,
        frame: np.ndarray,
        conf_threshold: float = 0.20,
        is_ensemble: bool = False,
        all_models: Optional[Dict[str, YOLO]] = None,
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
        ):
            return self._empty_result()

        if not model and not (
            is_ensemble and all_models
        ):
            return self._empty_result()

        # Preserve the configured confidence threshold.
        # Do not silently lower it or force every model to use
        # a different threshold.
        effective_conf = max(
            0.01,
            min(float(conf_threshold), 0.95),
        )

        # 1. Primary inference
        if is_ensemble and all_models:

            raw_boxes, raw_confs, raw_classes = (
                self._run_ensemble(
                    all_models,
                    frame,
                    effective_conf,
                )
            )

        else:

            raw_boxes, raw_confs, raw_classes = (
                self._run_model(
                    model,
                    frame,
                    effective_conf,
                )
            )

        # 2. Person fallback
        (
            raw_boxes,
            raw_confs,
            raw_classes,
        ) = self._add_person_fallback(
            frame,
            raw_boxes,
            raw_confs,
            raw_classes,
            all_models,
            model,
            effective_conf,
        )

        # 3. PPE enrichment
        (
            raw_boxes,
            raw_confs,
            raw_classes,
        ) = self._add_ppe_enrichment(
            frame,
            raw_boxes,
            raw_confs,
            raw_classes,
            all_models,
            model,
            effective_conf,
        )

        # 4. Filter invalid detections
        valid_indices = []

        for index, box in enumerate(raw_boxes):

            if not self._valid_box(box):
                continue

            confidence = float(raw_confs[index])

            if not np.isfinite(confidence):
                continue

            valid_indices.append(index)

        raw_boxes = (
            raw_boxes[valid_indices]
            if valid_indices
            else np.empty((0, 4))
        )

        raw_confs = (
            raw_confs[valid_indices]
            if valid_indices
            else np.empty((0,))
        )

        raw_classes = [
            raw_classes[index]
            for index in valid_indices
        ]

        # 5. Build person list and PPE categories
        person_indices = self._filter_person_boxes(
            raw_boxes,
            raw_confs,
            raw_classes,
        )

        ppe_categories, all_ppe = self._build_ppe_items(
            raw_boxes,
            raw_confs,
            raw_classes,
        )

        workers_info = []

        # 6. Associate PPE with individual workers
        for worker_index, detection_index in enumerate(
            person_indices,
            start=1,
        ):

            worker_box = [
                float(value)
                for value in raw_boxes[detection_index]
            ]

            x1, y1, x2, y2 = worker_box

            height = y2 - y1

            head_zone = [
                x1,
                y1,
                x2,
                y1 + height * self.head_ratio,
            ]

            worker = {
                "id": worker_index,
                "box": worker_box,
                "confidence": round(
                    float(raw_confs[detection_index]),
                    3,
                ),
                "head_zone": head_zone,
            }

            detected_ppe = []

            for category in self.ppe_categories:

                (
                    detected,
                    confidence,
                    matched_box,
                    matched_item,
                ) = self._associate_ppe(
                    worker_box,
                    ppe_categories[category],
                    category,
                )

                worker[f"has_{category}"] = detected

                worker[f"{category}_confidence"] = (
                    round(confidence, 3)
                    if detected
                    else 0.0
                )

                if category == "helmet":
                    worker["matched_helmet_box"] = (
                        matched_box
                    )

                if detected:

                    detected_ppe.append(category)

                    if matched_item is not None:

                        # A detection can be associated with more
                        # than one worker when workers overlap.
                        # Keep the strongest association.
                        previous_id = matched_item.get(
                            "matched_worker_id"
                        )

                        if previous_id is None:
                            matched_item[
                                "matched_worker_id"
                            ] = worker_index

                # Preserve the monitor's expected explicit
                # visibility fields. The monitor can override
                # these using its own visibility rules.
                worker[f"{category}_visible"] = (
                    self._default_visibility(
                        worker_box,
                        category,
                        frame.shape[1],
                        frame.shape[0],
                    )
                )

            # Detector reports observations only.
            # Site-specific compliance is decided in monitor.py.
            worker["detected_ppe"] = detected_ppe

            worker["missing_ppe"] = []

            worker["is_safe"] = False
            worker["status"] = "UNASSESSED"

            workers_info.append(worker)

        total_workers = len(workers_info)

        latency = (
            time.perf_counter() - started
        ) * 1000.0

        # Compliance is intentionally not calculated here.
        # monitor.py applies zone-specific required PPE and
        # visibility-aware missing/unverified evaluation.
        return {
            "workers": workers_info,

            "helmets": ppe_categories["helmet"],
            "gloves": ppe_categories["gloves"],
            "vests": ppe_categories["vest"],
            "boots": ppe_categories["boots"],
            "goggles": ppe_categories["goggles"],

            "all_ppe": all_ppe,

            "safe_workers_count": 0,
            "total_workers_count": total_workers,
            "violation_count": 0,
            "compliance_rate": 100.0,

            "latency_ms": round(latency, 1),
            "safety_index": "UNASSESSED",
        }

    # ---------------------------------------------------------
    # VISIBILITY ESTIMATION
    # ---------------------------------------------------------

    @staticmethod
    def _default_visibility(
        worker_box,
        category,
        frame_width,
        frame_height,
    ):
        """
        Estimate whether a body region is visible.

        This is only a geometric estimate. monitor.py remains
        responsible for the final visibility decision.
        """

        x1, y1, x2, y2 = map(float, worker_box)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        if category == "helmet":

            return (
                y1 >= 0
                and y1 + height * 0.38 <= frame_height
            )

        if category == "vest":

            torso_top = y1 + height * 0.18
            torso_bottom = y1 + height * 0.72

            return (
                torso_top >= 0
                and torso_bottom <= frame_height
                and width >= 25
                and height >= 50
            )

        if category == "boots":

            return (
                y1 + height * 0.72 >= 0
                and y2 <= frame_height - 2
                and height >= 50
            )

        # Gloves and goggles need more reliable local visibility
        # information than a full person bounding box provides.
        return False


hybrid_detector = HybridPPEDetector()
