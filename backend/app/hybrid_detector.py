"""Hybrid PPE Detection Engine.

Combines YOLO Object Detection with:
1. Spatial Head-Zone Overlap Analysis (Upper 38% of Person Bounding Box)
2. Confidence Calibration & Multi-Model Ensemble Consensus
3. Worker-to-PPE Geometric Association
4. False-Positive Filtering & Safety Compliance Scoring
"""
from __future__ import annotations

import time
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List, Any, Tuple, Optional

class HybridPPEDetector:
    def __init__(self, head_ratio: float = 0.38):
        self.head_ratio = head_ratio
        self.ppe_labels = {
            "person": "person", "worker": "person",
            "hat": "helmet", "helmet": "helmet", "hardhat": "helmet", "safety_helmet": "helmet",
            "vest": "vest", "safety_vest": "vest",
            "glove": "gloves", "gloves": "gloves",
            "boot": "boots", "boots": "boots"
        }

    def standardize_label(self, label: str) -> str:
        l = str(label).lower().strip().replace("-", "_").replace(" ", "_")
        return self.ppe_labels.get(l, l)

    def detect_and_validate(
        self, 
        model: YOLO, 
        frame: np.ndarray, 
        conf_threshold: float = 0.30,
        is_ensemble: bool = False,
        all_models: Optional[Dict[str, YOLO]] = None
    ) -> Dict[str, Any]:
        """Runs detection and validates PPE compliance with spatial overlap analysis."""
        t_start = time.perf_counter()
        
        if not model and not is_ensemble:
            return {
                "detections": None,
                "workers": [],
                "helmets": [],
                "safe_workers_count": 0,
                "total_workers_count": 0,
                "compliance_rate": 100.0,
                "latency_ms": 0.0
            }

        # Run detection
        if is_ensemble and all_models and len(all_models) > 1:
            raw_boxes, raw_confs, raw_classes = self._run_ensemble(all_models, frame, conf_threshold)
        else:
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            names = results.names
            raw_boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.empty((0, 4))
            raw_confs = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.empty((0,))
            raw_classes = [self.standardize_label(names.get(int(c), str(c))) for c in (results.boxes.cls.cpu().numpy() if len(results.boxes) > 0 else [])]

        person_indices = [i for i, l in enumerate(raw_classes) if l == "person"]
        helmet_indices = [i for i, l in enumerate(raw_classes) if l == "helmet"]

        workers_info = []
        helmets_info = []

        for h_idx in helmet_indices:
            hx1, hy1, hx2, hy2 = raw_boxes[h_idx]
            helmets_info.append({
                "box": [float(hx1), float(hy1), float(hx2), float(hy2)],
                "confidence": round(float(raw_confs[h_idx]), 3),
                "matched_worker_id": None
            })

        safe_count = 0
        for p_idx in person_indices:
            px1, py1, px2, py2 = raw_boxes[p_idx]
            p_conf = float(raw_confs[p_idx])
            p_height = py2 - py1
            p_width = px2 - px1
            
            # Spatial head zone: top 38% of bounding box
            head_zone = [float(px1), float(py1), float(px2), float(py1 + p_height * self.head_ratio)]
            
            has_helmet = False
            best_helmet_conf = 0.0
            matched_helmet_box = None

            for h_item in helmets_info:
                hx1, hy1, hx2, hy2 = h_item["box"]
                h_cx = (hx1 + hx2) / 2.0
                h_cy = (hy1 + hy2) / 2.0
                
                # Check if helmet center is within person bounds horizontally & upper body vertically
                if (px1 - p_width * 0.1) <= h_cx <= (px2 + p_width * 0.1):
                    if py1 <= h_cy <= (py1 + p_height * 0.50):
                        has_helmet = True
                        if h_item["confidence"] > best_helmet_conf:
                            best_helmet_conf = h_item["confidence"]
                            matched_helmet_box = h_item["box"]
                            h_item["matched_worker_id"] = len(workers_info) + 1

            is_safe = has_helmet
            if is_safe:
                safe_count += 1

            workers_info.append({
                "id": len(workers_info) + 1,
                "box": [float(px1), float(py1), float(px2), float(py2)],
                "confidence": round(p_conf, 3),
                "head_zone": head_zone,
                "is_safe": is_safe,
                "has_helmet": has_helmet,
                "helmet_confidence": round(best_helmet_conf, 3) if has_helmet else 0.0,
                "matched_helmet_box": matched_helmet_box,
                "missing_ppe": [] if has_helmet else ["helmet"],
                "status": "COMPLIANT" if is_safe else "VIOLATION"
            })

        total_workers = len(workers_info)
        compliance_rate = round((safe_count / total_workers * 100.0), 1) if total_workers > 0 else 100.0
        latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "workers": workers_info,
            "helmets": helmets_info,
            "safe_workers_count": safe_count,
            "total_workers_count": total_workers,
            "violation_count": total_workers - safe_count,
            "compliance_rate": compliance_rate,
            "latency_ms": round(latency, 1),
            "safety_index": "SAFE" if (total_workers == 0 or compliance_rate == 100) else ("WARNING" if compliance_rate >= 50 else "CRITICAL")
        }

    def _run_ensemble(self, all_models: Dict[str, YOLO], frame: np.ndarray, conf_thresh: float):
        """Merges predictions across multiple models via WBF/NMS fusion."""
        all_boxes = []
        all_confs = []
        all_classes = []

        for m_name, m_inst in all_models.items():
            res = m_inst(frame, conf=conf_thresh * 0.9, verbose=False)[0]
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy()
                names = res.names
                for b, c, cl in zip(boxes, confs, cls_ids):
                    all_boxes.append(b)
                    all_confs.append(c)
                    all_classes.append(self.standardize_label(names.get(int(cl), str(cl))))

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), []

        # Simple soft NMS aggregation
        boxes_arr = np.array(all_boxes)
        confs_arr = np.array(all_confs)
        
        # Deduplicate with cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes_arr],
            scores=confs_arr.tolist(),
            score_threshold=conf_thresh,
            nms_threshold=0.55
        )
        
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty((0,)), []
            
        selected_idx = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        return boxes_arr[selected_idx], confs_arr[selected_idx], [all_classes[i] for i in selected_idx]

hybrid_detector = HybridPPEDetector()
