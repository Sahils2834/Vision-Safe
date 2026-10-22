"""Hybrid PPE Detection Engine.

Combines YOLO Object Detection with:
1. Spatial Head-Zone Overlap Analysis (Upper 38% of Person Bounding Box)
2. Multi-PPE Spatial Association: Helmets, Construction Gloves, Safety Vests, Boots, Goggles
3. Dual-Model Cross-Validation (Ensures robust person detection in close-up webcam & site views)
4. Confidence Calibration & Multi-Model Ensemble Consensus
5. Geometric Worker-to-PPE Association & Compliance Scoring
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
            "vest": "vest", "safety_vest": "vest", "ppe_kit": "vest",
            "glove": "gloves", "gloves": "gloves", "construction_gloves": "gloves",
            "boot": "boots", "boots": "boots", "shoe": "boots", "shoes": "boots", "safety_boots": "boots",
            "goggle": "goggles", "goggles": "goggles", "glasses": "goggles", "safety_glasses": "goggles"
        }

    def standardize_label(self, label: str) -> str:
        l = str(label).lower().strip().replace("-", "_").replace(" ", "_")
        return self.ppe_labels.get(l, l)

    def detect_and_validate(
        self, 
        model: YOLO, 
        frame: np.ndarray, 
        conf_threshold: float = 0.20,
        is_ensemble: bool = False,
        all_models: Optional[Dict[str, YOLO]] = None
    ) -> Dict[str, Any]:
        """Runs detection and validates PPE compliance with spatial overlap analysis."""
        t_start = time.perf_counter()
        
        # Adaptive confidence: allow sensitive detection for PPE items and people
        effective_conf = min(conf_threshold, 0.20)

        if not model and not is_ensemble:
            return {
                "detections": None,
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
                "safety_index": "SAFE"
            }

        # Run primary detection
        if is_ensemble and all_models and len(all_models) > 1:
            raw_boxes, raw_confs, raw_classes = self._run_ensemble(all_models, frame, effective_conf)
        else:
            results = model(frame, conf=effective_conf, verbose=False)[0]
            names = results.names
            raw_boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.empty((0, 4))
            raw_confs = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.empty((0,))
            raw_classes = [self.standardize_label(names.get(int(c), str(c))) for c in (results.boxes.cls.cpu().numpy() if len(results.boxes) > 0 else [])]

        # Multi-model Cross-Enhancement:
        # 1. If person is not detected (e.g. close-up webcam desk shot where fine-tuned model has lower person confidence),
        # query other loaded models (like yolo11n, yolo12n) to ensure people are never missed.
        person_count = sum(1 for c in raw_classes if c == "person")
        if person_count == 0 and all_models:
            for fallback_key in ["yolo11n", "yolo12n", "yolo26n", "ppe_kit_gloves_yolo11n"]:
                fb_model = all_models.get(fallback_key)
                if fb_model is not None and fb_model is not model:
                    fb_res = fb_model(frame, conf=effective_conf, verbose=False)[0]
                    fb_names = fb_res.names
                    if len(fb_res.boxes) > 0:
                        for b, c, s in zip(fb_res.boxes.xyxy.cpu().numpy(), fb_res.boxes.cls.cpu().numpy(), fb_res.boxes.conf.cpu().numpy()):
                            lbl = self.standardize_label(fb_names.get(int(c), str(c)))
                            if lbl == "person":
                                raw_boxes = np.vstack([raw_boxes, b]) if len(raw_boxes) > 0 else np.array([b])
                                raw_confs = np.append(raw_confs, s)
                                raw_classes.append("person")
                        if any(l == "person" for l in raw_classes):
                            break

        # 2. Always enrich detections from ppe_kit_gloves_yolo11n (full 6-class PPE model)
        # when available. This ensures gloves, vests, boots, goggles, helmets are ALWAYS detected
        # regardless of which primary model is active.
        if all_models and "ppe_kit_gloves_yolo11n" in all_models:
            ppe_model = all_models["ppe_kit_gloves_yolo11n"]
            # Only re-run if the active model ISN'T already the full PPE model
            if model is not ppe_model:
                ppe_res = ppe_model(frame, conf=effective_conf, verbose=False)[0]
                ppe_names = ppe_res.names
                if len(ppe_res.boxes) > 0:
                    for b, c, s in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
                        lbl = self.standardize_label(ppe_names.get(int(c), str(c)))
                        if lbl in ("gloves", "vest", "boots", "goggles", "helmet"):
                            raw_boxes = np.vstack([raw_boxes, b]) if len(raw_boxes) > 0 else np.array([b])
                            raw_confs = np.append(raw_confs, s)
                            raw_classes.append(lbl)
            else:
                # ppe_kit_gloves_yolo11n IS the primary model - detections are already complete,
                # no need to re-run. But also pull from other loaded models for person detection.
                pass

        person_indices = [i for i, l in enumerate(raw_classes) if l == "person"]
        helmet_indices = [i for i, l in enumerate(raw_classes) if l == "helmet"]
        gloves_indices = [i for i, l in enumerate(raw_classes) if l == "gloves"]
        vest_indices = [i for i, l in enumerate(raw_classes) if l == "vest"]
        boots_indices = [i for i, l in enumerate(raw_classes) if l == "boots"]
        goggles_indices = [i for i, l in enumerate(raw_classes) if l == "goggles"]

        # Filter phantom person boxes (e.g. raised hand/arm falsely detected as a second worker)
        if len(person_indices) > 1:
            filtered_person_indices = []
            candidates = sorted(person_indices, key=lambda i: (raw_boxes[i][2] - raw_boxes[i][0]) * (raw_boxes[i][3] - raw_boxes[i][1]), reverse=True)
            for idx in candidates:
                b = raw_boxes[idx]
                w = b[2] - b[0]
                h = b[3] - b[1]
                area = w * h
                
                is_sub_part = False
                for p_accepted in filtered_person_indices:
                    ab = raw_boxes[p_accepted]
                    aw = ab[2] - ab[0]
                    ah = ab[3] - ab[1]
                    a_area = aw * ah
                    
                    # Compute intersection
                    ix1, iy1 = max(b[0], ab[0]), max(b[1], ab[1])
                    ix2, iy2 = min(b[2], ab[2]), min(b[3], ab[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / area > 0.30:
                            is_sub_part = True
                            break
                    # If this box is much smaller than the primary worker and within reach
                    if area < a_area * 0.40 and (ab[0] - aw * 0.3) <= (b[0] + b[2]) / 2 <= (ab[2] + aw * 0.3):
                        is_sub_part = True
                        break
                        
                if not is_sub_part:
                    filtered_person_indices.append(idx)
            person_indices = filtered_person_indices

        helmets_info = []
        gloves_info = []
        vests_info = []
        boots_info = []
        goggles_info = []
        all_ppe_info = []

        def _make_item(idx: int, label: str) -> Dict[str, Any]:
            bx1, by1, bx2, by2 = raw_boxes[idx]
            item = {
                "label": label,
                "box": [float(bx1), float(by1), float(bx2), float(by2)],
                "confidence": round(float(raw_confs[idx]), 3),
                "matched_worker_id": None
            }
            all_ppe_info.append(item)
            return item

        for idx in helmet_indices:
            helmets_info.append(_make_item(idx, "helmet"))
        for idx in gloves_indices:
            gloves_info.append(_make_item(idx, "gloves"))
        for idx in vest_indices:
            vests_info.append(_make_item(idx, "vest"))
        for idx in boots_indices:
            boots_info.append(_make_item(idx, "boots"))
        for idx in goggles_indices:
            goggles_info.append(_make_item(idx, "goggles"))

        workers_info = []
        safe_count = 0

        for p_idx in person_indices:
            px1, py1, px2, py2 = raw_boxes[p_idx]
            p_conf = float(raw_confs[p_idx])
            p_height = py2 - py1
            p_width = px2 - px1
            worker_id = len(workers_info) + 1
            
            # Spatial head zone: top 38% of bounding box
            head_zone = [float(px1), float(py1), float(px2), float(py1 + p_height * self.head_ratio)]
            
            # 1. Helmet Match (Head zone)
            has_helmet = False
            best_helmet_conf = 0.0
            matched_helmet_box = None
            for h_item in helmets_info:
                hx1, hy1, hx2, hy2 = h_item["box"]
                h_cx = (hx1 + hx2) / 2.0
                h_cy = (hy1 + hy2) / 2.0
                if (px1 - p_width * 0.15) <= h_cx <= (px2 + p_width * 0.15):
                    if (py1 - p_height * 0.10) <= h_cy <= (py1 + p_height * 0.45):
                        has_helmet = True
                        if h_item["confidence"] > best_helmet_conf:
                            best_helmet_conf = h_item["confidence"]
                            matched_helmet_box = h_item["box"]
                            h_item["matched_worker_id"] = worker_id

            # 2. Gloves Match (Hands/Arms/Worker bounding bounds)
            has_gloves = False
            best_gloves_conf = 0.0
            for g_item in gloves_info:
                gx1, gy1, gx2, gy2 = g_item["box"]
                g_cx = (gx1 + gx2) / 2.0
                g_cy = (gy1 + gy2) / 2.0
                if (px1 - p_width * 0.35) <= g_cx <= (px2 + p_width * 0.35):
                    if (py1 + p_height * 0.15) <= g_cy <= (py2 + p_height * 0.15):
                        has_gloves = True
                        if g_item["confidence"] > best_gloves_conf:
                            best_gloves_conf = g_item["confidence"]
                            g_item["matched_worker_id"] = worker_id

            # 3. Vest Match (Torso zone: 15% to 80% height)
            has_vest = False
            best_vest_conf = 0.0
            for v_item in vests_info:
                vx1, vy1, vx2, vy2 = v_item["box"]
                v_cx = (vx1 + vx2) / 2.0
                v_cy = (vy1 + vy2) / 2.0
                if (px1 - p_width * 0.20) <= v_cx <= (px2 + p_width * 0.20):
                    if (py1 + p_height * 0.10) <= v_cy <= (py1 + p_height * 0.85):
                        has_vest = True
                        if v_item["confidence"] > best_vest_conf:
                            best_vest_conf = v_item["confidence"]
                            v_item["matched_worker_id"] = worker_id

            # 4. Boots Match (Lower body zone: > 50% height)
            has_boots = False
            best_boots_conf = 0.0
            for b_item in boots_info:
                bx1, by1, bx2, by2 = b_item["box"]
                b_cx = (bx1 + bx2) / 2.0
                b_cy = (by1 + by2) / 2.0
                if (px1 - p_width * 0.20) <= b_cx <= (px2 + p_width * 0.20):
                    if (py1 + p_height * 0.50) <= b_cy <= (py2 + p_height * 0.20):
                        has_boots = True
                        if b_item["confidence"] > best_boots_conf:
                            best_boots_conf = b_item["confidence"]
                            b_item["matched_worker_id"] = worker_id

            # 5. Goggles Match (Head zone: upper 40% height)
            has_goggles = False
            best_goggles_conf = 0.0
            for gg_item in goggles_info:
                gx1, gy1, gx2, gy2 = gg_item["box"]
                g_cx = (gx1 + gx2) / 2.0
                g_cy = (gy1 + gy2) / 2.0
                if (px1 - p_width * 0.15) <= g_cx <= (px2 + p_width * 0.15):
                    if py1 <= g_cy <= (py1 + p_height * 0.40):
                        has_goggles = True
                        if gg_item["confidence"] > best_goggles_conf:
                            best_goggles_conf = gg_item["confidence"]
                            gg_item["matched_worker_id"] = worker_id

            # Build list of detected PPE for this worker
            detected_ppe = []
            if has_helmet: detected_ppe.append("helmet")
            if has_gloves: detected_ppe.append("gloves")
            if has_vest: detected_ppe.append("vest")
            if has_boots: detected_ppe.append("boots")
            if has_goggles: detected_ppe.append("goggles")

            # Default compliance evaluation (requires all 5 PPE items: helmet, gloves, vest, boots, goggles)
            missing = []
            if not has_helmet:  missing.append("helmet")
            if not has_gloves:  missing.append("gloves")
            if not has_vest:    missing.append("vest")
            if not has_boots:   missing.append("boots")
            if not has_goggles: missing.append("goggles")

            is_safe = (len(missing) == 0)
            if is_safe:
                safe_count += 1

            workers_info.append({
                "id": worker_id,
                "box": [float(px1), float(py1), float(px2), float(py2)],
                "confidence": round(p_conf, 3),
                "head_zone": head_zone,
                "is_safe": is_safe,
                "has_helmet": has_helmet,
                "helmet_confidence": round(best_helmet_conf, 3) if has_helmet else 0.0,
                "matched_helmet_box": matched_helmet_box,
                "has_gloves": has_gloves,
                "gloves_confidence": round(best_gloves_conf, 3) if has_gloves else 0.0,
                "has_vest": has_vest,
                "vest_confidence": round(best_vest_conf, 3) if has_vest else 0.0,
                "has_boots": has_boots,
                "boots_confidence": round(best_boots_conf, 3) if has_boots else 0.0,
                "has_goggles": has_goggles,
                "goggles_confidence": round(best_goggles_conf, 3) if has_goggles else 0.0,
                "detected_ppe": detected_ppe,
                "missing_ppe": missing,
                "status": "COMPLIANT" if is_safe else "VIOLATION"
            })

        total_workers = len(workers_info)
        compliance_rate = round((safe_count / total_workers * 100.0), 1) if total_workers > 0 else 100.0
        latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "workers": workers_info,
            "helmets": helmets_info,
            "gloves": gloves_info,
            "vests": vests_info,
            "boots": boots_info,
            "goggles": goggles_info,
            "all_ppe": all_ppe_info,
            "safe_workers_count": safe_count,
            "total_workers_count": total_workers,
            "violation_count": total_workers - safe_count,
            "compliance_rate": compliance_rate,
            "latency_ms": round(latency, 1),
            "safety_index": "SAFE" if (total_workers == 0 or compliance_rate == 100) else ("WARNING" if compliance_rate >= 50 else "CRITICAL")
        }

    def _run_ensemble(self, all_models: Dict[str, YOLO], frame: np.ndarray, conf_thresh: float):
        """Merges predictions across multiple models via NMS fusion."""
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

        boxes_arr = np.array(all_boxes)
        confs_arr = np.array(all_confs)
        
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
