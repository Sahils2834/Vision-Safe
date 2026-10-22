"""Frame processing: detection, ByteTrack IDs, PPE evaluation, spatial zones, and alerts."""
from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime, timezone
import cv2
import numpy as np
import supervision as sv

from .config import Settings
from .schemas import Alert, Zone
from .model_registry import model_registry
from .hybrid_detector import hybrid_detector


class PPEMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=180,
            minimum_matching_threshold=0.7,
            frame_rate=30,
            minimum_consecutive_frames=1,
        )
        self.zones: list[Zone] = []
        self.alerts: deque[Alert] = deque(maxlen=200)
        self._last_alert: dict[tuple[int, str], float] = {}
        self._track_workers: dict[int, int] = {}
        self._workers: dict[int, dict] = {}
        self._next_worker_id = 1
        
        # Real-time telemetry
        self.last_compliance_rate = 100.0
        self.last_worker_count = 0
        self.last_violation_count = 0
        self.fps = 0.0
        self._last_frame_time = time.perf_counter()

    @property
    def ready(self) -> bool:
        return len(model_registry.loaded_models) > 0 or model_registry.get_model() is not None

    @property
    def model_name(self) -> str | None:
        return f"YOLO-{model_registry.active_model_key.upper()}"

    @property
    def supports_ppe(self) -> bool:
        return True

    def set_zones(self, zones: list[Zone]) -> None:
        self.zones = zones

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[Alert]]:
        # Calculate FPS
        now_time = time.perf_counter()
        delta = now_time - self._last_frame_time
        if delta > 0:
            self.fps = round(0.9 * self.fps + 0.1 * (1.0 / delta), 1)
        self._last_frame_time = now_time

        active_key = model_registry.active_model_key
        model = model_registry.get_model()
        is_ensemble = (active_key == "ensemble")
        
        if not model and not is_ensemble:
            return self._draw_standby_hud(frame), []

        # Run Hybrid Detector
        diag = hybrid_detector.detect_and_validate(
            model=model,
            frame=frame,
            conf_threshold=self.settings.confidence_threshold,
            is_ensemble=is_ensemble,
            all_models=model_registry.loaded_models
        )

        workers = diag["workers"]
        helmets = diag["helmets"]
        gloves = diag.get("gloves", [])
        vests = diag.get("vests", [])
        boots = diag.get("boots", [])
        goggles = diag.get("goggles", [])

        self.last_worker_count = len(workers)
        self.last_violation_count = diag["violation_count"]
        self.last_compliance_rate = diag["compliance_rate"]

        new_alerts: list[Alert] = []
        h, w = frame.shape[:2]

        # Draw Zones first (behind detections)
        self._draw_zones(frame)

        # Helper to draw detection boxes
        def _draw_item_box(items, label_name: str, color: tuple[int, int, int]):
            for itm in items:
                ix1, iy1, ix2, iy2 = map(int, itm["box"])
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                cv2.putText(
                    frame, 
                    f"{label_name} {itm['confidence']:.2f}", 
                    (ix1, max(15, iy1 - 4)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color, 
                    1, 
                    cv2.LINE_AA
                )

        # Draw Helmets (Cyan/Gold: 255, 230, 0)
        _draw_item_box(helmets, "Helmet", (255, 230, 0))

        # Draw Gloves (Neon Amber/Orange: 0, 140, 255)
        _draw_item_box(gloves, "Gloves", (0, 140, 255))

        # Draw Vests (High-Vis Neon Lime: 0, 240, 255)
        _draw_item_box(vests, "Vest", (0, 240, 255))

        # Draw Boots (Sky Blue: 255, 180, 0)
        _draw_item_box(boots, "Boots", (255, 180, 0))

        # Draw Goggles (Neon Violet: 255, 50, 200)
        _draw_item_box(goggles, "Goggles", (255, 50, 200))

        # Process each detected worker
        safe_workers = 0
        for worker in workers:
            x1, y1, x2, y2 = map(int, worker["box"])
            worker_id = worker["id"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            zone = self._zone_for(center)

            # Evaluate required PPE: zone requirements or default full PPE kit (helmet + gloves + vest + boots + goggles)
            requirements = zone.required_ppe if (zone and zone.required_ppe) else ["helmet", "gloves", "vest", "boots", "goggles"]
            missing = []
            for req in requirements:
                r_lower = req.lower().strip()
                if r_lower in ("helmet", "hat", "hardhat", "safety_helmet") and not worker["has_helmet"]:
                    missing.append("helmet")
                elif r_lower in ("gloves", "glove", "construction_gloves") and not worker.get("has_gloves"):
                    missing.append("gloves")
                elif r_lower in ("vest", "safety_vest", "ppe_kit", "ppe kit") and not worker.get("has_vest"):
                    missing.append("vest")
                elif r_lower in ("boots", "boot", "safety_boots", "shoes", "shoe", "safety_shoes") and not worker.get("has_boots"):
                    missing.append("boots")
                elif r_lower in ("goggles", "goggle", "glasses", "safety_glasses") and not worker.get("has_goggles"):
                    missing.append("goggles")

            worker["missing_ppe"] = missing
            is_safe = (len(missing) == 0)
            worker["is_safe"] = is_safe
            worker["status"] = "COMPLIANT" if is_safe else "VIOLATION"

            if is_safe:
                safe_workers += 1

            # Compliant = Emerald Green (0, 230, 118), Violation = Crimson Red (30, 30, 255)
            color = (0, 230, 118) if is_safe else (30, 30, 255)

            # Draw worker bounding box with thick corners
            self._draw_styled_box(frame, (x1, y1, x2, y2), color)

            # Draw spatial head zone
            hx1, hy1, hx2, hy2 = map(int, worker["head_zone"])
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 200, 200), 1, cv2.LINE_AA)

            # Format status badges
            badge_str = "SAFE" if is_safe else f"NO {', '.join(missing).upper()}"
            status_text = f"Worker #{worker_id} - {'COMPLIANT' if is_safe else 'VIOLATION'}"

            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 14)), (x1 + tw + 16, y1), color, -1)
            cv2.putText(
                frame, 
                status_text, 
                (x1 + 8, max(th + 2, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.52, 
                (0, 0, 0) if is_safe else (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

            # Draw individual PPE status chips below worker box
            chip_y = min(frame.shape[0] - 10, y2 + 20)
            chip_x = x1
            ppe_status_list = [
                ("HELMET",  worker["has_helmet"],        (0, 220, 100), (30, 30, 255)),
                ("GLOVES",  worker.get("has_gloves"),    (0, 220, 100), (0, 100, 255)),
                ("VEST",    worker.get("has_vest"),      (0, 220, 100), (0, 200, 255)),
                ("BOOTS",   worker.get("has_boots"),     (0, 220, 100), (0, 140, 255)),
                ("GOGGLES", worker.get("has_goggles"),  (0, 220, 100), (200, 50, 255)),
            ]

            for chip_label, is_present, ok_col, miss_col in ppe_status_list:
                chip_txt = f"{chip_label}:OK" if is_present else f"NO {chip_label}"
                chip_bg = ok_col if is_present else miss_col
                (cw, ch), _ = cv2.getTextSize(chip_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                if chip_x + cw + 10 < frame.shape[1]:
                    cv2.rectangle(frame, (chip_x, chip_y - ch - 4), (chip_x + cw + 8, chip_y + 4), chip_bg, -1)
                    cv2.putText(frame, chip_txt, (chip_x + 4, chip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0) if is_present else (255, 255, 255), 1, cv2.LINE_AA)
                    chip_x += cw + 12

            # Record INDIVIDUAL separate alerts for EACH missing PPE item
            # Each PPE type gets its own distinct alert with unique message, severity and icon
            PPE_ALERT_MAP = {
                "helmet":  ("🪖 Safety Helmet",       "critical", f"Worker #{worker_id} — No Safety Helmet detected! Head injury risk."),
                "gloves":  ("🧤 Construction Gloves",  "warning",  f"Worker #{worker_id} — No Construction Gloves detected! Hand hazard."),
                "vest":    ("🦺 Safety Vest / PPE Kit","critical", f"Worker #{worker_id} — No Safety Vest / PPE Kit detected!"),
                "boots":   ("🥾 Safety Boots / Shoes", "warning",  f"Worker #{worker_id} — No Safety Boots detected! Foot injury risk."),
                "goggles": ("🥽 Safety Goggles",       "warning",  f"Worker #{worker_id} — No Safety Goggles detected! Eye hazard."),
            }
            for item in missing:
                if self._should_alert(worker_id, item):
                    disp_name, severity, msg = PPE_ALERT_MAP.get(
                        item, (item.capitalize(), "warning", f"Worker #{worker_id} — Missing {item}")
                    )
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        worker_id=worker_id,
                        zone_id=zone.id if zone else None,
                        missing_ppe=[disp_name],
                        severity=severity,
                        message=msg,
                        created_at=datetime.now(timezone.utc)
                    )
                    self.alerts.appendleft(alert)
                    new_alerts.append(alert)

        # Update telemetry
        total_w = len(workers)
        self.last_worker_count = total_w
        self.last_violation_count = total_w - safe_workers
        self.last_compliance_rate = round((safe_workers / total_w * 100.0), 1) if total_w > 0 else 100.0

        # Draw HUD telemetry banner
        self._draw_hud_banner(frame, active_key, diag["latency_ms"])

        return frame, new_alerts

    def _draw_styled_box(self, frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 4
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def _draw_hud_banner(self, frame: np.ndarray, model_key: str, latency_ms: float):
        h, w = frame.shape[:2]
        hud_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (0, 229, 255), 1)

        model_tag = f"AI ENGINE: YOLO-{model_key.upper()}"
        comp_tag = f"COMPLIANCE: {self.last_compliance_rate:.0f}%"
        perf_tag = f"{self.fps:.0f} FPS | {latency_ms:.0f}ms"
        comp_color = (0, 230, 118) if self.last_compliance_rate == 100 else ((0, 200, 255) if self.last_compliance_rate >= 60 else (30, 30, 255))
        
        cv2.putText(frame, model_tag, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, comp_tag, (w // 2 - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, comp_color, 2, cv2.LINE_AA)
        cv2.putText(frame, perf_tag, (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_standby_hud(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.putText(frame, "VisionSafe AI: Standby - No Model Loaded", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
        return frame

    def _zone_for(self, point: tuple[int, int]) -> Zone | None:
        for zone in self.zones:
            contour = np.array([[p.x, p.y] for p in zone.polygon], dtype=np.int32)
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return zone
        return None

    def _draw_zones(self, frame: np.ndarray) -> None:
        for zone in self.zones:
            contour = np.array([[p.x, p.y] for p in zone.polygon], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour], (255, 140, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [contour], True, (255, 180, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ZONE: {zone.name.upper()}", tuple(contour[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)

    def _should_alert(self, worker_id: int, issue: str) -> bool:
        key = (worker_id, issue)
        now = time.monotonic()
        if now - self._last_alert.get(key, 0) < self.settings.alert_cooldown_seconds:
            return False
        self._last_alert[key] = now
        return True
