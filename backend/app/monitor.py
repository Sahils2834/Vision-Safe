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
        self.last_worker_count = len(workers)
        self.last_violation_count = diag["violation_count"]
        self.last_compliance_rate = diag["compliance_rate"]

        new_alerts: list[Alert] = []
        h, w = frame.shape[:2]

        # Draw Zones first (behind workers)
        self._draw_zones(frame)

        # Draw Helmets (Cyan outlines)
        for h_item in helmets:
            hx1, hy1, hx2, hy2 = map(int, h_item["box"])
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 230, 0), 2)
            cv2.putText(
                frame, 
                f"Helmet {h_item['confidence']:.2f}", 
                (hx1, max(15, hy1 - 4)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (255, 230, 0), 
                1, 
                cv2.LINE_AA
            )

        # Draw Workers and evaluate safety zones
        for worker in workers:
            x1, y1, x2, y2 = map(int, worker["box"])
            is_safe = worker["is_safe"]
            worker_id = worker["id"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            zone = self._zone_for(center)

            # Missing requirements
            requirements = (zone.required_ppe if zone else ["helmet"])
            missing = [item for item in requirements if (item == "helmet" and not worker["has_helmet"])]

            # Color scheme: Compliant = Emerald Green (0, 230, 118), Violation = Crimson Red (20, 20, 255)
            color = (0, 230, 118) if (is_safe and not missing) else (30, 30, 255)

            # Draw worker bounding box with thick corners
            self._draw_styled_box(frame, (x1, y1, x2, y2), color)

            # Draw spatial head zone
            hx1, hy1, hx2, hy2 = map(int, worker["head_zone"])
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 200, 200), 1, cv2.LINE_AA)

            # Worker status label
            status_text = f"Worker #{worker_id} - SAFE" if (is_safe and not missing) else f"Worker #{worker_id} - NO HELMET"
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Label background badge
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 14, y1), color, -1)
            cv2.putText(frame, status_text, (x1 + 6, max(th + 2, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if is_safe else (255, 255, 255), 2, cv2.LINE_AA)

            # Record alert if violation
            if missing and self._should_alert(worker_id, ",".join(missing)):
                alert = Alert(
                    id=str(uuid.uuid4()),
                    worker_id=worker_id,
                    zone_id=zone.id if zone else None,
                    missing_ppe=missing,
                    severity="critical" if "helmet" in missing else "warning",
                    message=f"Worker #{worker_id} detected without required PPE ({', '.join(missing)}) in {'Zone: ' + zone.name if zone else 'General Site'}",
                    created_at=datetime.now(timezone.utc)
                )
                self.alerts.appendleft(alert)
                new_alerts.append(alert)

        # Draw Cyber HUD telemetry banner on top of frame
        self._draw_hud_banner(frame, active_key, diag["latency_ms"])

        return frame, new_alerts

    def _draw_styled_box(self, frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Add corner brackets
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
        # Semi-transparent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (0, 229, 255), 1)

        # Telemetry Texts
        model_tag = f"AI ENGINE: YOLO-{model_key.upper()}"
        comp_tag = f"COMPLIANCE: {self.last_compliance_rate:.0f}%"
        workers_tag = f"WORKERS: {self.last_worker_count} (VIOLATIONS: {self.last_violation_count})"
        perf_tag = f"{self.fps:.0f} FPS | {latency_ms:.0f}ms"

        # Draw texts
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
            # Transparent fill
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
