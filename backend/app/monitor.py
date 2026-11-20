
"""VisionSafe frame monitoring, worker tracking, PPE evaluation and alerts."""

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
    PPE_ITEMS = ("helmet", "gloves", "vest", "boots", "goggles")

    DEFAULT_REQUIREMENTS = ["helmet", "gloves", "vest"]

    PPE_ALIASES = {
        "hat": "helmet",
        "hardhat": "helmet",
        "safety_helmet": "helmet",
        "glove": "gloves",
        "construction_gloves": "gloves",
        "safety_vest": "vest",
        "ppe_kit": "vest",
        "ppe kit": "vest",
        "boot": "boots",
        "safety_boots": "boots",
        "shoe": "boots",
        "shoes": "boots",
        "safety_shoes": "boots",
        "goggle": "goggles",
        "glasses": "goggles",
        "safety_glasses": "goggles",
    }

    ALERT_MAP = {
        "helmet": (
            "Safety Helmet",
            "critical",
            "No Safety Helmet detected.",
        ),
        "gloves": (
            "Construction Gloves",
            "warning",
            "No Construction Gloves detected.",
        ),
        "vest": (
            "Safety Vest / PPE Kit",
            "critical",
            "No Safety Vest detected.",
        ),
        "boots": (
            "Safety Boots / Shoes",
            "warning",
            "No Safety Boots detected.",
        ),
        "goggles": (
            "Safety Goggles",
            "warning",
            "No Safety Goggles detected.",
        ),
    }

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

        # ByteTrack ID -> persistent VisionSafe worker ID.
        self._track_workers: dict[int, int] = {}
        self._next_worker_id = 1

        # Temporal PPE history:
        # worker_id -> PPE item -> recent visible/present observations.
        self._ppe_history: dict[
            int, dict[str, deque]
        ] = {}

        # Keep worker state briefly after a worker leaves the frame.
        self._worker_last_seen: dict[int, float] = {}

        self._history_length = 5
        self._missing_confirmations = 3
        self._worker_memory_seconds = 10.0

        # Real-time telemetry.
        self.last_compliance_rate = 100.0
        self.last_worker_count = 0
        self.last_violation_count = 0
        self.last_unverified_count = 0

        self.fps = 0.0
        self._last_frame_time = time.perf_counter()

    # ---------------------------------------------------------
    # MODEL STATUS
    # ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        return (
            len(model_registry.loaded_models) > 0
            or model_registry.get_model() is not None
        )

    @property
    def model_name(self) -> str | None:
        return f"YOLO-{model_registry.active_model_key.upper()}"

    @property
    def supports_ppe(self) -> bool:
        return True

    def set_zones(self, zones: list[Zone]) -> None:
        self.zones = zones

    # ---------------------------------------------------------
    # BOX VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _valid_box(
        box,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if box is None or len(box) != 4:
            return False

        try:
            x1, y1, x2, y2 = map(float, box)
        except (TypeError, ValueError):
            return False

        return (
            np.isfinite([x1, y1, x2, y2]).all()
            and x1 >= 0
            and y1 >= 0
            and x2 <= frame_width
            and y2 <= frame_height
            and x2 > x1
            and y2 > y1
        )

    # ---------------------------------------------------------
    # WORKER TRACKING
    # ---------------------------------------------------------

    def _update_worker_tracking(
        self,
        workers: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> list[dict]:
        """
        Run ByteTrack on person detections and assign persistent
        VisionSafe worker IDs to the detector's worker dictionaries.
        """

        valid_workers = []

        for worker in workers:
            box = worker.get("box")

            if self._valid_box(box, frame_width, frame_height):
                valid_workers.append(worker)

        if not valid_workers:
            # Update tracker with an empty detection set so lost tracks
            # can age out correctly.
            empty = sv.Detections.empty()
            self.tracker.update_with_detections(empty)
            self._cleanup_worker_history()
            return []

        boxes = np.array(
            [worker["box"] for worker in valid_workers],
            dtype=np.float32,
        )

        confidences = np.array(
            [
                float(worker.get("confidence", 0.5))
                for worker in valid_workers
            ],
            dtype=np.float32,
        )

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.zeros(len(boxes), dtype=int),
        )

        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None:
            self._cleanup_worker_history()
            return valid_workers

        for index, track_id in enumerate(tracked.tracker_id):
            if track_id is None:
                continue

            track_id = int(track_id)

            if track_id not in self._track_workers:
                self._track_workers[track_id] = self._next_worker_id
                self._next_worker_id += 1

            worker_id = self._track_workers[track_id]

            if index < len(valid_workers):
                valid_workers[index]["id"] = worker_id

                self._worker_last_seen[worker_id] = time.monotonic()

        self._cleanup_worker_history()

        return valid_workers

    def _cleanup_worker_history(self) -> None:
        """Remove stale tracking and PPE history."""

        now = time.monotonic()

        expired = [
            worker_id
            for worker_id, last_seen in self._worker_last_seen.items()
            if now - last_seen > self._worker_memory_seconds
        ]

        for worker_id in expired:
            self._worker_last_seen.pop(worker_id, None)
            self._ppe_history.pop(worker_id, None)

        active_worker_ids = set(self._worker_last_seen.values())

        # Remove stale alert cooldown entries.
        stale_keys = [
            key
            for key, timestamp in self._last_alert.items()
            if now - timestamp > max(
                60,
                self.settings.alert_cooldown_seconds * 10,
            )
        ]

        for key in stale_keys:
            self._last_alert.pop(key, None)

    # ---------------------------------------------------------
    # PPE VISIBILITY
    # ---------------------------------------------------------

    @staticmethod
    def _get_ppe_visibility(
        worker: dict,
        item: str,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Determine whether an absent PPE item can be assessed.

        Explicit detector visibility flags take priority.

        Gloves and goggles are unverified unless the detector
        explicitly provides their visibility.
        """

        explicit_key = f"{item}_visible"

        if explicit_key in worker:
            return bool(worker[explicit_key])

        worker_box = worker.get("box")

        if not PPEMonitor._valid_box(
            worker_box,
            frame_width,
            frame_height,
        ):
            return False

        x1, y1, x2, y2 = map(float, worker_box)

        if item == "helmet":
            head_zone = worker.get("head_zone")

            return PPEMonitor._valid_box(
                head_zone,
                frame_width,
                frame_height,
            )

        if item == "vest":
            worker_height = y2 - y1
            worker_width = x2 - x1

            if worker_height < 50 or worker_width < 25:
                return False

            torso_top = y1 + worker_height * 0.18
            torso_bottom = y1 + worker_height * 0.72

            return (
                torso_top >= 0
                and torso_bottom <= frame_height
                and x1 >= 0
                and x2 <= frame_width
            )

        if item == "boots":
            worker_height = y2 - y1

            if worker_height < 50:
                return False

            lower_body_top = y1 + worker_height * 0.72

            return (
                lower_body_top >= 0
                and y2 <= frame_height - 2
            )

        if item in ("gloves", "goggles"):
            return False

        return False

    @staticmethod
    def _get_item_presence(worker: dict, item: str) -> bool:
        return bool(worker.get(f"has_{item}", False))

    # ---------------------------------------------------------
    # TEMPORAL PPE CONFIRMATION
    # ---------------------------------------------------------

    def _get_worker_history(self, worker_id: int) -> dict:
        if worker_id not in self._ppe_history:
            self._ppe_history[worker_id] = {
                item: deque(maxlen=self._history_length)
                for item in self.PPE_ITEMS
            }

        return self._ppe_history[worker_id]

    def _smooth_ppe_status(
        self,
        worker_id: int,
        worker: dict,
        item: str,
        visible: bool,
    ) -> str:
        """
        Confirm missing PPE over multiple frames.

        Positive detections are accepted immediately.
        A missing result must persist for several visible frames.
        Invisible PPE remains unverified.
        """

        detected = self._get_item_presence(worker, item)

        history = self._get_worker_history(worker_id)[item]

        if detected:
            history.append(True)
            return "DETECTED"

        if not visible:
            # Do not turn an occluded or unobservable item into
            # a confirmed violation.
            history.append(None)
            return "UNVERIFIED"

        history.append(False)

        visible_observations = [
            value for value in history if value is not None
        ]

        missing_count = sum(
            value is False for value in visible_observations
        )

        if missing_count >= self._missing_confirmations:
            return "MISSING"

        return "UNVERIFIED"

    # ---------------------------------------------------------
    # PPE EVALUATION
    # ---------------------------------------------------------

    def _evaluate_worker(
        self,
        worker: dict,
        requirements: list[str],
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        missing = []
        unverified = []
        statuses = {}

        normalized_requirements = []

        for requirement in requirements:
            normalized = self.PPE_ALIASES.get(
                str(requirement).lower().strip(),
                str(requirement).lower().strip(),
            )

            if (
                normalized in self.PPE_ITEMS
                and normalized not in normalized_requirements
            ):
                normalized_requirements.append(normalized)

        worker_id = int(worker.get("id", 0))

        for item in self.PPE_ITEMS:
            detected = self._get_item_presence(worker, item)

            if detected:
                statuses[item] = "DETECTED"

                # Keep positive observations in temporal history.
                self._smooth_ppe_status(
                    worker_id,
                    worker,
                    item,
                    True,
                )
                continue

            if item not in normalized_requirements:
                statuses[item] = "NOT_REQUIRED"
                continue

            visible = self._get_ppe_visibility(
                worker,
                item,
                frame_width,
                frame_height,
            )

            status = self._smooth_ppe_status(
                worker_id,
                worker,
                item,
                visible,
            )

            statuses[item] = status

            if status == "MISSING":
                missing.append(item)

            elif status == "UNVERIFIED":
                unverified.append(item)

        return missing, unverified, statuses

    # ---------------------------------------------------------
    # MAIN FRAME PROCESSING
    # ---------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[Alert]]:

        now_time = time.perf_counter()
        delta = now_time - self._last_frame_time

        if delta > 0:
            instant_fps = 1.0 / delta

            if self.fps <= 0:
                self.fps = round(instant_fps, 1)
            else:
                self.fps = round(
                    0.9 * self.fps + 0.1 * instant_fps,
                    1,
                )

        self._last_frame_time = now_time

        active_key = model_registry.active_model_key
        model = model_registry.get_model()
        is_ensemble = active_key == "ensemble"

        if model is None and not is_ensemble:
            return self._draw_standby_hud(frame), []

        diag = hybrid_detector.detect_and_validate(
            model=model,
            frame=frame,
            conf_threshold=self.settings.confidence_threshold,
            is_ensemble=is_ensemble,
            all_models=model_registry.loaded_models,
        )

        workers = diag.get("workers", [])

        helmets = diag.get("helmets", [])
        gloves = diag.get("gloves", [])
        vests = diag.get("vests", [])
        boots = diag.get("boots", [])
        goggles = diag.get("goggles", [])

        h, w = frame.shape[:2]

        workers = self._update_worker_tracking(
            workers,
            w,
            h,
        )

        new_alerts: list[Alert] = []

        self._draw_zones(frame)

        # -----------------------------------------------------
        # PPE DETECTION BOXES
        # -----------------------------------------------------

        def draw_item_box(items, label_name, color):
            for item in items:
                box = item.get("box")

                if not self._valid_box(box, w, h):
                    continue

                ix1, iy1, ix2, iy2 = map(int, box)

                confidence = float(item.get("confidence", 0))

                cv2.rectangle(
                    frame,
                    (ix1, iy1),
                    (ix2, iy2),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    f"{label_name} {confidence:.2f}",
                    (ix1, max(15, iy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        draw_item_box(helmets, "Helmet", (255, 230, 0))
        draw_item_box(gloves, "Gloves", (0, 140, 255))
        draw_item_box(vests, "Vest", (0, 240, 255))
        draw_item_box(boots, "Boots", (255, 180, 0))
        draw_item_box(goggles, "Goggles", (255, 50, 200))

        # -----------------------------------------------------
        # WORKER EVALUATION
        # -----------------------------------------------------

        safe_workers = 0
        violation_workers = 0
        unverified_workers = 0
        assessed_workers = 0

        for worker in workers:

            box = worker.get("box")

            if not self._valid_box(box, w, h):
                continue

            x1, y1, x2, y2 = map(int, box)

            worker_id = int(
                worker.get("id", self._next_worker_id)
            )

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2,
            )

            zone = self._zone_for(center)

            requirements = (
                list(zone.required_ppe)
                if zone and zone.required_ppe
                else list(self.DEFAULT_REQUIREMENTS)
            )

            missing, unverified, statuses = self._evaluate_worker(
                worker,
                requirements,
                w,
                h,
            )

            worker["missing_ppe"] = missing
            worker["unverified_ppe"] = unverified
            worker["ppe_status"] = statuses

            # A confirmed missing item is a violation.
            # An unverified item is not automatically a violation.
            if missing:
                status = "VIOLATION"
                is_safe = False
                violation_workers += 1

            elif unverified:
                status = "UNVERIFIED"
                is_safe = False
                unverified_workers += 1

            else:
                status = "COMPLIANT"
                is_safe = True
                safe_workers += 1

            worker["is_safe"] = is_safe
            worker["status"] = status

            if not missing and not unverified:
                assessed_workers += 1

            if status == "COMPLIANT":
                color = (0, 230, 118)
            elif status == "UNVERIFIED":
                color = (0, 200, 255)
            else:
                color = (30, 30, 255)

            self._draw_styled_box(
                frame,
                (x1, y1, x2, y2),
                color,
            )

            # Head region.
            head_zone = worker.get("head_zone")

            if self._valid_box(head_zone, w, h):
                hx1, hy1, hx2, hy2 = map(int, head_zone)

                cv2.rectangle(
                    frame,
                    (hx1, hy1),
                    (hx2, hy2),
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            # Worker status label.
            status_text = f"Worker #{worker_id} - {status}"

            (tw, th), _ = cv2.getTextSize(
                status_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                2,
            )

            label_y = max(0, y1 - th - 14)

            cv2.rectangle(
                frame,
                (x1, label_y),
                (x1 + tw + 16, y1),
                color,
                -1,
            )

            cv2.putText(
                frame,
                status_text,
                (x1 + 8, max(th + 2, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0) if status == "COMPLIANT"
                else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # PPE STATUS CHIPS
            # -------------------------------------------------

            chip_y = min(h - 10, y2 + 20)
            chip_x = x1

            chip_colors = {
                "DETECTED": (0, 180, 90),
                "MISSING": (30, 30, 255),
                "UNVERIFIED": (0, 160, 220),
                "NOT_REQUIRED": (90, 90, 90),
            }

            for item in self.PPE_ITEMS:
                item_status = statuses[item]
                chip_text = f"{item.upper()}: {item_status}"
                chip_color = chip_colors[item_status]

                (cw, ch), _ = cv2.getTextSize(
                    chip_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    1,
                )

                if chip_x + cw + 10 >= w:
                    break

                cv2.rectangle(
                    frame,
                    (chip_x, chip_y - ch - 4),
                    (chip_x + cw + 8, chip_y + 4),
                    chip_color,
                    -1,
                )

                cv2.putText(
                    frame,
                    chip_text,
                    (chip_x + 4, chip_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                chip_x += cw + 12

            # -------------------------------------------------
            # INDIVIDUAL PPE ALERTS
            # -------------------------------------------------

            # Only confirmed MISSING items generate alerts.
            # UNVERIFIED items do not generate missing-PPE alerts.

            for item in missing:
                if not self._should_alert(worker_id, item):
                    continue

                display_name, severity, description = self.ALERT_MAP[item]

                alert = Alert(
                    id=str(uuid.uuid4()),
                    worker_id=worker_id,
                    zone_id=zone.id if zone else None,
                    missing_ppe=[display_name],
                    severity=severity,
                    message=f"Worker #{worker_id} — {description}",
                    created_at=datetime.now(timezone.utc),
                )

                self.alerts.appendleft(alert)
                new_alerts.append(alert)

        # -----------------------------------------------------
        # TELEMETRY
        # -----------------------------------------------------

        self.last_worker_count = len(workers)
        self.last_violation_count = violation_workers
        self.last_unverified_count = unverified_workers

        if assessed_workers > 0:
            self.last_compliance_rate = round(
                safe_workers / assessed_workers * 100.0,
                1,
            )

        elif workers:
            self.last_compliance_rate = 0.0

        else:
            self.last_compliance_rate = 100.0

        self._draw_hud_banner(
            frame,
            active_key,
            diag.get("latency_ms", 0.0),
        )

        return frame, new_alerts

    # ---------------------------------------------------------
    # DRAWING HELPERS
    # ---------------------------------------------------------

    def _draw_styled_box(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ):

        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        corner_len = max(
            1,
            min(20, (x2 - x1) // 4, (y2 - y1) // 4),
        )

        thickness = 4

        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def _draw_hud_banner(
        self,
        frame: np.ndarray,
        model_key: str,
        latency_ms: float,
    ):

        h, w = frame.shape[:2]
        hud_h = 44

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, hud_h),
            (15, 17, 23),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0,
            frame,
        )

        cv2.line(
            frame,
            (0, hud_h),
            (w, hud_h),
            (0, 229, 255),
            1,
        )

        model_tag = f"AI ENGINE: YOLO-{model_key.upper()}"

        comp_tag = (
            f"COMPLIANCE: {self.last_compliance_rate:.0f}%"
            f" | VIOLATIONS: {self.last_violation_count}"
            f" | UNVERIFIED: {self.last_unverified_count}"
        )

        perf_tag = f"{self.fps:.0f} FPS | {latency_ms:.0f}ms"

        comp_color = (
            (0, 230, 118)
            if self.last_compliance_rate == 100
            else (
                (0, 200, 255)
                if self.last_violation_count == 0
                else (30, 30, 255)
            )
        )

        cv2.putText(
            frame,
            model_tag,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 229, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            comp_tag,
            (w // 2 - 160, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            comp_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            perf_tag,
            (max(0, w - 180), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_standby_hud(self, frame: np.ndarray) -> np.ndarray:

        cv2.putText(
            frame,
            "VisionSafe AI: Standby - No Model Loaded",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 180, 255),
            2,
        )

        return frame

    # ---------------------------------------------------------
    # ZONES
    # ---------------------------------------------------------

    def _zone_for(
        self,
        point: tuple[int, int],
    ) -> Zone | None:

        for zone in self.zones:
            if len(zone.polygon) < 3:
                continue

            contour = np.array(
                [[p.x, p.y] for p in zone.polygon],
                dtype=np.int32,
            )

            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return zone

        return None

    def _draw_zones(self, frame: np.ndarray) -> None:

        for zone in self.zones:
            if len(zone.polygon) < 3:
                continue

            contour = np.array(
                [[p.x, p.y] for p in zone.polygon],
                dtype=np.int32,
            )

            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [contour],
                (255, 140, 0),
            )

            cv2.addWeighted(
                overlay,
                0.2,
                frame,
                0.8,
                0,
                frame,
            )

            cv2.polylines(
                frame,
                [contour],
                True,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

            if len(contour) > 0:
                cv2.putText(
                    frame,
                    f"ZONE: {zone.name.upper()}",
                    tuple(contour[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

    # ---------------------------------------------------------
    # ALERT COOLDOWN
    # ---------------------------------------------------------

    def _should_alert(
        self,
        worker_id: int,
        issue: str,
    ) -> bool:

        key = (worker_id, issue)
        now = time.monotonic()

        cooldown = max(
            0,
            self.settings.alert_cooldown_seconds,
        )

        if now - self._last_alert.get(key, 0) < cooldown:
            return False

        self._last_alert[key] = now

        return True
