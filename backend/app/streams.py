from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .monitor import PPEMonitor

CAMERA_BACKENDS = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)


def usable_frame(frame: np.ndarray | None) -> bool:
    """Reject the common Windows-camera black/static frame failure mode."""
    if frame is None or not frame.size or frame.shape[0] <= 32 or frame.shape[1] <= 32: return False
    # A real scene has a little luminance. This lets us fall through from a
    # DirectShow black frame to Media Foundation for many Logitech cameras.
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(luminance.mean()) > 8.0


def open_camera(index: int, probe_frames: int = 20) -> tuple[cv2.VideoCapture | None, np.ndarray | None, str | None]:
    """Open a camera only after a real frame has been received."""
    labels = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation", cv2.CAP_ANY: "Auto"}
    for backend in CAMERA_BACKENDS:
        capture = cv2.VideoCapture(index, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)
        if not capture.isOpened():
            capture.release()
            continue
        for _ in range(probe_frames):
            ok, frame = capture.read()
            if ok and usable_frame(frame):
                return capture, frame, labels[backend]
            time.sleep(.03)
        capture.release()
    return None, None, None


@dataclass
class Stream:
    source: str | int
    monitor: PPEMonitor
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    latest_frame: np.ndarray | None = None
    state: str = "starting"
    backend: str | None = None
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"ppe-stream-{self.id[:8]}")
        self._thread.start()

    def stop(self) -> None:
        self.active = False
        self.state = "stopped"

    def jpeg(self) -> bytes | None:
        with self._lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None: return None
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else None

    def _save_frame(self, frame: np.ndarray) -> None:
        annotated, _ = self.monitor.process(frame)
        with self._lock: self.latest_frame = annotated

    def _run(self) -> None:
        is_camera = isinstance(self.source, int)
        while self.active:
            if is_camera:
                capture, first_frame, backend = open_camera(self.source)
                if capture is None:
                    self.state = "camera unavailable — retrying"
                    time.sleep(1)
                    continue
                self.backend = backend
            else:
                capture, first_frame = cv2.VideoCapture(str(self.source)), None
                if not capture.isOpened():
                    self.state = "video could not be opened"
                    self.active = False
                    break
            self.state = "live"
            if first_frame is not None: self._save_frame(first_frame)
            failed_reads = 0
            invalid_frames = 0
            while self.active:
                ok, frame = capture.read()
                if not ok or not usable_frame(frame):
                    failed_reads += 1
                    invalid_frames += 1
                    if not is_camera or failed_reads >= 20: break
                    time.sleep(.05)
                    continue
                failed_reads = invalid_frames = 0
                self._save_frame(frame)
            capture.release()
            if not is_camera:
                self.state = "completed"
                self.active = False
            elif self.active:
                self.state = "reconnecting"
                time.sleep(.25)


class StreamManager:
    def __init__(self, monitor: PPEMonitor): self.monitor, self.streams = monitor, {}

    def start(self, source: str | int) -> Stream:
        stream = Stream(source, self.monitor); self.streams[stream.id] = stream; stream.start(); return stream

    @staticmethod
    def cameras(limit: int = 4) -> list[dict[str, int | str]]:
        """Discovery for the usual Windows camera indices."""
        found = []
        for index in range(limit):
            # Only use cv2.CAP_ANY for fast scanning to avoid DSHOW hangs on invalid cameras
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
            if capture.isOpened():
                ok, _ = capture.read()
                capture.release()
                if ok:
                    found.append({"id": index, "name": f"Camera {index + 1} (Auto)"})
        return found
