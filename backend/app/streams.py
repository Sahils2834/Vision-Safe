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
    if frame is None or not frame.size or frame.shape[0] <= 32 or frame.shape[1] <= 32:
        return False
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(luminance.mean()) > 8.0


def open_camera(index: int, probe_frames: int = 20) -> tuple[cv2.VideoCapture | None, np.ndarray | None, str | None]:
    """Open a camera only after a real frame has been received."""
    labels = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation", cv2.CAP_ANY: "Auto"}
    for backend in CAMERA_BACKENDS:
        try:
            capture = cv2.VideoCapture(index, backend)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Try setting optimal resolution; if unsupported, OpenCV keeps camera default
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)

            if not capture.isOpened():
                capture.release()
                continue

            for _ in range(probe_frames):
                ok, frame = capture.read()
                if ok and usable_frame(frame):
                    return capture, frame, labels.get(backend, "Unknown")
                time.sleep(0.04)
            capture.release()
        except Exception:
            continue
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
    _capture: cv2.VideoCapture | None = None
    _frame_counter: int = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"ppe-stream-{self.id[:8]}")
        self._thread.start()

    def stop(self) -> None:
        self.active = False
        self.state = "stopped"
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

    def jpeg(self, timeout: float = 2.5) -> bytes | None:
        """Get latest frame JPEG, waiting up to timeout seconds if initializing."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return encoded.tobytes() if ok else None
            if not self.active:
                break
            time.sleep(0.05)
        return None

    def iter_mjpeg(self):
        """Yield multipart MJPEG stream for continuous low-latency browser streaming."""
        last_id = -1
        # Wait up to 4s for initial frame
        for _ in range(40):
            if not self.active or self.latest_frame is not None:
                break
            time.sleep(0.1)

        while self.active:
            with self._lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                fid = self._frame_counter

            if frame is not None and fid != last_id:
                last_id = fid
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    data = encoded.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
            time.sleep(0.033)  # ~30fps max cadence

    def _save_frame(self, frame: np.ndarray) -> None:
        annotated, _ = self.monitor.process(frame)
        with self._lock:
            self.latest_frame = annotated
            self._frame_counter += 1

    def _run(self) -> None:
        is_camera = isinstance(self.source, int)
        while self.active:
            if is_camera:
                capture, first_frame, backend = open_camera(self.source)
                if capture is None:
                    self.state = "camera unavailable — retrying"
                    time.sleep(0.8)
                    continue
                with self._lock:
                    self._capture = capture
                self.backend = backend
            else:
                capture = cv2.VideoCapture(str(self.source))
                with self._lock:
                    self._capture = capture
                first_frame = None
                if not capture.isOpened():
                    self.state = "video could not be opened"
                    self.active = False
                    break

            self.state = "live"
            if first_frame is not None:
                self._save_frame(first_frame)

            failed_reads = 0
            while self.active:
                ok, frame = capture.read()
                if not ok or not usable_frame(frame):
                    failed_reads += 1
                    if not is_camera or failed_reads >= 25:
                        break
                    time.sleep(0.04)
                    continue
                failed_reads = 0
                self._save_frame(frame)

            try:
                capture.release()
            except Exception:
                pass
            with self._lock:
                self._capture = None

            if not is_camera:
                self.state = "completed"
                self.active = False
            elif self.active:
                self.state = "reconnecting"
                time.sleep(0.3)


class StreamManager:
    def __init__(self, monitor: PPEMonitor):
        self.monitor = monitor
        self.streams: dict[str, Stream] = {}

    def stop_all(self) -> None:
        """Stop all currently running streams so camera hardware is released cleanly."""
        for stream in list(self.streams.values()):
            if stream.active:
                stream.stop()
        time.sleep(0.2)

    def start(self, source: str | int) -> Stream:
        self.stop_all()
        stream = Stream(source, self.monitor)
        self.streams[stream.id] = stream
        stream.start()
        return stream

    @staticmethod
    def cameras(limit: int = 4) -> list[dict[str, int | str]]:
        """Discovery for the usual Windows camera indices."""
        found = []
        for index in range(limit):
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
            if capture.isOpened():
                ok, _ = capture.read()
                capture.release()
                if ok:
                    label = f"Camera {index + 1} (Default)" if index == 0 else f"Camera {index + 1}"
                    found.append({"id": index, "name": label})
        return found
