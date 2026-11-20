
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .monitor import PPEMonitor


CAMERA_BACKENDS = (
    cv2.CAP_DSHOW,
    cv2.CAP_MSMF,
    cv2.CAP_ANY,
)


def valid_frame(frame: np.ndarray | None) -> bool:
    """Check whether a frame is a valid, non-empty image."""
    if frame is None:
        return False

    if not isinstance(frame, np.ndarray):
        return False

    if frame.size == 0 or frame.ndim != 3:
        return False

    if frame.shape[0] <= 32 or frame.shape[1] <= 32:
        return False

    return True


def usable_frame(frame: np.ndarray | None) -> bool:
    """
    Check whether a webcam frame is usable.

    The brightness check is intentionally restricted to webcam
    initialization because legitimate uploaded videos may be dark.
    """
    if not valid_frame(frame):
        return False

    try:
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(luminance.mean()) > 8.0
    except Exception:
        return False


def open_camera(
    index: int,
    probe_frames: int = 20,
) -> tuple[cv2.VideoCapture | None, np.ndarray | None, str | None]:
    """Open a camera only after receiving a usable frame."""

    labels = {
        cv2.CAP_DSHOW: "DirectShow",
        cv2.CAP_MSMF: "Media Foundation",
        cv2.CAP_ANY: "Auto",
    }

    for backend in CAMERA_BACKENDS:
        capture = None

        try:
            capture = cv2.VideoCapture(index, backend)

            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)

            for _ in range(probe_frames):
                ok, frame = capture.read()

                if ok and usable_frame(frame):
                    return capture, frame, labels.get(backend, "Unknown")

                time.sleep(0.04)

            capture.release()

        except Exception:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

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
        """Start stream processing in a background thread."""

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"ppe-stream-{self.id[:8]}",
        )

        self._thread.start()

    def stop(self) -> None:
        """Stop the stream and release its capture device."""

        self.active = False
        self.state = "stopped"

        with self._lock:
            capture = self._capture
            self._capture = None

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def jpeg(self, timeout: float = 2.5) -> bytes | None:
        """Return the latest processed frame as JPEG."""

        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                frame = (
                    self.latest_frame.copy()
                    if self.latest_frame is not None
                    else None
                )

            if frame is not None:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )

                return encoded.tobytes() if ok else None

            if not self.active:
                break

            time.sleep(0.05)

        return None

    def iter_mjpeg(self):
        """Yield multipart MJPEG frames for browser streaming."""

        last_id = -1

        # Wait for the first frame.
        for _ in range(40):
            if not self.active:
                break

            with self._lock:
                has_frame = self.latest_frame is not None

            if has_frame:
                break

            time.sleep(0.1)

        while self.active:
            with self._lock:
                frame = (
                    self.latest_frame.copy()
                    if self.latest_frame is not None
                    else None
                )

                frame_id = self._frame_counter

            if frame is not None and frame_id != last_id:
                last_id = frame_id

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )

                if ok:
                    data = encoded.tobytes()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )

            time.sleep(0.033)

    def _save_frame(self, frame: np.ndarray) -> None:
        """Run PPE detection and save the annotated frame."""

        annotated, _ = self.monitor.process(frame)

        if annotated is None or not isinstance(annotated, np.ndarray):
            return

        with self._lock:
            self.latest_frame = annotated.copy()
            self._frame_counter += 1

    def _set_capture(self, capture: cv2.VideoCapture | None) -> None:
        """Safely update the active capture reference."""

        with self._lock:
            self._capture = capture

    def _release_capture(self, capture: cv2.VideoCapture | None) -> None:
        """Release capture and clear the active reference."""

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        with self._lock:
            if self._capture is capture:
                self._capture = None

    def _run_video(self) -> None:
        """Process an uploaded video while respecting its original FPS."""

        capture = None

        try:
            video_path = str(self.source)

            capture = cv2.VideoCapture(video_path)

            if not capture.isOpened():
                self.state = "video could not be opened"
                self.active = False
                return

            self._set_capture(capture)

            fps = capture.get(cv2.CAP_PROP_FPS)

            # Some video files report invalid FPS metadata.
            if not np.isfinite(fps) or fps <= 0 or fps > 240:
                fps = 30.0

            frame_interval = 1.0 / fps
            next_frame_time = time.monotonic()

            self.state = "live"

            while self.active:
                ok, frame = capture.read()

                if not ok:
                    # A failed read at the end of a video is normal.
                    self.state = "completed"
                    break

                # Do not apply the webcam brightness filter to videos.
                # Dark footage is still valid footage.
                if not valid_frame(frame):
                    continue

                self._save_frame(frame)

                # Pace playback approximately according to the
                # video's original frame rate.
                next_frame_time += frame_interval

                remaining = next_frame_time - time.monotonic()

                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # If detection takes longer than the source FPS,
                    # continue without accumulating a playback delay.
                    next_frame_time = time.monotonic()

        except Exception as exc:
            self.state = f"video processing error: {exc}"

        finally:
            self._release_capture(capture)
            self.active = False

            if self.state == "live":
                self.state = "completed"

    def _run_camera(self) -> None:
        """Process a webcam with automatic reconnection."""

        while self.active:
            capture = None

            try:
                capture, first_frame, backend = open_camera(self.source)

                if capture is None:
                    self.state = "camera unavailable - retrying"
                    time.sleep(0.8)
                    continue

                self._set_capture(capture)
                self.backend = backend
                self.state = "live"

                if first_frame is not None:
                    self._save_frame(first_frame)

                failed_reads = 0

                while self.active:
                    ok, frame = capture.read()

                    if not ok or not usable_frame(frame):
                        failed_reads += 1

                        if failed_reads >= 25:
                            break

                        time.sleep(0.04)
                        continue

                    failed_reads = 0

                    self._save_frame(frame)

            except Exception as exc:
                self.state = f"camera processing error: {exc}"

            finally:
                self._release_capture(capture)

            if self.active:
                self.state = "reconnecting"
                time.sleep(0.3)

        if self.state not in ("stopped",):
            self.state = "stopped"

    def _run(self) -> None:
        """Choose the appropriate processing pipeline."""

        try:
            is_camera = isinstance(self.source, int)

            if is_camera:
                self._run_camera()
            else:
                self._run_video()

        except Exception as exc:
            self.state = f"stream error: {exc}"

        finally:
            self.active = False

            with self._lock:
                capture = self._capture
                self._capture = None

            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass


class StreamManager:
    """Manage active webcam and uploaded-video streams."""

    def __init__(self, monitor: PPEMonitor):
        self.monitor = monitor
        self.streams: dict[str, Stream] = {}

    def stop_all(self) -> None:
        """Stop all running streams."""

        for stream in list(self.streams.values()):
            if stream.active:
                stream.stop()

        time.sleep(0.2)

    def start(self, source: str | int) -> Stream:
        """
        Start a new stream.

        Existing streams are stopped before starting a new one,
        preserving the original single-active-stream behavior.
        """

        self.stop_all()

        stream = Stream(
            source=source,
            monitor=self.monitor,
        )

        self.streams[stream.id] = stream
        stream.start()

        return stream

    @staticmethod
    def cameras(limit: int = 4) -> list[dict[str, int | str]]:
        """Discover available Windows camera devices."""

        found = []

        for index in range(limit):
            capture = None

            try:
                capture = cv2.VideoCapture(index, cv2.CAP_ANY)

                if capture.isOpened():
                    ok, frame = capture.read()

                    if ok and usable_frame(frame):
                        label = (
                            f"Camera {index + 1} (Default)"
                            if index == 0
                            else f"Camera {index + 1}"
                        )

                        found.append({
                            "id": index,
                            "name": label,
                        })

            except Exception:
                pass

            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass

        return found
