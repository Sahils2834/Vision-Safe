from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # A general YOLO model is included for a usable first-run camera demo. Replace
    # it with the PPE-trained weights produced by training/train.py for helmet PPE.
    model_path: str = "../yolo11n.pt"
    confidence_threshold: float = 0.20
    alert_cooldown_seconds: int = 5
    upload_dir: str = "../data/uploads"
    output_dir: str = "../data/outputs"

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def model_file(self) -> Path:
        # Check trained PPE weights first
        candidates = [
            Path("../weights/ppe_yolo11n.pt").resolve(),
            Path("../runs/detect/ppe_yolo11n/weights/best.pt").resolve(),
            Path("../runs/detect/gpu_batch8_test/weights/best.pt").resolve(),
            Path(self.model_path).resolve()
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path(self.model_path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
