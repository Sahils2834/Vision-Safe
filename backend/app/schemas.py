from datetime import datetime
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Point(BaseModel):
    x: int
    y: int

class Zone(BaseModel):
    id: str
    name: str
    polygon: list[Point] = Field(min_length=3)
    required_ppe: list[str] = ["helmet"]

class Alert(BaseModel):
    id: str
    worker_id: int
    zone_id: Optional[str] = None
    missing_ppe: list[str]
    severity: Literal["warning", "critical"] = "warning"
    message: Optional[str] = None
    created_at: datetime
    snapshot_url: Optional[str] = None

class StreamStart(BaseModel):
    camera_index: int = 0

class ModelSelectRequest(BaseModel):
    model_key: str

class WorkerItem(BaseModel):
    id: int
    box: List[float]
    confidence: float
    head_zone: List[float]
    is_safe: bool
    has_helmet: bool
    helmet_confidence: float
    matched_helmet_box: Optional[List[float]] = None
    missing_ppe: List[str] = []
    status: str

class HelmetItem(BaseModel):
    box: List[float]
    confidence: float
    matched_worker_id: Optional[int] = None

class DiagnosisResponse(BaseModel):
    success: bool
    active_model: str
    workers: List[WorkerItem]
    helmets: List[HelmetItem]
    safe_workers_count: int
    total_workers_count: int
    violation_count: int
    compliance_rate: float
    latency_ms: float
    safety_index: str
    annotated_image_base64: Optional[str] = None

class VideoFrameAudit(BaseModel):
    frame_index: int
    timestamp_sec: float
    total_workers: int
    safe_workers: int
    violations: int
    compliance_rate: float

class VideoAnalysisResponse(BaseModel):
    success: bool
    filename: str
    total_frames: int
    analyzed_frames: int
    avg_compliance_rate: float
    peak_violations: int
    total_violation_events: int
    safety_index: str
    frame_audits: List[VideoFrameAudit]
    latency_ms: float
