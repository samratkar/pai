"""Pydantic data schemas for the 3 vector lanes and WES MQTT payloads."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VisionPayload(BaseModel):
    lane: Literal["vision"] = "vision"
    source_type: Literal["few_shot_reference", "live_observation"]
    label: str
    zone_id: str
    forklift_id: str
    pose_xyz: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    timestamp_epoch: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    ephemeral: bool = True
    wes_task_id: Optional[str] = None
    severity: Optional[str] = None


class AudioPayload(BaseModel):
    lane: Literal["audio"] = "audio"
    transcript: str
    intent_tag: str = "operator_note"
    operator_id: str = "OP-UNKNOWN"
    zone_id: str
    mast_height_mm: Optional[int] = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    timestamp_epoch: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    ephemeral: bool = True
    synced_to_wes: bool = False


class OpsTelemetryPayload(BaseModel):
    lane: Literal["ops_telemetry"] = "ops_telemetry"
    wes_task_id: str
    sku_id: Optional[str] = None
    action: str = "PUTAWAY"
    zone_id: str
    target_bin: Optional[str] = None
    load_weight_kg: float = 0.0
    mast_height_mm: int = 0
    hydraulic_pressure_bar: float = 0.0
    battery_soc_pct: float = 100.0
    semantic_summary: str = ""
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    timestamp_epoch: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    ephemeral: bool = True
    status: str = "IN_PROGRESS"


class EdgeSemanticIncident(BaseModel):
    event_type: Literal["EDGE_SEMANTIC_INCIDENT"] = "EDGE_SEMANTIC_INCIDENT"
    forklift_id: str
    zone_id: str
    vision_classification: str
    vision_confidence: float
    operator_note: Optional[str] = None
    audio_confidence: Optional[float] = None
    active_wes_task: Optional[str] = None
    load_weight_kg: Optional[float] = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)


class UnknownObjectEvent(BaseModel):
    event_type: Literal["EDGE_UNKNOWN_OBJECT"] = "EDGE_UNKNOWN_OBJECT"
    forklift_id: str
    zone_id: str
    candidate_label: Optional[str] = None
    similarity_score: float
    roi_jpeg_b64: str
    vector_512d: List[float]
    timestamp_utc: str = Field(default_factory=utc_now_iso)


class EnrollSkuCommand(BaseModel):
    label: str
    reference_vectors: List[List[float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForgetZoneCommand(BaseModel):
    zone_id: str

