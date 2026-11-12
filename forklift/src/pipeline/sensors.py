"""Vehicle sensor & SocketCAN frame representations for the forklift edge node."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraFrame:
    """Represents a fork/mast optical camera ROI capture."""

    zone_id: str
    pose_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    visual_descriptor: str = ""
    roi_jpeg_bytes: bytes = b"MOCK_ROI_JPEG"

    @property
    def roi_jpeg_b64(self) -> str:
        return base64.b64encode(self.roi_jpeg_bytes).decode("ascii")


@dataclass
class CabAudioClip:
    """Represents a cab/mast microphone capture."""

    zone_id: str
    operator_id: str
    raw_audio_or_transcript: bytes | str
    mast_height_mm: int = 0
    intent_tag: str = "operator_note"


@dataclass
class CanBusSnapshot:
    """1 Hz SocketCAN telemetry snapshot paired with active WES task context."""

    wes_task_id: str
    zone_id: str
    action: str = "PUTAWAY"
    sku_id: Optional[str] = None
    target_bin: Optional[str] = None
    load_weight_kg: float = 0.0
    mast_height_mm: int = 0
    hydraulic_pressure_bar: float = 0.0
    battery_soc_pct: float = 100.0

