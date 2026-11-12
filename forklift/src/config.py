"""Configuration for vector lanes, similarity thresholds, and MQTT topics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for a single isolated Qdrant Edge vector lane."""

    name: str
    dim: int
    threshold: float
    unknown_floor: float = 0.0
    encoder_name: str = ""


DEFAULT_LANES: Dict[str, LaneConfig] = {
    "forklift_vision_lane": LaneConfig(
        name="forklift_vision_lane",
        dim=512,
        threshold=0.82,
        unknown_floor=0.60,
        encoder_name="MobileCLIP-S0-INT8",
    ),
    "forklift_audio_lane": LaneConfig(
        name="forklift_audio_lane",
        dim=384,
        threshold=0.74,
        encoder_name="Whisper-Tiny+bge-small-en-v1.5",
    ),
    "forklift_ops_lane": LaneConfig(
        name="forklift_ops_lane",
        dim=384,
        threshold=0.78,
        encoder_name="StateSerializer+bge-small-en-v1.5",
    ),
}


@dataclass
class EdgeNodeConfig:
    """Top-level configuration for a forklift edge compute node."""

    forklift_id: str = "JH-ETV-216i-09"
    storage_dir: Path = field(default_factory=lambda: Path("./edge_data/qdrant_edge"))
    outbox_db_path: Path = field(default_factory=lambda: Path("./edge_data/outbox.db"))
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 8883
    mqtt_use_tls: bool = True
    freshness_half_life_sec: float = 900.0  # 15-minute half-life for recency decay
    freshness_weight: float = 0.10
    lanes: Dict[str, LaneConfig] = field(default_factory=lambda: dict(DEFAULT_LANES))

    @property
    def topic_incident(self) -> str:
        return f"jh/fleet/{self.forklift_id}/edge/incident"

    @property
    def topic_unknown_object(self) -> str:
        return f"jh/fleet/{self.forklift_id}/edge/unknown_object"

    @property
    def topic_enroll_sku(self) -> str:
        return "jh/fleet/broadcast/edge/enroll_sku"

    @property
    def topic_forget_zone(self) -> str:
        return "jh/fleet/broadcast/edge/forget_zone"

