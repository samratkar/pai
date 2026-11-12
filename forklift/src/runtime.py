"""ForkliftEdgeNode: Top-level runtime combining SiMa encoders, Qdrant Edge, and MQTT bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.config import EdgeNodeConfig
from src.memory.orchestrator import SiMaForkliftEdgeMemory, VisionDecision
from src.pipeline.encoders import AudioEncoder, TelemetryEncoder, VisionEncoder
from src.pipeline.sensors import CabAudioClip, CameraFrame, CanBusSnapshot
from src.sync.mqtt_bridge import EdgeMqttBridge, MqttTransport
from src.sync.outbox import SQLiteOutbox


@dataclass
class FrameProcessingResult:
    decision: VisionDecision
    sync_status: Optional[str] = None  # "SENT" | "QUEUED" | None


class ForkliftEdgeNode:
    """Autonomous edge-memory runtime deployed on a Jungheinrich forklift's SiMa.ai MLSoC."""

    def __init__(
        self,
        config: EdgeNodeConfig,
        transport: MqttTransport,
        prefer_qdrant: bool = True,
        use_fastembed: bool = False,
    ):
        self.config = config
        self.vision_encoder = VisionEncoder(
            dim=config.lanes[SiMaForkliftEdgeMemory.VISION_LANE].dim,
            use_fastembed=use_fastembed,
        )
        self.audio_encoder = AudioEncoder(
            dim=config.lanes[SiMaForkliftEdgeMemory.AUDIO_LANE].dim,
            use_fastembed=use_fastembed,
        )
        self.ops_encoder = TelemetryEncoder(
            dim=config.lanes[SiMaForkliftEdgeMemory.OPS_LANE].dim,
            use_fastembed=use_fastembed,
        )
        self.memory = SiMaForkliftEdgeMemory(config=config, prefer_qdrant=prefer_qdrant)
        self.outbox = SQLiteOutbox(config.outbox_db_path)
        self.bridge = EdgeMqttBridge(self.memory, self.outbox, transport)

        self._last_audio_vec_by_zone: Dict[str, List[float]] = {}
        self._last_ops_vec_by_zone: Dict[str, List[float]] = {}
        self._active_task_by_zone: Dict[str, str] = {}

    def process_can_snapshot(self, snap: CanBusSnapshot) -> str:
        summary = self.ops_encoder.serialize_can_state(
            action=snap.action,
            load_weight_kg=snap.load_weight_kg,
            sku_id=snap.sku_id,
            zone_id=snap.zone_id,
            mast_height_mm=snap.mast_height_mm,
            battery_soc_pct=snap.battery_soc_pct,
        )
        vec = self.ops_encoder.encode_state(summary)
        self._last_ops_vec_by_zone[snap.zone_id] = vec
        self._active_task_by_zone[snap.zone_id] = snap.wes_task_id
        return self.memory.ingest_ops_telemetry(
            vector=vec,
            wes_task_id=snap.wes_task_id,
            zone_id=snap.zone_id,
            semantic_summary=summary,
            action=snap.action,
            sku_id=snap.sku_id,
            target_bin=snap.target_bin,
            load_weight_kg=snap.load_weight_kg,
            mast_height_mm=snap.mast_height_mm,
            hydraulic_pressure_bar=snap.hydraulic_pressure_bar,
            battery_soc_pct=snap.battery_soc_pct,
        )

    def process_audio_clip(self, clip: CabAudioClip) -> str:
        transcript = self.audio_encoder.transcribe(clip.raw_audio_or_transcript)
        vec = self.audio_encoder.encode_transcript(transcript)
        self._last_audio_vec_by_zone[clip.zone_id] = vec
        return self.memory.ingest_audio_event(
            vector=vec,
            transcript=transcript,
            zone_id=clip.zone_id,
            operator_id=clip.operator_id,
            intent_tag=clip.intent_tag,
            mast_height_mm=clip.mast_height_mm,
        )

    def process_camera_frame(self, frame: CameraFrame) -> FrameProcessingResult:
        vec = self.vision_encoder.encode_label_or_frame(frame.visual_descriptor)
        active_task = self._active_task_by_zone.get(frame.zone_id)
        decision = self.memory.evaluate_vision_observation(
            frame_vector=vec,
            zone_id=frame.zone_id,
            roi_jpeg_b64=frame.roi_jpeg_b64,
            pose_xyz=frame.pose_xyz,
            wes_task_id=active_task,
        )

        if decision.status == "MATCHED":
            audio_vec = self._last_audio_vec_by_zone.get(frame.zone_id, [0.0] * 384)
            ops_vec = self._last_ops_vec_by_zone.get(frame.zone_id, [0.0] * 384)
            incident = self.memory.build_wes_incident_bundle(
                zone_id=frame.zone_id,
                vision_vec=vec,
                audio_vec=audio_vec,
                ops_vec=ops_vec,
            )
            sync_status = self.bridge.publish_or_queue(
                self.config.topic_incident,
                incident.model_dump(),
            )
            return FrameProcessingResult(decision=decision, sync_status=sync_status)

        if decision.status == "UNKNOWN_OBJECT" and decision.unknown_event is not None:
            sync_status = self.bridge.publish_or_queue(
                self.config.topic_unknown_object,
                decision.unknown_event.model_dump(),
            )
            return FrameProcessingResult(decision=decision, sync_status=sync_status)

        return FrameProcessingResult(decision=decision, sync_status=None)

    def reconnect_and_sync(self) -> int:
        return self.bridge.sync_outbox()

    def close(self) -> None:
        self.outbox.close()
        self.memory.close()

