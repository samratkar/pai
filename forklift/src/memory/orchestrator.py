"""SiMaForkliftEdgeMemory: Edge Memory Orchestrator for Jungheinrich Forklifts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import EdgeNodeConfig
from src.memory.store import SearchHit, create_vector_store
from src.models import (
    AudioPayload,
    EdgeSemanticIncident,
    OpsTelemetryPayload,
    UnknownObjectEvent,
    VisionPayload,
)


@dataclass
class VisionDecision:
    """Result of gating a live camera observation against forklift_vision_lane."""

    status: str  # "MATCHED" | "UNKNOWN_OBJECT" | "IGNORED"
    label: str
    score: float
    point_id: Optional[str] = None
    unknown_event: Optional[UnknownObjectEvent] = None


class SiMaForkliftEdgeMemory:
    """Edge Memory Orchestrator managing 3 isolated Qdrant Edge lanes."""

    VISION_LANE = "forklift_vision_lane"
    AUDIO_LANE = "forklift_audio_lane"
    OPS_LANE = "forklift_ops_lane"

    def __init__(self, config: Optional[EdgeNodeConfig] = None, prefer_qdrant: bool = True):
        self.config = config or EdgeNodeConfig()
        self.store = create_vector_store(self.config.storage_dir, prefer_qdrant=prefer_qdrant)
        for lane_cfg in self.config.lanes.values():
            self.store.ensure_lane(lane_cfg.name, lane_cfg.dim)

    def enroll_few_shot_object(
        self,
        label: str,
        reference_vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Store 2-3 WES-pushed reference embeddings (ephemeral=False) and flush to disk."""
        meta = dict(metadata or {})
        ids: List[str] = []
        for vec in reference_vectors:
            pid = str(uuid.uuid4())
            payload = VisionPayload(
                source_type="few_shot_reference",
                label=label,
                zone_id=meta.pop("zone_id", "GLOBAL"),
                forklift_id=self.config.forklift_id,
                ephemeral=False,
                severity=meta.get("severity"),
            ).model_dump()
            payload.update(meta)
            self.store.upsert(self.VISION_LANE, pid, vec, payload)
            ids.append(pid)
        self.store.flush()
        return ids

    def ingest_vision_observation(
        self,
        vector: List[float],
        label: str,
        zone_id: str,
        pose_xyz: Optional[List[float]] = None,
        wes_task_id: Optional[str] = None,
        ephemeral: bool = True,
    ) -> str:
        pid = str(uuid.uuid4())
        payload = VisionPayload(
            source_type="live_observation",
            label=label,
            zone_id=zone_id,
            forklift_id=self.config.forklift_id,
            pose_xyz=pose_xyz or [0.0, 0.0, 0.0],
            ephemeral=ephemeral,
            wes_task_id=wes_task_id,
        ).model_dump()
        self.store.upsert(self.VISION_LANE, pid, vector, payload)
        return pid

    def ingest_audio_event(
        self,
        vector: List[float],
        transcript: str,
        zone_id: str,
        operator_id: str = "OP-UNKNOWN",
        intent_tag: str = "operator_note",
        mast_height_mm: Optional[int] = None,
        ephemeral: bool = True,
    ) -> str:
        pid = str(uuid.uuid4())
        payload = AudioPayload(
            transcript=transcript,
            intent_tag=intent_tag,
            operator_id=operator_id,
            zone_id=zone_id,
            mast_height_mm=mast_height_mm,
            ephemeral=ephemeral,
        ).model_dump()
        self.store.upsert(self.AUDIO_LANE, pid, vector, payload)
        return pid

    def ingest_ops_telemetry(
        self,
        vector: List[float],
        wes_task_id: str,
        zone_id: str,
        semantic_summary: str,
        action: str = "PUTAWAY",
        sku_id: Optional[str] = None,
        target_bin: Optional[str] = None,
        load_weight_kg: float = 0.0,
        mast_height_mm: int = 0,
        hydraulic_pressure_bar: float = 0.0,
        battery_soc_pct: float = 100.0,
        ephemeral: bool = True,
    ) -> str:
        pid = str(uuid.uuid4())
        payload = OpsTelemetryPayload(
            wes_task_id=wes_task_id,
            sku_id=sku_id,
            action=action,
            zone_id=zone_id,
            target_bin=target_bin,
            load_weight_kg=load_weight_kg,
            mast_height_mm=mast_height_mm,
            hydraulic_pressure_bar=hydraulic_pressure_bar,
            battery_soc_pct=battery_soc_pct,
            semantic_summary=semantic_summary,
            ephemeral=ephemeral,
        ).model_dump()
        self.store.upsert(self.OPS_LANE, pid, vector, payload)
        return pid

    def query_lane(
        self,
        lane_name: str,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        must_match: Optional[Dict[str, Any]] = None,
        apply_freshness: bool = False,
    ) -> List[SearchHit]:
        lane_cfg = self.config.lanes[lane_name]
        thresh = lane_cfg.threshold if score_threshold is None else score_threshold
        return self.store.query(
            lane=lane_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=thresh,
            must_match=must_match,
            freshness_half_life_sec=self.config.freshness_half_life_sec if apply_freshness else None,
            freshness_weight=self.config.freshness_weight if apply_freshness else 0.0,
        )

    def evaluate_vision_observation(
        self,
        frame_vector: List[float],
        zone_id: str,
        roi_jpeg_b64: str,
        pose_xyz: Optional[List[float]] = None,
        wes_task_id: Optional[str] = None,
    ) -> VisionDecision:
        """Apply calibrated vision thresholds: >=0.82 MATCHED, 0.60-0.81 UNKNOWN_OBJECT, <0.60 IGNORED."""
        vision_cfg = self.config.lanes[self.VISION_LANE]
        hits = self.query_lane(
            self.VISION_LANE,
            frame_vector,
            top_k=1,
            score_threshold=vision_cfg.unknown_floor,
            must_match={"source_type": "few_shot_reference"},
        )
        if not hits:
            return VisionDecision(status="IGNORED", label="background_noise", score=0.0)

        top = hits[0]
        candidate_label = str(top.payload.get("label", "unknown_object"))
        if top.score >= vision_cfg.threshold:
            pid = self.ingest_vision_observation(
                vector=frame_vector,
                label=candidate_label,
                zone_id=zone_id,
                pose_xyz=pose_xyz,
                wes_task_id=wes_task_id,
                ephemeral=True,
            )
            return VisionDecision(
                status="MATCHED",
                label=candidate_label,
                score=top.score,
                point_id=pid,
            )

        unknown_evt = UnknownObjectEvent(
            forklift_id=self.config.forklift_id,
            zone_id=zone_id,
            candidate_label=candidate_label,
            similarity_score=top.score,
            roi_jpeg_b64=roi_jpeg_b64,
            vector_512d=frame_vector,
        )
        return VisionDecision(
            status="UNKNOWN_OBJECT",
            label="unknown_object",
            score=top.score,
            unknown_event=unknown_evt,
        )

    def build_wes_incident_bundle(
        self,
        zone_id: str,
        vision_vec: List[float],
        audio_vec: List[float],
        ops_vec: List[float],
    ) -> EdgeSemanticIncident:
        """Correlate vision, operator voice, and CAN-bus task context across all 3 lanes."""
        vision_hits = self.query_lane(self.VISION_LANE, vision_vec, top_k=1)
        audio_hits = self.query_lane(
            self.AUDIO_LANE,
            audio_vec,
            top_k=1,
            must_match={"zone_id": zone_id},
            apply_freshness=True,
        )
        ops_hits = self.query_lane(
            self.OPS_LANE,
            ops_vec,
            top_k=1,
            must_match={"zone_id": zone_id},
            apply_freshness=True,
        )

        detected_label = (
            str(vision_hits[0].payload.get("label", "unknown_object"))
            if vision_hits
            else "unknown_object"
        )
        return EdgeSemanticIncident(
            forklift_id=self.config.forklift_id,
            zone_id=zone_id,
            vision_classification=detected_label,
            vision_confidence=vision_hits[0].score if vision_hits else 0.0,
            operator_note=audio_hits[0].payload.get("transcript") if audio_hits else None,
            audio_confidence=audio_hits[0].score if audio_hits else None,
            active_wes_task=ops_hits[0].payload.get("wes_task_id") if ops_hits else None,
            load_weight_kg=ops_hits[0].payload.get("load_weight_kg") if ops_hits else None,
        )

    def forget_zone_ephemeral_state(self, zone_id: str) -> Dict[str, int]:
        """Purge ephemeral observations for `zone_id` while preserving few-shot references."""
        deleted = {}
        for lane_name in (self.VISION_LANE, self.AUDIO_LANE, self.OPS_LANE):
            deleted[lane_name] = self.store.delete_where(
                lane_name,
                {"zone_id": zone_id, "ephemeral": True},
            )
        self.store.flush()
        return deleted

    def close(self) -> None:
        self.store.close()

