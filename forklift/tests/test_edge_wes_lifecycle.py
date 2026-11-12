"""Integration & unit tests for the SiMa.ai + Jungheinrich WES Edge Memory prototype."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import EdgeNodeConfig
from src.memory.orchestrator import SiMaForkliftEdgeMemory
from src.pipeline.sensors import CabAudioClip, CameraFrame, CanBusSnapshot
from src.runtime import ForkliftEdgeNode
from src.sync.mqtt_bridge import InMemoryForkliftTransport, InMemoryMqttBus
from src.wes.simulator import JungheinrichWesSimulator


def _make_env(tmp_path: Path):
    bus = InMemoryMqttBus()
    wes = JungheinrichWesSimulator(bus)
    transport = InMemoryForkliftTransport(bus, online=True)
    cfg = EdgeNodeConfig(
        forklift_id="JH-ETV-216i-09",
        storage_dir=tmp_path / "qdrant_edge",
        outbox_db_path=tmp_path / "outbox.db",
    )
    node = ForkliftEdgeNode(config=cfg, transport=transport, prefer_qdrant=False)
    return bus, wes, transport, cfg, node


def test_full_four_phase_lifecycle(tmp_path: Path) -> None:
    bus, wes, transport, cfg, node = _make_env(tmp_path)
    try:
        # Phase 1: Few-shot enrollment from WES
        ref_vecs = [
            node.vision_encoder.encode_label_or_frame("damaged euro pallet"),
            node.vision_encoder.encode_label_or_frame("damaged_euro_pallet broken slat"),
        ]
        wes.dispatch_few_shot_sku("damaged_euro_pallet", ref_vecs, {"severity": "high"})
        assert node.memory.store.count(SiMaForkliftEdgeMemory.VISION_LANE) == 2

        # Phase 2: Enter Wi-Fi dead zone
        transport.set_online(False)
        node.process_can_snapshot(
            CanBusSnapshot(
                wes_task_id="TSK-88412",
                zone_id="AISLE-04",
                action="PUTAWAY",
                sku_id="SKU-99201-CHEM",
                load_weight_kg=840.0,
                mast_height_mm=4250,
                battery_soc_pct=68.0,
            )
        )
        node.process_audio_clip(
            CabAudioClip(
                zone_id="AISLE-04",
                operator_id="OP-402",
                raw_audio_or_transcript="Broken pallet blocking bay 12",
                mast_height_mm=4250,
            )
        )
        res = node.process_camera_frame(
            CameraFrame(
                zone_id="AISLE-04",
                pose_xyz=[14.2, 8.5, 0.0],
                visual_descriptor="damaged euro pallet",
            )
        )
        assert res.decision.status == "MATCHED"
        assert res.decision.label == "damaged_euro_pallet"
        assert res.decision.score >= 0.82
        assert res.sync_status == "QUEUED"
        assert node.outbox.pending_count() == 1
        assert len(wes.received_incidents) == 0

        # Phase 3: Reconnect & drain SQLite outbox
        transport.set_online(True)
        drained = node.reconnect_and_sync()
        assert drained == 1
        assert node.outbox.pending_count() == 0
        assert len(wes.received_incidents) == 1
        incident = wes.received_incidents[0]
        assert incident["zone_id"] == "AISLE-04"
        assert incident["active_wes_task"] == "TSK-88412"
        assert incident["load_weight_kg"] == 840.0
        assert "Broken pallet" in (incident["operator_note"] or "")
        assert wes.aisle_status["AISLE-04"] == "OBSTRUCTED:damaged_euro_pallet"

        # Phase 4: WES broadcasts forget_zone -> ephemeral observations purged, few-shot preserved
        wes.broadcast_forget_zone("AISLE-04")
        assert wes.aisle_status["AISLE-04"] == "CLEAR"
        assert node.memory.store.count(SiMaForkliftEdgeMemory.VISION_LANE) == 2
        assert node.memory.store.count(SiMaForkliftEdgeMemory.AUDIO_LANE) == 0
        assert node.memory.store.count(SiMaForkliftEdgeMemory.OPS_LANE) == 0
    finally:
        node.close()


def test_unknown_object_band_and_noise_rejection(tmp_path: Path) -> None:
    _, wes, _, _, node = _make_env(tmp_path)
    try:
        # Enroll a synthetic unit vector along axis 0 for exact cosine control
        ref = [0.0] * 512
        ref[0] = 1.0
        wes.dispatch_few_shot_sku("hazard_cone", [ref], {"severity": "medium"})

        # Construct a vector with exact cosine similarity ~0.70 (within 0.60..0.81 band)
        mid_vec = [0.0] * 512
        mid_vec[0] = 0.70
        mid_vec[1] = (1.0 - 0.70**2) ** 0.5
        decision_mid = node.memory.evaluate_vision_observation(
            frame_vector=mid_vec,
            zone_id="AISLE-02",
            roi_jpeg_b64="RkFLRV9KUEVH",
        )
        assert decision_mid.status == "UNKNOWN_OBJECT"
        assert decision_mid.unknown_event is not None
        assert 0.60 <= decision_mid.score < 0.82

        # Construct orthogonal noise vector (cosine similarity = 0.0 < 0.60)
        noise_vec = [0.0] * 512
        noise_vec[2] = 1.0
        decision_noise = node.memory.evaluate_vision_observation(
            frame_vector=noise_vec,
            zone_id="AISLE-02",
            roi_jpeg_b64="RkFLRV9KUEVH",
        )
        assert decision_noise.status == "IGNORED"
    finally:
        node.close()


def test_persistence_across_power_cycle_and_freshness_ranking(tmp_path: Path) -> None:
    bus, wes, transport, cfg, node = _make_env(tmp_path)
    ref_vec = node.vision_encoder.encode_label_or_frame("cold_storage_drum")
    wes.dispatch_few_shot_sku("cold_storage_drum", [ref_vec])
    node.close()

    # Simulate power cycle / battery swap by creating a fresh node on the same storage_dir
    node2 = ForkliftEdgeNode(config=cfg, transport=transport, prefer_qdrant=False)
    try:
        assert node2.memory.store.count(SiMaForkliftEdgeMemory.VISION_LANE) == 1

        # Insert an older and a newer identical-vector audio note in AISLE-07
        vec = node2.audio_encoder.encode_transcript("Hydraulic valve check")
        old_id = node2.memory.ingest_audio_event(
            vector=vec,
            transcript="Older hydraulic note",
            zone_id="AISLE-07",
        )
        # Backdate the first point's timestamp_epoch in SQLite
        row = node2.memory.store._conn.execute(  # type: ignore[attr-defined]
            "SELECT payload_json FROM lane_points WHERE point_id = ?", (old_id,)
        ).fetchone()
        import json

        payload = json.loads(row[0])
        payload["timestamp_epoch"] = time.time() - 3600.0
        node2.memory.store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE lane_points SET payload_json = ? WHERE point_id = ?",
            (json.dumps(payload), old_id),
        )
        node2.memory.store._conn.commit()  # type: ignore[attr-defined]

        node2.memory.ingest_audio_event(
            vector=vec,
            transcript="Newer hydraulic note",
            zone_id="AISLE-07",
        )
        hits = node2.memory.query_lane(
            SiMaForkliftEdgeMemory.AUDIO_LANE,
            vec,
            top_k=1,
            must_match={"zone_id": "AISLE-07"},
            apply_freshness=True,
        )
        assert hits[0].payload["transcript"] == "Newer hydraulic note"
    finally:
        node2.close()

