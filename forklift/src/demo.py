"""End-to-end 4-phase lifecycle demo: Enroll -> Dead-Zone Capture -> Reconnect Sync -> Forget."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import EdgeNodeConfig
from src.memory.orchestrator import SiMaForkliftEdgeMemory
from src.pipeline.sensors import CabAudioClip, CameraFrame, CanBusSnapshot
from src.runtime import ForkliftEdgeNode
from src.sync.mqtt_bridge import InMemoryForkliftTransport, InMemoryMqttBus
from src.wes.simulator import JungheinrichWesSimulator


def run_demo(workspace_dir: Path = Path("./edge_data_demo")) -> None:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    bus = InMemoryMqttBus()
    wes = JungheinrichWesSimulator(bus)
    transport = InMemoryForkliftTransport(bus, online=True)

    cfg = EdgeNodeConfig(
        forklift_id="JH-ETV-216i-09",
        storage_dir=workspace_dir / "qdrant_edge",
        outbox_db_path=workspace_dir / "outbox.db",
    )
    node = ForkliftEdgeNode(config=cfg, transport=transport)

    try:
        print("=== PHASE 1: Few-Shot SKU / Hazard Enrollment (WES -> Forklift) ===")
        ref_vecs = [
            node.vision_encoder.encode_label_or_frame("damaged euro pallet"),
            node.vision_encoder.encode_label_or_frame("damaged_euro_pallet broken wood"),
        ]
        wes.dispatch_few_shot_sku(
            label="damaged_euro_pallet",
            reference_vectors=ref_vecs,
            metadata={"severity": "high"},
        )
        print(
            "Vision lane points after enrollment:",
            node.memory.store.count(SiMaForkliftEdgeMemory.VISION_LANE),
        )

        print("\n=== PHASE 2: Offline Aisle Operation (Wi-Fi Dead Zone) ===")
        transport.set_online(False)
        node.process_can_snapshot(
            CanBusSnapshot(
                wes_task_id="TSK-88412",
                zone_id="AISLE-04",
                action="PUTAWAY",
                sku_id="SKU-99201-CHEM",
                target_bin="AISLE-04-BAY-14-LVL-03",
                load_weight_kg=840.0,
                mast_height_mm=4250,
                battery_soc_pct=68.0,
            )
        )
        node.process_audio_clip(
            CabAudioClip(
                zone_id="AISLE-04",
                operator_id="OP-402",
                raw_audio_or_transcript="Broken pallet blocking bay 12 in aisle four",
                mast_height_mm=4250,
                intent_tag="maintenance_anomaly",
            )
        )
        res = node.process_camera_frame(
            CameraFrame(
                zone_id="AISLE-04",
                pose_xyz=[14.2, 8.5, 0.0],
                visual_descriptor="damaged euro pallet",
            )
        )
        print(f"Camera decision: {res.decision.status} ({res.decision.label}, score={res.decision.score})")
        print(f"Sync status while offline: {res.sync_status}, outbox pending={node.outbox.pending_count()}")
        print(f"WES incidents received while offline: {len(wes.received_incidents)}")

        print("\n=== PHASE 3: Wi-Fi Restored -> Store-and-Forward Drain ===")
        transport.set_online(True)
        drained = node.reconnect_and_sync()
        print(f"Drained {drained} outbox message(s), outbox pending={node.outbox.pending_count()}")
        print("WES Digital Twin aisle_status:", wes.aisle_status)
        print("WES received incident:", wes.received_incidents[-1])

        print("\n=== PHASE 4: Dynamic State Pruning ('Forget' Trigger) ===")
        wes.broadcast_forget_zone("AISLE-04")
        print("WES Digital Twin aisle_status after clear:", wes.aisle_status)
        print(
            "Vision lane points remaining (permanent few-shot refs preserved):",
            node.memory.store.count(SiMaForkliftEdgeMemory.VISION_LANE),
        )
    finally:
        node.close()


if __name__ == "__main__":
    run_demo()

