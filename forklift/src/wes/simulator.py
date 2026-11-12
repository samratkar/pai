"""Jungheinrich WES Simulator: Digital Twin, Few-Shot Dispatcher, and Ephemeral Reaper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models import EnrollSkuCommand, ForgetZoneCommand
from src.sync.mqtt_bridge import InMemoryMqttBus


class JungheinrichWesSimulator:
    """Central Fleet Orchestration simulator connected to the MQTT bus."""

    def __init__(self, bus: InMemoryMqttBus):
        self.bus = bus
        self.aisle_status: Dict[str, str] = {}
        self.received_incidents: List[Dict[str, Any]] = []
        self.unknown_object_queue: List[Dict[str, Any]] = []

        self.bus.subscribe("jh/fleet/+/edge/incident", self._on_incident)
        self.bus.subscribe("jh/fleet/+/edge/unknown_object", self._on_unknown_object)

    def _on_incident(self, _topic: str, payload: Dict[str, Any]) -> None:
        self.received_incidents.append(payload)
        zone_id = payload.get("zone_id")
        if zone_id:
            self.aisle_status[zone_id] = f"OBSTRUCTED:{payload.get('vision_classification')}"

    def _on_unknown_object(self, _topic: str, payload: Dict[str, Any]) -> None:
        self.unknown_object_queue.append(payload)

    def dispatch_few_shot_sku(
        self,
        label: str,
        reference_vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Push 2-3 reference vectors to all forklifts for zero-retrain edge recognition."""
        cmd = EnrollSkuCommand(
            label=label,
            reference_vectors=reference_vectors,
            metadata=metadata or {},
        )
        self.bus.dispatch("jh/fleet/broadcast/edge/enroll_sku", cmd.model_dump())

    def broadcast_forget_zone(self, zone_id: str) -> None:
        """Clear Digital Twin zone state and broadcast forget trigger to fleet."""
        self.aisle_status[zone_id] = "CLEAR"
        cmd = ForgetZoneCommand(zone_id=zone_id)
        self.bus.dispatch("jh/fleet/broadcast/edge/forget_zone", cmd.model_dump())

