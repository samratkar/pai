"""Async MQTT over TLS bridge with InMemoryMqttBus for local simulation."""

from __future__ import annotations

import json
import ssl
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.memory.orchestrator import SiMaForkliftEdgeMemory
from src.models import EnrollSkuCommand, ForgetZoneCommand
from src.sync.outbox import SQLiteOutbox


MessageHandler = Callable[[str, Dict[str, Any]], None]


class MqttTransport(Protocol):
    @property
    def is_connected(self) -> bool: ...
    def publish(self, topic: str, payload: Dict[str, Any]) -> bool: ...
    def subscribe(self, topic_filter: str, handler: MessageHandler) -> None: ...


def _topic_matches(topic_filter: str, topic: str) -> bool:
    f_parts = topic_filter.split("/")
    t_parts = topic.split("/")
    for i, f in enumerate(f_parts):
        if f == "#":
            return True
        if i >= len(t_parts):
            return False
        if f != "+" and f != t_parts[i]:
            return False
    return len(f_parts) == len(t_parts)


class InMemoryMqttBus:
    """In-process MQTT pub/sub bus supporting simulated Wi-Fi dead-zone dropouts."""

    def __init__(self):
        self._subscriptions: List[Tuple[str, MessageHandler]] = []

    def subscribe(self, topic_filter: str, handler: MessageHandler) -> None:
        self._subscriptions.append((topic_filter, handler))

    def dispatch(self, topic: str, payload: Dict[str, Any]) -> None:
        for topic_filter, handler in list(self._subscriptions):
            if _topic_matches(topic_filter, topic):
                handler(topic, payload)


class InMemoryForkliftTransport:
    """Per-forklift transport attached to InMemoryMqttBus with online/offline toggle."""

    def __init__(self, bus: InMemoryMqttBus, online: bool = True):
        self.bus = bus
        self._online = online

    @property
    def is_connected(self) -> bool:
        return self._online

    def set_online(self, online: bool) -> None:
        self._online = online

    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        if not self._online:
            return False
        self.bus.dispatch(topic, payload)
        return True

    def subscribe(self, topic_filter: str, handler: MessageHandler) -> None:
        def _guarded(topic: str, payload: Dict[str, Any]) -> None:
            if self._online:
                handler(topic, payload)

        self.bus.subscribe(topic_filter, _guarded)


class PahoMqttTransport:
    """Production paho-mqtt v2 client over TLS for Jungheinrich WES broker."""

    def __init__(
        self,
        client_id: str,
        host: str,
        port: int = 8883,
        use_tls: bool = True,
        ca_certs: Optional[str] = None,
    ):
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        if use_tls:
            self._client.tls_set(ca_certs=ca_certs, cert_reqs=ssl.CERT_REQUIRED)
        self._host = host
        self._port = port
        self._connected = False
        self._handlers: List[Tuple[str, MessageHandler]] = []

        def _on_connect(client, userdata, flags, reason_code, properties):
            self._connected = reason_code == 0
            if self._connected:
                for topic_filter, _ in self._handlers:
                    client.subscribe(topic_filter, qos=1)

        def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
            self._connected = False

        def _on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return
            for topic_filter, handler in self._handlers:
                if _topic_matches(topic_filter, msg.topic):
                    handler(msg.topic, payload)

        self._client.on_connect = _on_connect
        self._client.on_disconnect = _on_disconnect
        self._client.on_message = _on_message

    def connect(self) -> None:
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        info = self._client.publish(topic, json.dumps(payload), qos=1)
        return info.rc == 0

    def subscribe(self, topic_filter: str, handler: MessageHandler) -> None:
        self._handlers.append((topic_filter, handler))
        if self._connected:
            self._client.subscribe(topic_filter, qos=1)


class EdgeMqttBridge:
    """Connects SiMaForkliftEdgeMemory + SQLiteOutbox to the WES MQTT bus."""

    def __init__(
        self,
        edge_memory: SiMaForkliftEdgeMemory,
        outbox: SQLiteOutbox,
        transport: MqttTransport,
    ):
        self.edge_memory = edge_memory
        self.outbox = outbox
        self.transport = transport
        cfg = edge_memory.config
        self.transport.subscribe(cfg.topic_enroll_sku, self._handle_enroll_sku)
        self.transport.subscribe(cfg.topic_forget_zone, self._handle_forget_zone)

    def _handle_enroll_sku(self, _topic: str, payload: Dict[str, Any]) -> None:
        cmd = EnrollSkuCommand(**payload)
        self.edge_memory.enroll_few_shot_object(
            label=cmd.label,
            reference_vectors=cmd.reference_vectors,
            metadata=cmd.metadata,
        )

    def _handle_forget_zone(self, _topic: str, payload: Dict[str, Any]) -> None:
        cmd = ForgetZoneCommand(**payload)
        self.edge_memory.forget_zone_ephemeral_state(cmd.zone_id)

    def publish_or_queue(self, topic: str, payload: Dict[str, Any]) -> str:
        """Send immediately if online (after draining outbox), else queue in SQLiteOutbox."""
        if self.transport.is_connected:
            self.outbox.drain(self.transport.publish)
            if self.outbox.pending_count() == 0 and self.transport.publish(topic, payload):
                return "SENT"
        self.outbox.enqueue(topic, payload)
        return "QUEUED"

    def sync_outbox(self) -> int:
        """Drain pending outbox messages when connectivity is restored."""
        if not self.transport.is_connected:
            return 0
        return self.outbox.drain(self.transport.publish)

