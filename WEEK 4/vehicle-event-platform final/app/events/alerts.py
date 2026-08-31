"""
app/events/alerts.py — Alert engine with pluggable channels.

Channels:
  - InAppWebSocketChannel: broadcasts to connected dashboard clients
  - SimulatedWebhookChannel: logs what would be sent to webhook URL
  - SimulatedEmailChannel: logs what would be sent via email

If external credentials aren't available, channels simulate the notification
(log the payload that *would* have been sent) rather than skipping the feature.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from app.events.event_manager import Event

logger = logging.getLogger(__name__)


def _build_alert_payload(event: Event) -> Dict[str, Any]:
    """Standard alert payload."""
    return {
        "alert_id": event.event_id,
        "event_type": event.event_type.value,
        "severity": event.severity.value,
        "source_id": event.source_id,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "track_id": event.track_id,
        "class_name": event.class_name,
        "description": event.description,
        "timestamp": event.created_at,
        "evidence_path": event.evidence_path,
        "metadata": event.metadata,
    }


class AlertChannel(ABC):
    """Base class for alert delivery channels."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @abstractmethod
    def send(self, event: Event) -> None:
        """Send an alert for the given event."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


class InAppWebSocketChannel(AlertChannel):
    """
    Broadcasts alerts to all connected WebSocket dashboard clients.

    Requires a broadcast callable (provided by the WebSocket connection manager).
    """

    def __init__(self, broadcast_fn: Optional[Callable] = None, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._broadcast = broadcast_fn

    def set_broadcast(self, fn: Callable) -> None:
        self._broadcast = fn

    def send(self, event: Event) -> None:
        if not self.enabled:
            return
        payload = _build_alert_payload(event)
        payload["channel"] = "in_app"

        if self._broadcast:
            try:
                # Broadcast is async — schedule via asyncio if a loop is running
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._broadcast(payload))
                except RuntimeError:
                    # No running loop — run directly
                    asyncio.run(self._broadcast(payload))
            except Exception as e:
                logger.warning(f"InAppWebSocketChannel.send error: {e}")
        else:
            logger.debug(f"[IN-APP ALERT] {json.dumps(payload, default=str)}")


class SimulatedWebhookChannel(AlertChannel):
    """
    Simulates posting an alert to a webhook URL.

    In a real deployment, this would make an HTTP POST request.
    Without credentials, logs exactly what *would* be sent.
    """

    def __init__(self, endpoint: str = "", method: str = "POST", enabled: bool = True):
        super().__init__(enabled=enabled)
        self._endpoint = endpoint
        self._method = method

    def send(self, event: Event) -> None:
        if not self.enabled:
            return

        payload = _build_alert_payload(event)
        logger.info(
            f"[SIMULATED WEBHOOK] {self._method} {self._endpoint}\n"
            f"Payload: {json.dumps(payload, indent=2, default=str)}"
        )

        # If a real endpoint were configured, we'd do:
        # import httpx
        # httpx.post(self._endpoint, json=payload, timeout=5)


class SimulatedEmailChannel(AlertChannel):
    """
    Simulates sending an email alert.
    Logs what *would* be emailed rather than sending via SMTP.
    """

    def __init__(self, recipient: str = "", enabled: bool = True):
        super().__init__(enabled=enabled)
        self._recipient = recipient

    def send(self, event: Event) -> None:
        if not self.enabled:
            return

        payload = _build_alert_payload(event)
        logger.info(
            f"[SIMULATED EMAIL] To: {self._recipient}\n"
            f"Subject: [{event.severity.value}] {event.event_type.value} — {event.zone_name}\n"
            f"Body: {event.description}\n"
            f"Evidence: {event.evidence_path or 'N/A'}\n"
            f"Full payload: {json.dumps(payload, indent=2, default=str)}"
        )


class AlertEngine:
    """
    Routes events to all configured alert channels.

    Adds channels at initialization; each channel independently decides
    whether to send based on its enabled flag.
    """

    def __init__(self, channels: Optional[List[AlertChannel]] = None):
        self._channels: List[AlertChannel] = channels or []
        logger.info(
            f"AlertEngine initialized with {len(self._channels)} channels: "
            f"{[c.name() for c in self._channels]}"
        )

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)

    def send_alert(self, event: Event) -> None:
        """Send alert to all enabled channels."""
        for channel in self._channels:
            if not channel.enabled:
                continue
            try:
                channel.send(event)
            except Exception as e:
                logger.error(f"Alert channel {channel.name()} failed: {e}")

    def set_websocket_broadcast(self, fn: Callable) -> None:
        """Inject the WebSocket broadcast function after initialization."""
        for channel in self._channels:
            if isinstance(channel, InAppWebSocketChannel):
                channel.set_broadcast(fn)


def build_alert_engine(
    channel_configs: List[Any],
    ws_broadcast_fn: Optional[Callable] = None,
) -> AlertEngine:
    """Build AlertEngine from config channel list."""
    channels: List[AlertChannel] = []

    for cfg in channel_configs:
        channel_type = getattr(cfg, "type", "")
        enabled = getattr(cfg, "enabled", True)

        if channel_type == "in_app_websocket":
            channels.append(InAppWebSocketChannel(
                broadcast_fn=ws_broadcast_fn,
                enabled=enabled,
            ))
        elif channel_type == "webhook_simulated":
            channels.append(SimulatedWebhookChannel(
                endpoint=getattr(cfg, "endpoint", ""),
                method=getattr(cfg, "method", "POST"),
                enabled=enabled,
            ))
        elif channel_type == "email_simulated":
            channels.append(SimulatedEmailChannel(
                recipient=getattr(cfg, "recipient", ""),
                enabled=enabled,
            ))
        else:
            logger.warning(f"Unknown alert channel type: '{channel_type}' — skipping")

    return AlertEngine(channels=channels)
