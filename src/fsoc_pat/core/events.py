"""Event bus for decoupled subsystem communication.

Provides a simple publish-subscribe mechanism so subsystems can
emit events (e.g., beacon lost, state transition, target acquired)
without direct dependencies on each other.

Thread safety: This implementation is NOT thread-safe. If multi-thread
operation is needed in future phases, wrap publish/subscribe with a lock.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Predefined event types for the PAT system."""

    # Simulation lifecycle
    SIMULATION_STARTED = auto()
    SIMULATION_STEP = auto()
    SIMULATION_STOPPED = auto()

    # PAT state transitions
    PAT_STATE_CHANGED = auto()

    # Detection events
    BEACON_DETECTED = auto()
    BEACON_LOST = auto()
    BEACON_REACQUIRED = auto()

    # Control events
    CONTROLLER_SATURATED = auto()
    CONTROLLER_FALLBACK = auto()

    # Estimation events
    FILTER_DIVERGED = auto()
    OUTLIER_REJECTED = auto()

    # Link events
    LINK_STATE_CHANGED = auto()

    # Error events
    NUMERICAL_ERROR = auto()
    CONFIGURATION_ERROR = auto()


@dataclass(frozen=True, slots=True)
class Event:
    """An event emitted by a subsystem.

    Attributes:
        event_type: The type of event.
        timestamp: Simulation time when the event occurred (s).
        source: Name of the subsystem that emitted the event.
        data: Optional payload with event-specific data.
    """

    event_type: EventType
    timestamp: float
    source: str
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for event handler callbacks
EventHandler = Callable[[Event], None]


class EventBus:
    """Publish-subscribe event bus.

    Subsystems subscribe to event types they care about. When an event
    is published, all registered handlers for that type are called
    synchronously in registration order.

    Example:
        bus = EventBus()
        bus.subscribe(EventType.BEACON_LOST, my_handler)
        bus.publish(Event(EventType.BEACON_LOST, t, "detector"))
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._record_history: bool = True

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type.

        Args:
            event_type: The event type to listen for.
            handler: Callable that accepts an Event.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler for an event type.

        Args:
            event_type: The event type.
            handler: The handler to remove.
        """
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers.

        Handlers are called synchronously in registration order.
        Exceptions in handlers are logged but do not prevent other
        handlers from executing.

        Args:
            event: The event to publish.
        """
        if self._record_history:
            self._history.append(event)

        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event handler %s failed for event %s",
                    handler.__name__,
                    event.event_type.name,
                )

    @property
    def history(self) -> list[Event]:
        """Chronological list of all published events."""
        return self._history

    def clear_history(self) -> None:
        """Clear the event history."""
        self._history.clear()

    def clear_all(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._history.clear()
