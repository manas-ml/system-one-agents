"""Audit trail: every Jev decision, tool call, question to the customer and gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from jevkit import DecisionRecord

EventKind = Literal[
    "customer", "agent", "decision", "tool_call", "tool_result", "tool_error",
    "gate", "approval", "note", "outcome",
]


@dataclass(frozen=True)
class Event:
    kind: EventKind
    data: dict[str, Any]


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)
    listeners: list[Callable[[Event], None]] = field(default_factory=list)

    def emit(self, kind: EventKind, /, **data: Any) -> None:
        event = Event(kind, data)
        self.events.append(event)
        for listener in self.listeners:
            listener(event)

    def on_decision(self, record: DecisionRecord) -> None:
        self.emit("decision", **asdict(record))

    def decisions(self) -> list[dict[str, Any]]:
        return [e.data for e in self.events if e.kind == "decision"]

    def tool_calls(self) -> list[str]:
        return [e.data["tool"] for e in self.events if e.kind == "tool_call"]

    def to_json(self) -> str:
        return json.dumps([{"kind": e.kind, **e.data} for e in self.events], indent=2,
                          default=str)
