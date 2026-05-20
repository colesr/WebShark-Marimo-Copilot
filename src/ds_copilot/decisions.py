"""Append-only decision log for plan-then-approve provenance.

Every planner_widget interaction emits three event types:

  plan_requested  -- goal, target, profile shape, backend/model/effort.
  plan_returned   -- summary, per-cell titles + costs + warnings, audit.
  cells_applied   -- which proposed cells the user accepted vs rejected.

Events are appended one-per-line to `.ds_copilot/decisions.jsonl` (relative
to CWD by default). The file is the auditability layer the wedge promises:
a practitioner reviewing a notebook can replay every decision the agent
proposed and the human accepted, including the overrides.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


EventType = Literal["plan_requested", "plan_returned", "cells_applied"]

DEFAULT_LOG_PATH = ".ds_copilot/decisions.jsonl"


class DecisionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    session_id: str
    event_type: EventType
    payload: dict[str, Any]


class DecisionLog:
    """Append-only JSONL persistence for DecisionEvents.

    Atomic-enough for our use case: each `record()` opens the file in
    append mode, writes one line, and closes. No locking; two concurrent
    notebooks writing the same file would interleave events but each line
    would still be a valid JSON record.
    """

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH) -> None:
        self.path = Path(path)

    def record(
        self,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> DecisionEvent:
        event = DecisionEvent(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            event_type=event_type,
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def history(
        self,
        *,
        session_id: str | None = None,
        event_type: EventType | None = None,
    ) -> list[DecisionEvent]:
        """Read the log back. Filter by session and/or event type."""
        if not self.path.exists():
            return []
        events: list[DecisionEvent] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = DecisionEvent.model_validate_json(stripped)
                except Exception:
                    # Skip corrupt lines rather than failing the whole read.
                    continue
                if session_id is not None and event.session_id != session_id:
                    continue
                if event_type is not None and event.event_type != event_type:
                    continue
                events.append(event)
        return events


def history(
    *,
    path: str | Path = DEFAULT_LOG_PATH,
    session_id: str | None = None,
    event_type: EventType | None = None,
) -> list[DecisionEvent]:
    """Convenience top-level reader: `from ds_copilot import history`."""
    return DecisionLog(path).history(session_id=session_id, event_type=event_type)
