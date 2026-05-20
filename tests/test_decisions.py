from datetime import datetime
from pathlib import Path

import pytest

from ds_copilot.decisions import DecisionEvent, DecisionLog, history


def test_record_creates_parent_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "subdir" / "decisions.jsonl"
    assert not log_path.parent.exists()
    log = DecisionLog(log_path)
    log.record("session-1", "plan_requested", {"goal": "x"})
    assert log_path.exists()
    assert log_path.parent.is_dir()


def test_record_appends_jsonl_per_event(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.record("session-1", "plan_requested", {"goal": "x"})
    log.record("session-1", "plan_returned", {"summary": "ok", "cells": []})
    log.record("session-1", "cells_applied", {"applied_titles": ["A"]})

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_history_reads_back_events(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.record("session-1", "plan_requested", {"goal": "explore"})
    log.record("session-1", "plan_returned", {"summary": "two-cell plan"})

    events = log.history()
    assert len(events) == 2
    assert all(isinstance(e, DecisionEvent) for e in events)
    assert events[0].event_type == "plan_requested"
    assert events[0].payload["goal"] == "explore"
    assert events[1].payload["summary"] == "two-cell plan"
    assert isinstance(events[0].timestamp, datetime)


def test_history_filters_by_session(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.record("session-A", "plan_requested", {"goal": "1"})
    log.record("session-B", "plan_requested", {"goal": "2"})
    log.record("session-A", "cells_applied", {"applied_titles": []})

    a_events = log.history(session_id="session-A")
    b_events = log.history(session_id="session-B")
    assert [e.event_type for e in a_events] == ["plan_requested", "cells_applied"]
    assert [e.event_type for e in b_events] == ["plan_requested"]


def test_history_filters_by_event_type(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.record("s", "plan_requested", {})
    log.record("s", "plan_returned", {})
    log.record("s", "cells_applied", {})

    applied = log.history(event_type="cells_applied")
    assert len(applied) == 1
    assert applied[0].event_type == "cells_applied"


def test_history_returns_empty_when_file_missing(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "does_not_exist.jsonl")
    assert log.history() == []


def test_history_skips_corrupt_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.record("s", "plan_requested", {"goal": "x"})
    # Inject a corrupt line between two valid ones
    with log_path.open("a", encoding="utf-8") as f:
        f.write("not-json-not-json\n")
    log.record("s", "plan_returned", {"summary": "y"})

    events = log.history()
    assert [e.event_type for e in events] == ["plan_requested", "plan_returned"]


def test_module_level_history_helper(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    DecisionLog(log_path).record("s", "plan_requested", {"goal": "x"})
    events = history(path=log_path)
    assert len(events) == 1
    assert events[0].payload["goal"] == "x"


def test_event_payload_round_trip(tmp_path: Path) -> None:
    """Complex payloads (nested dicts, lists) survive write/read."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    payload = {
        "goal": "explore",
        "profile": {"n_rows": 1000, "columns": ["a", "b", "c"]},
        "flagged": ["a"],
        "nested": {"deep": {"value": 42}},
    }
    log.record("s", "plan_requested", payload)

    events = log.history()
    assert events[0].payload == payload


def test_jsonl_format_is_one_line_per_event(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    # Record a payload that contains a newline in a string value.
    log.record("s", "plan_returned", {"summary": "line 1\nline 2"})
    # Even with the embedded newline, the file should have exactly one line.
    raw = log_path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
