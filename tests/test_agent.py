import json
import os
import subprocess
from typing import Any

import numpy as np
import polars as pl
import pytest

from ds_copilot.agent import (
    PLAN_TOOL,
    Plan,
    ProposedCell,
    _build_user_prompt,
    plan,
)
from ds_copilot.profiler import profile


class _FakeToolUseBlock:
    type = "tool_use"
    name = "submit_plan"

    def __init__(self, input_dict: dict) -> None:
        self.input = input_dict


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(
        self,
        content: list,
        stop_reason: str = "tool_use",
        request_id: str = "req_test",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self._request_id = request_id


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeAnthropic:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


@pytest.fixture
def churn_profile():
    rng = np.random.default_rng(42)
    n = 1_000
    df = pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["m2m", "1y", "2y"], n),
            "churned": rng.choice([0, 1], n, p=[0.74, 0.26]),
        }
    )
    return profile(df, target="churned")


@pytest.fixture
def tiny_profile():
    return profile(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))


def _valid_plan_input() -> dict:
    return {
        "summary": "Walk through schema profile then train a baseline.",
        "cells": [
            {
                "title": "Drop ID",
                "rationale": "customer_id is unique-per-row and leaks no signal.",
                "code": "X = df.drop('customer_id')",
                "est_cost": "cheap",
                "depends_on": [],
                "warnings": [],
            },
            {
                "title": "Train baseline",
                "rationale": "Logistic regression sets a floor for AUC.",
                "code": "model.fit(X_train, y_train)",
                "est_cost": "medium",
                "depends_on": ["Drop ID"],
                "warnings": ["AUC may collapse to chance on synthetic data."],
            },
        ],
        "overall_warnings": [
            "Synthetic churn dataset has no real signal between features and target.",
        ],
    }


def test_plan_parses_tool_use_response(tiny_profile) -> None:
    fake = _FakeAnthropic(
        _FakeResponse(content=[_FakeToolUseBlock(_valid_plan_input())])
    )
    p = plan(
        goal="baseline churn classifier",
        profile=tiny_profile,
        backend="anthropic",
        client=fake,
    )

    assert isinstance(p, Plan)
    assert p.goal == "baseline churn classifier"
    assert len(p.cells) == 2
    assert isinstance(p.cells[0], ProposedCell)
    assert p.cells[1].depends_on == ["Drop ID"]
    assert p.cells[1].est_cost == "medium"
    assert "no real signal" in p.overall_warnings[0]


def test_plan_request_uses_correct_parameters(tiny_profile) -> None:
    fake = _FakeAnthropic(
        _FakeResponse(content=[_FakeToolUseBlock(_valid_plan_input())])
    )
    plan(goal="explore", profile=tiny_profile, backend="anthropic", client=fake)
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None

    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    # No removed-on-4.7 sampling params leaked into the request
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs

    # Forced tool use
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_plan"}
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "submit_plan"
    assert kwargs["tools"][0]["strict"] is True

    # System prompt is cacheable
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    # User prompt mentions the goal
    msgs = kwargs["messages"]
    assert msgs[0]["role"] == "user"
    assert "explore" in msgs[0]["content"]


def test_plan_raises_when_no_tool_use_block(tiny_profile) -> None:
    fake = _FakeAnthropic(
        _FakeResponse(
            content=[_FakeTextBlock("I cannot help with that.")],
            stop_reason="end_turn",
        )
    )
    with pytest.raises(RuntimeError, match="did not return a submit_plan"):
        plan(goal="x", profile=tiny_profile, backend="anthropic", client=fake)


def test_plan_raises_on_pydantic_validation_failure(tiny_profile) -> None:
    # Tool input missing required `cells` field
    bad_input = {"summary": "incomplete", "overall_warnings": []}
    fake = _FakeAnthropic(_FakeResponse(content=[_FakeToolUseBlock(bad_input)]))
    with pytest.raises(RuntimeError, match="failed pydantic validation"):
        plan(goal="x", profile=tiny_profile, backend="anthropic", client=fake)


def test_plan_anthropic_backend_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dotenv

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # python-dotenv walks up from the calling file, so chdir alone isn't
    # enough -- block load_dotenv() from re-populating the env from our .env.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        plan(
            goal="x",
            profile=profile(pl.DataFrame({"a": [1]})),
            backend="anthropic",
        )


def test_user_prompt_includes_grounded_schema(churn_profile) -> None:
    prompt = _build_user_prompt(
        goal="explore churn drivers",
        profile=churn_profile,
        existing_cells=["import polars as pl", "df = pl.DataFrame(...)"],
    )

    assert "explore churn drivers" in prompt
    assert "rows: 1,000" in prompt
    assert "target column: `churned`" in prompt
    assert "target positive rate" in prompt
    # All real columns named
    for col in ["customer_id", "tenure_months", "monthly_charges", "contract_type"]:
        assert col in prompt
    # Leakage hint surfaced to the planner
    assert "WARN" in prompt and "customer_id" in prompt
    # Existing cells included
    assert "import polars as pl" in prompt
    # Tail directive
    assert "submit_plan" in prompt


def test_user_prompt_works_without_existing_cells(tiny_profile) -> None:
    prompt = _build_user_prompt(
        goal="describe the data",
        profile=tiny_profile,
        existing_cells=[],
    )
    assert "Existing notebook cells" not in prompt
    assert "describe the data" in prompt


def test_claude_cli_backend_parses_structured_output(
    monkeypatch: pytest.MonkeyPatch, tiny_profile
) -> None:
    """The CLI returns JSON with `structured_output` keyed to our schema."""
    plan_payload = _valid_plan_input()
    cli_json = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Plan submitted.",
        "structured_output": plan_payload,
        "total_cost_usd": 0.0,
    }
    captured: dict[str, Any] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class _Completed:
            returncode = 0
            stdout = json.dumps(cli_json)
            stderr = ""

        return _Completed()

    import ds_copilot.agent as agent_module

    monkeypatch.setattr(agent_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        agent_module.shutil, "which", lambda name: "/fake/path/claude"
    )

    p = plan(
        goal="explore churn",
        profile=tiny_profile,
        backend="claude-cli",
    )

    assert p.summary == plan_payload["summary"]
    assert len(p.cells) == 2

    # Right CLI flags went out.
    args = captured["args"]
    assert "--print" in args
    assert "--output-format" in args
    assert args[args.index("--output-format") + 1] == "json"
    assert "--json-schema" in args
    assert "--model" in args
    assert args[args.index("--model") + 1] == "claude-opus-4-7"
    assert "--effort" in args
    assert args[args.index("--effort") + 1] == "high"
    assert "--no-session-persistence" in args
    # stdin redirected to DEVNULL so claude doesn't wait 3s for input
    assert captured["kwargs"].get("stdin") is subprocess.DEVNULL


def test_claude_cli_backend_propagates_error_payload(
    monkeypatch: pytest.MonkeyPatch, tiny_profile
) -> None:
    cli_json = {
        "type": "result",
        "is_error": True,
        "result": "Not logged in. Please run /login",
    }

    def fake_run(args, **kwargs):
        class _Completed:
            returncode = 0
            stdout = json.dumps(cli_json)
            stderr = ""

        return _Completed()

    import ds_copilot.agent as agent_module

    monkeypatch.setattr(agent_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        agent_module.shutil, "which", lambda name: "/fake/path/claude"
    )

    with pytest.raises(RuntimeError, match="Not logged in"):
        plan(goal="x", profile=tiny_profile, backend="claude-cli")


def test_claude_cli_backend_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tiny_profile
) -> None:
    cli_json = {
        "type": "result",
        "is_error": False,
        "result": "",
        "structured_output": {"summary": "x"},  # missing required `cells`
    }

    def fake_run(args, **kwargs):
        class _Completed:
            returncode = 0
            stdout = json.dumps(cli_json)
            stderr = ""

        return _Completed()

    import ds_copilot.agent as agent_module

    monkeypatch.setattr(agent_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        agent_module.shutil, "which", lambda name: "/fake/path/claude"
    )

    with pytest.raises(RuntimeError, match="failed validation"):
        plan(goal="x", profile=tiny_profile, backend="claude-cli")


def test_claude_cli_backend_missing_executable(
    monkeypatch: pytest.MonkeyPatch, tiny_profile
) -> None:
    import ds_copilot.agent as agent_module

    monkeypatch.setattr(agent_module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="claude CLI not found"):
        plan(goal="x", profile=tiny_profile, backend="claude-cli")


def test_plan_tool_schema_is_strict_compatible() -> None:
    schema = PLAN_TOOL["input_schema"]
    # Every property of the top-level object is in `required`
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False

    # And the same for each cell
    cell_schema = schema["properties"]["cells"]["items"]
    assert set(cell_schema["properties"]) == set(cell_schema["required"])
    assert cell_schema["additionalProperties"] is False
