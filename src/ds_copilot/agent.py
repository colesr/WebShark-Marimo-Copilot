"""Plan-first agent: produces a structured Plan of ProposedCell objects.

Calls Claude Opus 4.7 with forced tool use to convert a user goal plus
grounded context (schema profile, existing notebook cells) into a reviewable
plan. The agent's job is to PROPOSE, not to EXECUTE -- ds_copilot.ui renders
the plan with checkboxes so the practitioner approves cells before they hit
the notebook.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from ds_copilot.profiler import Profile

CostEstimate = Literal["cheap", "medium", "expensive"]
Backend = Literal["claude-cli", "anthropic"]

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_BACKEND: Backend = "claude-cli"


class ProposedCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    rationale: str
    code: str
    est_cost: CostEstimate
    depends_on: list[str]
    warnings: list[str]


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    goal: str
    summary: str
    cells: list[ProposedCell]
    overall_warnings: list[str]


SYSTEM_PROMPT = """\
You are a plan-first data-science copilot for Marimo reactive notebooks.
Your job is to PROPOSE, not to EXECUTE. Every cell you propose will be
reviewed and individually approved by a human before it runs.

Hard rules:
1. Ground every proposal in the dataframe schema you are given. Never invent
   column names. Use only columns that appear in the schema profile.
2. Propose discrete, runnable cells -- one logical step per cell.
3. For each cell, write a short rationale explaining WHY you propose it.
4. Mark cell cost honestly: "cheap" (<1s, simple ops), "medium" (1-30s,
   light model training, plots), "expensive" (>30s, heavy compute).
5. In `depends_on`, list the titles of earlier cells in THIS plan that the
   cell needs. Use [] when there are no plan-internal dependencies.
6. Surface concerns. If a column looks like leakage, if the target rate
   signals imbalance, if a transformation could lose information, say so in
   the cell's `warnings` or in `overall_warnings`. Be specific.
7. Reference symbols already defined in the notebook (`df`, `mo`, `pl`, `np`).
   Do NOT re-import them. Marimo enforces single-cell name definition --
   re-importing crashes the cell.
8. Use Polars (preferred). Display via the existing `mo` alias. For ML use
   scikit-learn (already installed). For plotting, use altair or matplotlib.

Submit your plan via the submit_plan tool. Be specific. Be honest about
cost. Be conservative about scope -- a small set of well-considered cells
beats a long list of sketchy ones."""


PLAN_TOOL: dict[str, Any] = {
    "name": "submit_plan",
    "description": (
        "Submit a structured multi-cell plan for the user to review and "
        "approve cell-by-cell."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "One-paragraph overview of what this plan accomplishes "
                    "and how the cells fit together."
                ),
            },
            "cells": {
                "type": "array",
                "description": (
                    "Ordered list of proposed cells. The user will toggle "
                    "each on/off before applying."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": (
                                "Short imperative label, e.g. 'Encode "
                                "categoricals'."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this cell is in the plan.",
                        },
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code for the cell body. Plain code, "
                                "no markdown fences."
                            ),
                        },
                        "est_cost": {
                            "type": "string",
                            "enum": ["cheap", "medium", "expensive"],
                            "description": "Compute cost class.",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Titles of earlier cells in THIS plan that "
                                "this cell needs. [] if none."
                            ),
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Specific concerns to surface (leakage, "
                                "scope, lost information). [] if none."
                            ),
                        },
                    },
                    "required": [
                        "title",
                        "rationale",
                        "code",
                        "est_cost",
                        "depends_on",
                        "warnings",
                    ],
                    "additionalProperties": False,
                },
            },
            "overall_warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Plan-level concerns. [] if none.",
            },
        },
        "required": ["summary", "cells", "overall_warnings"],
        "additionalProperties": False,
    },
}


def plan(
    goal: str,
    profile: Profile,
    existing_cells: list[str] | None = None,
    *,
    backend: Backend = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    client: anthropic.Anthropic | None = None,
    claude_executable: str | None = None,
    timeout_s: int = 300,
) -> Plan:
    """Ask Claude to propose a plan grounded in the provided schema profile.

    Args:
        goal: The user's natural-language goal, e.g. "explore churn drivers".
        profile: A `Profile` from `ds_copilot.profiler.profile()`.
        existing_cells: Optional one-line summaries of existing notebook cells
            (e.g. "import polars as pl"). Helps the planner avoid re-importing
            or re-defining names.
        backend: "claude-cli" (default) uses the local `claude` CLI in headless
            mode and bills against the user's Claude Code subscription.
            "anthropic" uses the Anthropic Python SDK and bills against
            ANTHROPIC_API_KEY.
        model: Claude model ID. Defaults to opus-4-7.
        effort: Effort level: "low", "medium", "high", "xhigh", "max".
        client: Optional pre-built Anthropic client (only used with backend
            "anthropic"). Defaults to a new client reading ANTHROPIC_API_KEY
            from env or .env.
        claude_executable: Optional path to the `claude` CLI (only used with
            backend "claude-cli"). Defaults to whatever is on PATH.
        timeout_s: Wall-clock timeout for the call.

    Raises:
        RuntimeError: backend setup failed, the call failed, or the returned
            plan failed pydantic validation.
    """
    user_prompt = _build_user_prompt(
        goal=goal, profile=profile, existing_cells=existing_cells or []
    )

    if backend == "anthropic":
        return _plan_via_anthropic(
            goal=goal,
            user_prompt=user_prompt,
            model=model,
            effort=effort,
            client=client,
        )
    if backend == "claude-cli":
        return _plan_via_claude_cli(
            goal=goal,
            user_prompt=user_prompt,
            model=model,
            effort=effort,
            claude_executable=claude_executable,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unknown backend: {backend!r}")


def _plan_via_anthropic(
    *,
    goal: str,
    user_prompt: str,
    model: str,
    effort: str,
    client: anthropic.Anthropic | None,
) -> Plan:
    if client is None:
        client = _build_anthropic_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=16_384,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[PLAN_TOOL],
            tool_choice={"type": "tool", "name": "submit_plan"},
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(f"Anthropic authentication failed: {e.message}") from e

    tool_block = next(
        (
            b
            for b in response.content
            if getattr(b, "type", None) == "tool_use"
            and getattr(b, "name", None) == "submit_plan"
        ),
        None,
    )
    if tool_block is None:
        block_types = [getattr(b, "type", "?") for b in response.content]
        raise RuntimeError(
            "Claude did not return a submit_plan tool call. "
            f"stop_reason={response.stop_reason!r}; "
            f"content blocks={block_types!r}; "
            f"request_id={getattr(response, '_request_id', None)!r}"
        )

    try:
        return Plan(goal=goal, **tool_block.input)
    except ValidationError as e:
        raise RuntimeError(f"Plan failed pydantic validation: {e}") from e


def _plan_via_claude_cli(
    *,
    goal: str,
    user_prompt: str,
    model: str,
    effort: str,
    claude_executable: str | None,
    timeout_s: int,
) -> Plan:
    """Invoke the Claude Code CLI in headless mode with structured output.

    Uses `claude --print --output-format json --json-schema ...`. The CLI
    bills against the user's Claude Code subscription (OAuth/keychain), so
    no ANTHROPIC_API_KEY is needed -- and account billing isn't either.
    """
    executable = claude_executable or shutil.which("claude")
    if not executable:
        raise RuntimeError(
            "claude CLI not found on PATH. Install Claude Code from "
            "https://claude.com/code or pass claude_executable=..."
        )

    args = [
        executable,
        "--print",
        "--output-format", "json",
        "--json-schema", json.dumps(PLAN_TOOL["input_schema"]),
        "--system-prompt", SYSTEM_PROMPT,
        "--model", model,
        "--effort", effort,
        "--no-session-persistence",
        user_prompt,
    ]

    # Strip ANTHROPIC_API_KEY from the child env. If it's set, the claude CLI
    # prefers API auth over OAuth/subscription auth -- and we're using this
    # backend specifically because we don't have API credits.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
            encoding="utf-8",
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"claude CLI timed out after {timeout_s}s. "
            "Try lower effort or a simpler prompt."
        ) from e

    if completed.returncode != 0:
        # Errors may land on either stream; surface both.
        err = (completed.stderr or "").strip()
        out = (completed.stdout or "").strip()
        detail = err or out or "(no output)"
        raise RuntimeError(
            f"claude CLI exited rc={completed.returncode}: {detail[:800]}"
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude CLI returned invalid JSON: "
            f"{completed.stdout[:500]}"
        ) from e

    if result.get("is_error"):
        raise RuntimeError(
            f"claude CLI returned an error: "
            f"{result.get('result', '<no message>')}"
        )

    structured = result.get("structured_output")
    if structured is None:
        raise RuntimeError(
            "claude CLI returned no structured_output. "
            f"result text was: {str(result.get('result', ''))[:500]}"
        )

    try:
        return Plan(goal=goal, **structured)
    except ValidationError as e:
        raise RuntimeError(f"Plan from claude CLI failed validation: {e}") from e


def _build_anthropic_client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
            "add your key, or pass a pre-built Anthropic client to plan()."
        )
    return anthropic.Anthropic()


def _build_user_prompt(
    *, goal: str, profile: Profile, existing_cells: list[str]
) -> str:
    """Compose the per-call user message grounded in the live schema."""
    lines: list[str] = [f"# Goal\n\n{goal}\n", "# Dataframe schema (live)"]
    lines.append(f"- rows: {profile.n_rows:,}")
    lines.append(f"- cols: {profile.n_cols}")
    if profile.target:
        lines.append(f"- target column: `{profile.target}`")
        if profile.target_rate is not None:
            lines.append(f"- target positive rate: {profile.target_rate:.3f}")
    if profile.sampled and profile.sampled_from is not None:
        lines.append(f"- profile sampled from {profile.sampled_from:,} rows")
    lines.append("")
    lines.append("Columns:")
    for c in profile.columns:
        bits = [
            f"`{c.name}` ({c.dtype})",
            f"null_rate={c.null_rate}",
            f"cardinality={c.cardinality}",
        ]
        if c.numeric_stats:
            bits.append(
                f"min={c.numeric_stats.min:.3g} "
                f"mean={c.numeric_stats.mean:.3g} "
                f"max={c.numeric_stats.max:.3g}"
            )
        if c.categorical_stats:
            top = ", ".join(
                f"{v}({n})" for v, n in c.categorical_stats.top_values[:3]
            )
            bits.append(f"top={top}")
        if c.leakage_hints:
            reasons = "; ".join(h.reason for h in c.leakage_hints)
            bits.append(f"WARN: {reasons}")
        lines.append(f"  - {' | '.join(bits)}")
    if profile.warnings:
        lines.append("")
        lines.append("Profile-level warnings:")
        for w in profile.warnings:
            lines.append(f"  - {w}")
    if existing_cells:
        lines.append("")
        lines.append("# Existing notebook cells (do not redefine these names)")
        for i, snippet in enumerate(existing_cells, 1):
            lines.append(f"{i}. {snippet}")
    lines.append("")
    lines.append("Propose a plan using the submit_plan tool.")
    return "\n".join(lines)
