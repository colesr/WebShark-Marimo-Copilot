"""Marimo UI for the plan-then-approve agent.

`planner_widget(df, goal, target=None)` builds a schema profile, calls
ds_copilot.agent.plan(), and renders the resulting plan as an interactive
widget with per-cell checkboxes and a submit form. Approved cells are
materialized in the live notebook via marimo._code_mode.

Typical usage in a Marimo notebook is two cells:

    # cell 1: build, display
    from ds_copilot.ui import planner_widget
    planner = planner_widget(df, goal="explore churn", target="churned")
    planner

    # cell 2: re-runs when the form is submitted; creates the chosen cells
    result = await planner.apply()
    result
"""

from __future__ import annotations

from typing import Any

from ds_copilot.agent import Plan, plan
from ds_copilot.profiler import Profile, profile


COST_BADGE = {"cheap": "$", "medium": "$$", "expensive": "$$$"}


class PlannerWidget:
    """Renderable, applicable plan-then-approve UI for a Marimo notebook.

    Display the instance directly in a cell (Marimo invokes `_mime_` /
    `_display_` to render). Submit the form, then call `await widget.apply()`
    from a downstream cell to materialize the checked cells in the notebook.
    """

    def __init__(
        self,
        plan_obj: Plan,
        profile_obj: Profile,
        *,
        hide_code_by_default: bool = True,
    ) -> None:
        import marimo as mo

        self.plan = plan_obj
        self.profile = profile_obj
        self._hide_code = hide_code_by_default

        self._checkboxes = mo.ui.array(
            [
                mo.ui.checkbox(value=True, label="approve")
                for _ in plan_obj.cells
            ]
        )
        self._form = mo.ui.form(
            self._checkboxes,
            submit_button_label="Apply selected cells",
            bordered=False,
        )
        self._rendered = self._build_render()

    @property
    def form(self):
        return self._form

    @property
    def rendered(self):
        return self._rendered

    def _build_render(self):
        import marimo as mo

        elements: list[Any] = []
        elements.append(mo.md(f"## Plan: *{self.plan.goal}*"))
        elements.append(mo.md(f"**Summary.** {self.plan.summary}"))

        if self.plan.overall_warnings:
            ow = "\n".join(f"- {w}" for w in self.plan.overall_warnings)
            elements.append(mo.md(f"### Overall warnings\n{ow}"))

        elements.append(
            mo.md(
                f"### Proposed cells ({len(self.plan.cells)}) — toggle off any "
                f"you do not want, then submit"
            )
        )

        for i, (cell, checkbox) in enumerate(
            zip(self.plan.cells, self._checkboxes), start=1
        ):
            badge = COST_BADGE.get(cell.est_cost, "?")
            header = mo.md(f"#### [{i}] {cell.title}  ·  `{badge} {cell.est_cost}`")
            rationale = mo.md(f"_{cell.rationale}_")

            card: list[Any] = [header, rationale]
            if cell.depends_on:
                deps = ", ".join(f"`{d}`" for d in cell.depends_on)
                card.append(mo.md(f"Depends on: {deps}"))
            if cell.warnings:
                wlist = "\n".join(f"- {w}" for w in cell.warnings)
                card.append(mo.md(f"**Warnings**\n{wlist}"))
            card.append(mo.md(f"```python\n{cell.code}\n```"))

            card_box = mo.vstack(card)
            row = mo.hstack(
                [checkbox, card_box], justify="start", align="start", gap=1
            )
            elements.append(row)

        elements.append(self._form)
        return mo.vstack(elements)

    # ----- Marimo / IPython display hooks ----------------------------------
    def _display_(self):
        return self._rendered

    def _mime_(self):
        # Delegate to whatever mime the rendered vstack reports.
        if hasattr(self._rendered, "_mime_"):
            return self._rendered._mime_()
        return ("text/html", repr(self._rendered))

    def _repr_html_(self):
        if hasattr(self._rendered, "_repr_html_"):
            return self._rendered._repr_html_()
        return f"<pre>{self._rendered!r}</pre>"

    # ----- Apply -----------------------------------------------------------
    async def apply(
        self,
        *,
        marimo_base_url: str = "http://127.0.0.1:2718",
    ) -> str:
        """Materialize the checked proposed cells in the live notebook.

        Why this is awkward: `marimo._code_mode.get_context()` requires the
        kernel's document context variable, which is only set when code
        enters the kernel through the `/api/kernel/execute` endpoint. The
        widget itself runs INSIDE the kernel, so it cannot call code_mode
        directly. Workaround: from a daemon thread, POST the create_cell
        script back to `/api/kernel/execute` so it runs in the proper
        context after our cell completes.

        Called from a downstream cell as `await widget.apply()`. Re-runs
        each time the form is re-submitted.
        """
        if self._form.value is None:
            return "(no cells applied — submit the form in the planner above)"

        checked: list[bool] = list(self._form.value)
        selected_pairs = [
            (i, cell)
            for i, (cell, on) in enumerate(zip(self.plan.cells, checked), start=1)
            if on
        ]
        if not selected_pairs:
            return "(no cells selected — check at least one box and re-submit)"

        # Build the script that will run via /api/kernel/execute. We embed
        # the cell payloads as a JSON literal so any quotes/backslashes in
        # the agent's code survive round-tripping through Python source.
        import json

        cells_payload = [
            {
                "title": c.title,
                "code": c.code,
                "hide_code": self._hide_code,
            }
            for _, c in selected_pairs
        ]
        cells_json = json.dumps(cells_payload)
        create_script = (
            "import marimo._code_mode as cm\n"
            "import json\n"
            f"_cells = json.loads({cells_json!r})\n"
            "async with cm.get_context() as ctx:\n"
            "    _anchor = list(ctx.cells.keys())[-1]\n"
            "    for _c in _cells:\n"
            "        _new = ctx.create_cell(\n"
            "            _c['code'], after=_anchor, hide_code=_c['hide_code']\n"
            "        )\n"
            "        ctx.run_cell(_new)\n"
            "        _anchor = _new\n"
            "print('marimo-pair: created', len(_cells), 'cells')\n"
        )

        # Fire-and-forget from a daemon thread so the POST runs after the
        # current cell completes. Inline httpx (a marimo transitive dep);
        # don't surface POST errors to the caller -- we report them via the
        # marimo server log if anything goes wrong.
        import os
        import threading
        import httpx

        def _post_create() -> None:
            try:
                env_token = os.environ.get("MARIMO_TOKEN")
                headers = {"Content-Type": "application/json"}
                if env_token:
                    headers["Authorization"] = f"Bearer {env_token}"
                with httpx.Client(timeout=120.0) as client:
                    sessions = client.get(
                        f"{marimo_base_url}/api/sessions", headers=headers
                    ).json()
                    if not sessions:
                        return
                    session_id = next(iter(sessions))
                    headers["Marimo-Session-Id"] = session_id
                    client.post(
                        f"{marimo_base_url}/api/kernel/execute",
                        json={"code": create_script},
                        headers=headers,
                    )
            except Exception:  # noqa: BLE001 -- intentional: log & forget
                # Last-ditch debug: write to a per-project file. Silent
                # failure is preferable to crashing a daemon thread.
                try:
                    import tempfile
                    import traceback

                    tmp = os.path.join(tempfile.gettempdir(), "ds_copilot_apply.log")
                    with open(tmp, "a", encoding="utf-8") as f:
                        f.write(traceback.format_exc() + "\n---\n")
                except Exception:
                    pass

        threading.Thread(target=_post_create, daemon=True).start()

        titles = ", ".join(f"[{i}] {c.title}" for i, c in selected_pairs)
        return (
            f"Queued {len(selected_pairs)} cell(s) to be created below:\n"
            f"  {titles}\n"
            f"They will appear in the notebook within a second or two."
        )


def planner_widget(
    df: Any,
    goal: str,
    *,
    target: str | None = None,
    existing_cells: list[str] | None = None,
    **plan_kwargs: Any,
) -> PlannerWidget:
    """Build a profile, ask Claude for a plan, return a renderable widget.

    Blocks for ~60-90s while the plan is generated (Opus 4.7 + high effort
    via the claude-cli backend). For faster iteration during dev, pass
    `effort="medium"` or `backend="anthropic"` (with API credits).

    Args:
        df: A Polars or pandas DataFrame to profile.
        goal: Natural-language goal handed to the planner.
        target: Optional target column name (enables leakage hints in the
            profile and target-rate context in the planner prompt).
        existing_cells: Optional one-line summaries of cells already in the
            notebook. The planner uses these to avoid re-importing names
            and to compose new cells with the existing dataflow.
        **plan_kwargs: Forwarded to `ds_copilot.agent.plan()`
            (e.g. `backend`, `model`, `effort`, `timeout_s`).
    """
    p = profile(df, target=target)
    plan_obj = plan(
        goal=goal,
        profile=p,
        existing_cells=existing_cells,
        **plan_kwargs,
    )
    return PlannerWidget(plan_obj, p)
