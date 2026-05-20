"""Week 0 gut-check notebook.

Validates that:
  1. Marimo runs on this machine.
  2. We can import the core stack (polars, anthropic, ds_copilot).
  3. `mo.ui.chat` can host a callback that calls Claude with schema-grounded context.
  4. The grounded agent (knows the dataframe schema) feels qualitatively better
     than asking Claude in a vanilla chat window.

Run with:
    marimo edit notebooks/00_gut_check.py

Requires ANTHROPIC_API_KEY in .env (or environment).
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Week 0 gut-check — DS Copilot

    The goal: confirm that an AI chat panel sitting next to a live dataframe can
    produce code suggestions that beat vanilla Claude — without any custom UI
    beyond `mo.ui.chat`.

    Below is a sample Polars dataframe. The chat panel knows its schema.
    Try asking things like:

    - *"What's the relationship between tenure and churn?"*
    - *"Write Polars code to compute the churn rate by contract type."*
    - *"Are there any columns I should worry about leaking the target?"*
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np

    rng = np.random.default_rng(42)
    n = 1_000

    df = pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["month-to-month", "1-year", "2-year"], n),
            "internet_service": rng.choice(["dsl", "fiber", "none"], n),
            "payment_method": rng.choice(
                ["bank-transfer", "credit-card", "electronic-check", "mailed-check"], n
            ),
            "churned": rng.choice([0, 1], n, p=[0.74, 0.26]),
        }
    )
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"### Sample data ({df.height:,} rows)")
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    """Build a compact schema summary that we feed to Claude as grounded context."""

    def schema_summary(frame) -> str:
        lines = [f"# Dataframe: shape={frame.shape}", "", "## Columns"]
        for name, dtype in zip(frame.columns, frame.dtypes):
            try:
                sample_vals = frame[name].drop_nulls().head(3).to_list()
            except Exception:
                sample_vals = []
            lines.append(f"- `{name}` ({dtype}) — examples: {sample_vals}")
        return "\n".join(lines)

    summary = schema_summary(df)
    return (summary,)


@app.cell
def _(mo, summary):
    mo.md(f"### Schema context fed to Claude\n\n```\n{summary}\n```")
    return


@app.cell
def _():
    """Set up the Anthropic client. Reads ANTHROPIC_API_KEY from .env or env."""
    import os
    from dotenv import load_dotenv
    from anthropic import Anthropic

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None
    return (client,)


@app.cell
def _(client, mo, summary):
    if client is None:
        chat_widget = mo.callout(
            "**ANTHROPIC_API_KEY missing.** Copy `.env.example` to `.env` and add your key, "
            "then re-run the notebook.",
            kind="warn",
        )
    else:
        SYSTEM = (
            "You are a data-science copilot embedded in a Marimo notebook. "
            "You have grounded context about the user's current dataframe `df` "
            "(schema + sample values). When the user asks a question:\n"
            "  1. Reason from the actual columns and dtypes shown below — do NOT hallucinate columns.\n"
            "  2. If proposing code, write Polars (preferred) or SQL (via DuckDB) "
            "in a fenced code block the user can copy into a new cell.\n"
            "  3. Prefer concise answers. One paragraph + one code block is ideal.\n\n"
            f"## Current dataframe `df`\n\n{summary}\n"
        )

        def claude_callback(messages, config):
            """Marimo chat callback. Streams from Claude using prompt-cached system prompt."""
            api_messages = [
                {"role": m.role, "content": m.content} for m in messages
            ]
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=api_messages,
            )
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )

        chat_widget = mo.ui.chat(claude_callback)
    return (chat_widget,)


@app.cell
def _(chat_widget):
    chat_widget
    return


if __name__ == "__main__":
    app.run()
