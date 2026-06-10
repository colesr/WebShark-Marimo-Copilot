"""Week 3 demo: plan-then-approve catches a deliberate target leak.

The dataset below has a deliberately leaky feature `churn_score` that is
essentially a noisy copy of the target `churned`. A practitioner who ran
the autonomous mode would silently train on it and ship a model with
suspiciously high AUC.

We let `planner_widget` look at the schema first. We expect:
    1. The profiler flags churn_score as near-perfectly correlated with
       the target.
    2. The agent's LeakageAudit lists churn_score under columns_to_drop.
    3. If the plan does not drop it before training, training_safe=False
       and Apply blocks any training cell until we explicitly override.

Run with:
    marimo edit notebooks/01_demo_churn.py --no-token
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
    # Plan-then-approve vs a deliberate target leak

    Below: a synthetic churn dataset where `churn_score` is essentially a
    noisy copy of the target. A naive autonomous agent would train on it
    and produce a misleadingly high AUC. We test whether plan-first
    catches it before any model gets fit.
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np

    rng = np.random.default_rng(7)
    n = 1_000

    churned = rng.choice([0, 1], n, p=[0.74, 0.26])

    df = pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["m2m", "1y", "2y"], n),
            "internet_service": rng.choice(["dsl", "fiber", "none"], n),
            # DELIBERATE LEAK: this is essentially the target plus noise.
            "churn_score": (churned * 100.0 + rng.normal(0, 0.01, n)).round(3),
            "churned": churned,
        }
    )
    return df, pl


@app.cell
def _(df, mo):
    mo.md(f"""
    ### Dataset shape: {df.shape} — preview below
    """)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: profile the schema (no AI yet)

    Even without the LLM, the profiler's leakage hints catch high-cardinality
    IDs and any column with |corr| > 0.9 with the target. Watch `churn_score`
    light up.
    """)
    return


@app.cell
def _(df):
    from ds_copilot import profile_widget

    profile_widget(df, target="churned")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: plan a baseline churn classifier

    The planner is told the target is `churned` and asked to build a
    baseline. We expect its `LeakageAudit` to list `churn_score` (and
    probably `customer_id`) under columns_to_drop and to mark the plan
    `training_safe=True` only if it drops them before training.

    First call blocks ~60-90s.
    """)
    return


@app.cell
def _(df):
    from ds_copilot import planner_widget

    planner = planner_widget(
        df,
        goal=(
            "Build a baseline binary classifier predicting `churned`. "
            "Encode categoricals, split train/test, fit a logistic "
            "regression, report accuracy + ROC AUC + a confusion matrix."
        ),
        target="churned",
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            "import numpy as np",
            (
                "df = synthetic churn dataframe with columns: customer_id, "
                "tenure_months, monthly_charges, contract_type, "
                "internet_service, churn_score, churned"
            ),
        ],
        effort="high",
    )
    return (planner,)


@app.cell
def _(planner):
    planner
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: try to apply

    Submit the form above. If `training_safe=False`, applying any
    training-looking cell will be blocked unless you check the override
    box and re-submit. If the agent did the right thing, the plan should
    drop `churn_score` (and `customer_id`) before training and Apply
    should let you proceed without override.

    *Note: clicking **Apply** materializes cells like the static baseline
    below it. Re-clicking stacks duplicates that marimo flags as
    redefinitions — clear the extra cells if that happens.*
    """)
    return


@app.cell
async def _(planner):
    result = await planner.apply()
    return (result,)


@app.cell
def _(mo, result):
    mo.md(f"""
    ```\n{result}\n```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The applied baseline (one good plan's output)

    These cells mirror what the planner proposes and Apply materializes:
    drop the leaking columns first, then encode → split → fit → evaluate.
    Because the leaks are gone, the ROC AUC is honest (~0.5), not the
    misleading ~1.0 you'd get by training on `churn_score`.
    """)
    return


@app.cell
def _(df, mo):
    leaking_cols = ["customer_id", "churn_score"]
    feature_cols = [c for c in df.columns if c not in leaking_cols + ["churned"]]
    mo.md(
        f"**Dropped (leakage / ID):** {leaking_cols}\n\n"
        f"**Features:** {feature_cols}\n\n"
        f"**Target:** churned"
    )
    return (feature_cols,)


@app.cell
def _(df, feature_cols, mo):
    X_df = df.select(feature_cols).to_dummies(columns=["contract_type", "internet_service"])
    X = X_df.to_numpy()
    y = df["churned"].to_numpy()
    mo.md(
        f"X shape: {X.shape}  |  y positive rate: {y.mean():.3f}\n\n"
        f"Encoded columns:\n\n{X_df.columns}"
    )
    return X, y


@app.cell
def _(X, mo, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    mo.md(
        f"Train: {X_train.shape}, pos rate {y_train.mean():.3f}  \n"
        f"Test:  {X_test.shape}, pos rate {y_test.mean():.3f}"
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, mo, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    mo.md(f"Fitted. Train accuracy: {model.score(X_train, y_train):.3f}")
    return (model,)


@app.cell
def _(X_test, mo, model, pl, y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pl.DataFrame({
        "": ["actual_neg", "actual_pos"],
        "pred_neg": [int(cm[0, 0]), int(cm[1, 0])],
        "pred_pos": [int(cm[0, 1]), int(cm[1, 1])],
    })
    mo.vstack([
        mo.md(
            f"**Test accuracy:** {acc:.4f}  \n"
            f"**Test ROC AUC:** {auc:.4f}  \n\n"
            f"(Baseline from always predicting 0: {1 - y_test.mean():.4f})"
        ),
        mo.ui.table(cm_df, selection=None),
    ])
    return (y_pred,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: read the decision log

    Every plan/accept/reject was appended to `.ds_copilot/decisions.jsonl`.
    The history helper reads it back — useful for "what did we decide and
    why" audits later.
    """)
    return


@app.cell
def _(mo):
    def _():
        from ds_copilot import history
        import polars as pl

        def summarize(e):
            p = e.payload
            if e.event_type == "plan_requested":
                return f"goal={str(p.get('goal', ''))[:50]}... target={p.get('target')}"
            if e.event_type == "plan_returned":
                audit = p.get("leakage_audit") or {}
                return (
                    f"{len(p.get('cells', []))} cells; "
                    f"training_safe={audit.get('training_safe')}; "
                    f"drops={audit.get('columns_to_drop', [])}"
                )
            if e.event_type == "cells_applied":
                return (
                    f"applied {len(p.get('applied_titles', []))} cells "
                    f"(override={p.get('override_used', False)})"
                )
            return ""

        events = history()
        rows = [
            {
                "ts": e.timestamp.isoformat(),
                "session": e.session_id[:8],
                "type": e.event_type,
                "summary": summarize(e),
            }
            for e in events[-20:]
        ]
        return mo.vstack(
            [
                mo.md(
                    f"### Decision history — last {len(rows)} of "
                    f"{len(events)} events"
                ),
                pl.DataFrame(rows) if rows else mo.md("_(no events yet)_"),
            ]
        )

    _()
    return


@app.cell
def _():
    from ds_copilot import dag_widget

    dag_widget("notebooks/01_demo_churn.py")
    return


@app.cell(hide_code=True)
def _(df, mo):
    feature_cols = ["tenure_months", "monthly_charges", "contract_type", "internet_service"]
    X_raw = df.select(feature_cols)
    y = df.select("churned").to_series()
    mo.md(f"**Features kept:** {feature_cols}  \n**Dropped (leakage):** `customer_id` (ID), `churn_score` (|corr|=1.0 with target)  \n**Rows:** {X_raw.height}, **Positive rate:** {y.mean():.3f}")
    return X_raw, feature_cols, y


@app.cell(hide_code=True)
def _(X_raw, mo):
    X_encoded = X_raw.to_dummies(columns=["contract_type", "internet_service"], drop_first=True)
    mo.md(f"**Encoded feature matrix shape:** {X_encoded.shape}  \n**Columns:** {X_encoded.columns}")
    return (X_encoded,)


@app.cell(hide_code=True)
def _(X_encoded, mo, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.to_numpy(),
        y.to_numpy(),
        test_size=0.2,
        stratify=y.to_numpy(),
        random_state=42,
    )
    mo.md(f"**Train:** {X_train.shape[0]} rows | **Test:** {X_test.shape[0]} rows | **Train pos rate:** {y_train.mean():.3f} | **Test pos rate:** {y_test.mean():.3f}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(X_train, mo, y_train):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    mo.md("**Logistic regression fit complete** — StandardScaler + LogisticRegression(max_iter=1000)")
    return (model,)


@app.cell(hide_code=True)
def _(X_test, mo, model, y_test):
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mo.md(f"### Test-set metrics\n- **Accuracy:** {acc:.4f}\n- **ROC AUC:** {auc:.4f}")
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo, pl, y_pred, y_test):
    from sklearn.metrics import confusion_matrix
    import polars as _pl  # alias-safe; uses existing pl

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pl.DataFrame({
        "": ["actual: no churn", "actual: churn"],
        "pred: no churn": [int(cm[0, 0]), int(cm[1, 0])],
        "pred: churn": [int(cm[0, 1]), int(cm[1, 1])],
    })
    mo.vstack([mo.md("### Confusion matrix (test set)"), mo.ui.table(cm_df)])
    return


if __name__ == "__main__":
    app.run()
