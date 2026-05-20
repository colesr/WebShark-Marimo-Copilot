"""Week 1 marimo-pair trial notebook.

Starting state: a synthetic churn dataframe `df` is preloaded as a Polars DataFrame.
The marimo-pair agent picks up from here.

Trial task given to the agent:
    "Build a churn baseline classifier on `df`. Profile the schema first,
     then train a simple model predicting `churned`, and flag any columns
     that might be leaking the target."

What we are watching for (Week 1 findings):
    - Does it correctly identify the schema without hallucinating columns?
    - Does it plan ahead, or just react cell-by-cell?
    - Does it propose, or just execute? (Autonomy stinginess.)
    - Does it surface leakage risk on its own?
    - Does it install packages it needs (sklearn) via ctx.packages.add()?

Run with:
    marimo edit notebooks/01_marimo_pair_trial.py --no-token
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
    # Week 1 — marimo-pair trial

    A synthetic churn dataset is loaded as `df` below. The marimo-pair agent
    is about to take over and build out an analysis.

    No approval gates — this is the autonomous mode the wedge plan critiques.
    Watch the cells appear and note anything that feels off.
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
    return df, np, pl


@app.cell
def _(df, mo):
    mo.md(f"""
    ### Starting dataframe — shape={df.shape}
    """)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    # First-class schema profile from ds_copilot.profiler
    # (Compare with the ad-hoc cell DBId below that the agent wrote earlier.)
    from ds_copilot.profiler import profile_widget
    profile_widget(df, target="churned")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Churn baseline -- built by marimo-pair

    The agent is taking over from the preloaded `df` cell above and building a
    baseline classifier predicting `churned`. Each step below is its own cell so
    you can see the order in which the agent reasoned about the problem.
    """)
    return


@app.cell
def _(df, mo, pl):
    # Schema profile: dtypes, null rates, cardinality, target rate
    _n = df.height
    _target_rate = float(df["churned"].mean())
    _rows = []
    for _col in df.columns:
        _s = df[_col]
        _nu = int(_s.n_unique())
        _rows.append({
            "column": _col,
            "dtype": str(_s.dtype),
            "null_rate": round(float(_s.null_count()) / _n, 4),
            "cardinality": _nu,
            "card_pct": round(_nu / _n, 4),
        })
    _profile = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(f"**rows:** {_n:,}  **cols:** {len(df.columns)}  **target rate (churned=1):** {_target_rate:.3f}"),
        _profile,
    ])

    return


@app.cell
def _(df, mo, np, pl):
    # Leakage check: flag high-cardinality cols and any high |corr| with target.
    # np and pl already imported by the data-loading cell.
    _target = df["churned"].to_numpy()
    _n = df.height
    _flags = []
    for _col in df.columns:
        if _col == "churned":
            continue
        _s = df[_col]
        _nu = int(_s.n_unique())
        _card_pct = _nu / _n
        _reasons = []
        if _card_pct > 0.95:
            _reasons.append(f"unique-per-row ({_nu}/{_n}) - likely an ID, no signal")
        if _s.dtype.is_numeric():
            _vals = _s.to_numpy().astype(float)
            if np.std(_vals) > 0:
                _corr = float(np.corrcoef(_vals, _target)[0, 1])
                if abs(_corr) > 0.95:
                    _reasons.append(f"|corr| with churned = {abs(_corr):.3f}")
        if _reasons:
            _flags.append({"column": _col, "reasons": "; ".join(_reasons)})
    _leak_df = pl.DataFrame(_flags) if _flags else None
    mo.vstack([
        mo.md("**Flagged columns (will be dropped or watched)**"),
        _leak_df if _leak_df is not None else mo.md("_None flagged._"),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoding strategy

    - **Drop `customer_id`** -- unique-per-row, flagged as leakage / no signal.
    - **One-hot** `contract_type`, `internet_service`, `payment_method` (3-4 cats each).
    - Numeric features (`tenure_months`, `monthly_charges`) pass through.
    - Target `churned` already 0/1.
    """)
    return


@app.cell
def _(df, mo, pl):
    # Build feature matrix X and target y
    _cat_cols = ["contract_type", "internet_service", "payment_method"]
    _num_cols = ["tenure_months", "monthly_charges"]
    _X_cat = df.select(_cat_cols).to_dummies(_cat_cols)
    _X_num = df.select(_num_cols).cast(pl.Float64)
    X = pl.concat([_X_num, _X_cat], how="horizontal")
    y = df["churned"].to_numpy()
    mo.vstack([
        mo.md(f"**X shape:** {X.shape}  |  **y shape:** {y.shape}"),
        mo.md(f"**X columns:** {X.columns}"),
    ])

    return X, y


@app.cell
def _(X, mo, y):
    # Stratified 80/20 split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=0.2, stratify=y, random_state=42
    )
    mo.md(
        f"**Train:** {X_train.shape[0]} rows  |  churn rate {y_train.mean():.3f}\n\n"
        f"**Test:** {X_test.shape[0]} rows  |  churn rate {y_test.mean():.3f}"
    )

    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, mo, y_train):
    # Baseline: logistic regression with default hyperparameters
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    mo.md(f"**Fit complete.** n_iter = {int(model.n_iter_[0])}, features = {X_train.shape[1]}")

    return (model,)


@app.cell
def _(X_test, mo, model, y_test):
    # Evaluate on holdout
    from sklearn.metrics import accuracy_score, roc_auc_score
    _pred = model.predict(X_test)
    _proba = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, _pred))
    auc = float(roc_auc_score(y_test, _proba))
    _baseline = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    mo.md(
        f"**Test accuracy:** {acc:.3f}\n\n"
        f"**Test ROC AUC:** {auc:.3f}\n\n"
        f"_Majority-class baseline accuracy: {_baseline:.3f}_"
    )

    return acc, auc


@app.cell
def _(X_test, mo, model, pl, y_test):
    # Confusion matrix as a polars table
    from sklearn.metrics import confusion_matrix
    _cm = confusion_matrix(y_test, model.predict(X_test))
    _cm_df = pl.DataFrame({
        "actual": ["churn=0 (no)", "churn=1 (yes)"],
        "pred_no":  [int(_cm[0, 0]), int(_cm[1, 0])],
        "pred_yes": [int(_cm[0, 1]), int(_cm[1, 1])],
    })
    mo.vstack([mo.md("**Confusion matrix**"), _cm_df])

    return


@app.cell(hide_code=True)
def _(acc, auc, mo):
    mo.md(rf"""
    ### Summary

    Pipeline ran end-to-end in 10 agent-written cells: schema profile, leakage
    flag, encoding strategy, one-hot, stratified split, fit, accuracy / ROC AUC,
    confusion matrix.

    **Result on the holdout (n=200):** accuracy = **{acc:.3f}**, ROC AUC =
    **{auc:.3f}**. The confusion matrix shows the classifier predicts `0` for
    **every** test row -- it has collapsed to the majority class.

    That is the **correct** outcome: this dataset was generated with
    `rng.choice([0, 1], n, p=[0.74, 0.26])` for `churned`, independent of all
    feature columns. Accuracy of 0.745 simply matches the base rate. The leakage
    check correctly flagged only `customer_id` (unique-per-row), and no
    features showed real signal because there is none to find.

    **Lesson for the wedge:** "accuracy alone" is deceptive on imbalanced
    targets. A practitioner-grade copilot should surface the **majority-class
    baseline** alongside accuracy by default, and warn when predictions are
    trivially constant -- exactly the kind of friendly guard rail the pivoted
    plan-then-approve wedge can add on top of `marimo-pair`'s raw cell CRUD.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Week 2 demo: plan-then-approve agent

    Below: `planner_widget(df, goal=..., target="churned")` calls the schema
    profiler, then asks Claude Opus 4.7 (via the free claude-cli backend) for
    a structured Plan. The widget renders the plan with one checkbox per
    proposed cell. Toggle off anything you don't want, click "Apply selected
    cells", and the next cell (`await planner.apply()`) materializes only
    the approved cells into this notebook.

    First call blocks for ~60-90s. Be patient -- the trade-off is one slow
    plan instead of an autonomous agent surprising you.
    """)
    return


@app.cell
def _(df):
    from ds_copilot.ui import planner_widget
    planner = planner_widget(
        df,
        goal=(
            "Suggest 3 feature engineering ideas that might add signal for a "
            "churn classifier on this synthetic data. For each idea, propose a "
            "diagnostic cell that checks whether the new feature actually "
            "correlates with the target before any model training."
        ),
        target="churned",
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            "import numpy as np",
            "df = synthetic churn dataframe (1000 rows, 7 cols, columns include customer_id, tenure_months, monthly_charges, contract_type, internet_service, payment_method, churned)",
            "the notebook already has a baseline logistic regression in cells below",
        ],
        effort="high",
    )
    planner

    return (planner,)


@app.cell
async def _(planner):
    # Re-runs automatically when you click "Apply selected cells" in the planner above.
    # Until you submit, returns a placeholder string.
    result = await planner.apply()
    result

    return


if __name__ == "__main__":
    app.run()
