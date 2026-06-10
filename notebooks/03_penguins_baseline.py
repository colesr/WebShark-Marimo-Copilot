"""Plan-then-approve on a small, fully reproducible real dataset.

Unlike 02_favorita_realdata.py (which needs the gitignored Kaggle CSVs),
this demo runs on Seaborn's built-in `penguins` set, so it works on a
fresh clone with no data download. The planner is given a neutral
classification goal and asked to be honest about what conclusions are
robust at ~333 rows; the cells below the planner are what `apply()`
materialized from the approved plan (encode -> stratified split ->
logreg baseline -> confusion matrix -> 5-fold CV).

Run from the project root with:
    .\\.venv\\Scripts\\marimo.exe edit notebooks/03_penguins_baseline.py --no-token
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import seaborn as sns

    # pl.from_pandas needs pyarrow for pandas-extension dtypes (Int64, etc.);
    # we route via numpy arrays for portability.
    _penguins = sns.load_dataset("penguins").dropna()
    df = pl.DataFrame({col: _penguins[col].to_numpy() for col in _penguins.columns})
    df.head(10)
    return df, mo


@app.cell
def _(df):
    from ds_copilot import profile_widget
    profile_widget(df, target="species")
    return


@app.cell
def _(df):
    from ds_copilot import planner_widget

    planner = planner_widget(
        df,
        goal=(
            "Build a baseline classifier predicting `species`. "
            "Encode categoricals as needed, do a stratified train/test "
            "split, fit a logistic regression baseline, report accuracy "
            "and a confusion matrix. The dataset is small (~333 rows after "
            "dropna) -- be honest about what conclusions are robust at "
            "that sample size."
        ),
        target="species",
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            "import numpy as np",
            "import seaborn as sns",
            ("df = polars DataFrame from sns.load_dataset('penguins').dropna(). "
             "Columns: species (target, 3 classes), island, bill_length_mm, "
             "bill_depth_mm, flipper_length_mm, body_mass_g, sex"),
        ],
    )
    planner
    return (planner,)


@app.cell
async def _(planner):
    result = await planner.apply()
    result
    return


@app.cell(hide_code=True)
def _(df, mo):
    feature_cols = ['island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X_df = df.select(feature_cols).to_dummies(columns=['island', 'sex'], drop_first=False)
    y = df['species'].to_numpy()
    X = X_df.to_numpy()
    mo.md(f"Feature matrix shape: {X.shape}; columns: {X_df.columns}")

    return X, X_df, y


@app.cell(hide_code=True)
def _(X, X_df, mo, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    mo.md(f"Train: {X_train.shape[0]} rows; Test: {X_test.shape[0]} rows")

    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(X_train, mo, y_train):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=0)),
    ])
    model.fit(X_train, y_train)
    mo.md(f"Trained. Classes: {list(model.classes_)}")

    return (model,)


@app.cell(hide_code=True)
def _(X_test, mo, model, y_test):
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    mo.md(f"**Test accuracy:** {acc:.3f}\n\n```\n{report}\n```")

    return (y_pred,)


@app.cell(hide_code=True)
def _(model, y_pred, y_test):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(model.classes_))); ax.set_xticklabels(model.classes_, rotation=45, ha='right')
    ax.set_yticks(range(len(model.classes_))); ax.set_yticklabels(model.classes_)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion matrix (test)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig

    return


@app.cell(hide_code=True)
def _(X, mo, model, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mo.md(f"**5-fold CV accuracy:** {scores.mean():.3f} ± {scores.std():.3f}  \nFold scores: {[round(s, 3) for s in scores]}")

    return


@app.cell
def _():
    from ds_copilot import dag_widget
    dag_widget
    return


if __name__ == "__main__":
    app.run()
