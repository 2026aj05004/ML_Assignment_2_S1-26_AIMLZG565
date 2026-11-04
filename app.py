"""Streamlit app: evaluate the saved classifiers on an uploaded test CSV.

    streamlit run app.py

The app only loads the pipelines in model/ and calls predict / predict_proba.
It never trains or refits anything.
"""

import csv
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from data_loader import CLASSES, DISPLAY_NAMES, TARGET, UploadError, validate_upload
from metrics import (
    METRIC_NAMES,
    classification_report_frame,
    compute_metrics,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
BUNDLED_CSV = ROOT / "test_data.csv"
DICTIONARY_JSON = ROOT / "data" / "feature_dictionary.json"
REPO_URL = "https://github.com/2026aj05004/ML_Assignment_2_S1-26_AIMLZG565"
UCI_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

st.set_page_config(page_title="Students' Dropout and Academic Success", layout="wide")


@st.cache_resource
def load_models():
    # Loading joblib files runs pickle; these are our own files from train.py.
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / f) for name, f in meta["model_files"].items()}
    return meta, models


def evaluate(model, X, y):
    y_pred = model.predict(X)
    classes = [str(c) for c in model.classes_]
    return y_pred, compute_metrics(y, y_pred, model.predict_proba(X), classes)


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def best_model_sentence(f1: pd.Series) -> str:
    """Name the best model by macro F1, and say so when the runner-up is effectively tied.

    The table shows 3 decimals, so two models can look identical; the sentence uses 4.
    """
    ranked = f1.sort_values(ascending=False)
    (best, s1), (second, s2) = ranked.iloc[:2].items()
    if s1 - s2 < 0.001:
        return (
            f"**{best}** ({s1:.4f}) and **{second}** ({s2:.4f}) are effectively tied on "
            f"macro F1: the gap of {s1 - s2:.4f} is too small to call either better on this file."
        )
    return f"Best on this file by macro F1: **{best}** ({s1:.4f}), ahead of {second} ({s2:.4f})."


@st.cache_data
def load_dictionary():
    """Variable descriptions and code meanings from UCI (data/feature_dictionary.json)."""
    variables = json.loads(DICTIONARY_JSON.read_text())["variables"]
    return {v["column"]: v for v in variables}


def decode(df: pd.DataFrame, dictionary: dict) -> pd.DataFrame:
    """Replace integer codes with their UCI labels, for display only."""
    out = df.copy()
    for col in out.columns:
        codes = dictionary.get(col, {}).get("codes")
        if codes:
            out[col] = out[col].map(lambda v: codes.get(str(int(v)), f"code {v}"))
    return out


def confusion_matrix_chart(y_true, y_pred, classes):
    """Confusion-matrix heatmap as an Altair chart, so it follows the light/dark theme.

    Cell text switches to white on dark cells, which keeps it readable in both themes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    cells = pd.DataFrame(
        [(a, p, int(cm[i, j])) for i, a in enumerate(classes) for j, p in enumerate(classes)],
        columns=["Actual", "Predicted", "Students"],
    )
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=list(classes), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=list(classes)),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Students:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual", "Predicted", "Students"],
    )
    text = base.mark_text(fontSize=15).encode(
        text="Students:Q",
        color=alt.condition(
            alt.datum.Students > cells["Students"].max() / 2, alt.value("white"), alt.value("black")
        ),
    )
    return (heat + text).properties(height=300)


meta, models = load_models()
dictionary = load_dictionary()

st.title("Predict Students' Dropout and Academic Success")
st.caption(
    "Five classifiers trained on UCI dataset #697 predict whether a university student will "
    "**drop out**, **still be enrolled** or **graduate**. Check how they perform on labelled data."
)

# ---------------------------------------------------------------- about (top of page)
with st.expander("About the dataset", expanded=True):
    st.markdown(
        f"""
**Why it exists.** The dataset comes from a project at a higher-education institution in
Portugal that aimed to **reduce academic dropout and failure** by using machine learning to
spot students at risk early in their studies, so that support could be put in place.
Each row is one student, enrolled in degrees such as agronomy, design, education, nursing,
journalism, management, social service and technologies. The data was combined from several
separate institutional databases, cleaned of anomalies and outliers by its authors, and has
**no missing values**. Source: [UCI dataset #697]({UCI_URL}) (CC BY 4.0).
"""
    )
    data_col, class_col, use_col = st.columns([5, 4, 4], gap="large")
    with data_col:
        st.markdown(
            """
**36 features per student**, in five groups:
- **Enrolment and background**: course, application mode and order, previous
  qualification and grade, admission grade, age, nationality, gender, ...
- **Family**: each parent's qualification and occupation
- **Financial**: tuition fees up to date, debtor, scholarship holder
- **First-year results**: units credited, enrolled, evaluated and approved, and the
  average grade, for each of the two semesters
- **Economy**: unemployment rate, inflation rate and GDP

Many columns are numeric codes (for example `Course = 9500` is Nursing). The
**Feature dictionary** below explains every column and code.
"""
        )
    with class_col:
        # A markdown table (not st.dataframe) so it wraps instead of clipping in a narrow column.
        st.markdown(
            """
**What is predicted**: the outcome at the end of the normal course duration

| Outcome | Students | Share |
|---|---:|---:|
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |
| Graduate | 2,209 | 49.9% |

*Enrolled* means still studying after the normal duration. It is the smallest class and
the hardest to predict.
"""
        )
    with use_col:
        st.markdown(
            """
**When the prediction is made.** The features include first- and second-semester
results, so this is a prediction **at the end of the first year**, not at admission.
That still leaves time for a student-support office to contact students at risk.

**Using this app**
1. Pick a model in the sidebar.
2. Keep the bundled `test_data.csv` (885 students held out from training, the 80/20
   split the dataset's authors recommend), or upload your own labelled CSV.
3. Read the metrics and confusion matrix, or compare all five models.
"""
        )

with st.expander("Feature dictionary: what every column means"):
    rows = []
    for col, v in dictionary.items():
        codes = v["codes"]
        if codes and len(codes) > 2:  # long code lists: one example, full list in the lookup below
            code, label = next(iter(codes.items()))
            meaning = f"{len(codes)} coded categories, e.g. {code} = {label}"
        elif codes:
            meaning = " / ".join(f"{k} = {lab}" for k, lab in codes.items())
        else:
            meaning = v["description"]
        rows.append({"Column": col, "Group": v["group"], "Meaning": meaning})
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Column": st.column_config.TextColumn(width="medium"),
            "Group": st.column_config.TextColumn(width="medium"),
            "Meaning": st.column_config.TextColumn(width="large"),
        },
    )
    coded = [c for c, v in dictionary.items() if v["codes"] and len(v["codes"]) > 2]
    pick = st.selectbox("Look up all codes for a column", coded,
                        format_func=lambda c: DISPLAY_NAMES.get(c, c))
    st.dataframe(
        pd.DataFrame(list(dictionary[pick]["codes"].items()), columns=["Code", "Meaning"]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Descriptions and codes as published by UCI, with typos corrected. The nationality "
        "column is spelled *Nacionality*, as in the original dataset, so raw UCI files work "
        "unchanged."
    )

with st.expander("How the metrics are defined"):
    st.markdown(
        f"""
All metrics are computed on the rows you provide:
- **Accuracy**: share of students whose outcome was predicted correctly.
- **AUC**: macro-averaged one-vs-rest ROC AUC over the three classes, from predicted
  probabilities. Needs every class to be present; otherwise shown as n/a.
- **Precision / Recall / F1**: macro averages, so each class counts equally regardless of size.
  A class missing from the file counts as 0, which caps these at 0.667 with two classes present.
- **MCC**: Matthews correlation coefficient (multiclass form); 0 = no better than chance, 1 = perfect.

The primary comparison criterion, fixed before training, is **macro F1**.

Models were trained once by `train.py` (scikit-learn {meta['sklearn_version']}); this app
only loads them. Source code: [{REPO_URL}]({REPO_URL})
"""
    )

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Choose a model")
    model_name = st.selectbox("Model", list(models))
    st.caption(f"Tuned settings: `{meta['best_params'][model_name]}`")

    st.header("2. Provide test data")
    use_bundled = st.checkbox(
        "Use bundled test_data.csv", value=True,
        help="The stratified 20% hold-out saved by train.py (885 students).",
    )
    uploaded = st.file_uploader(
        "...or upload your own CSV", type="csv", disabled=use_bundled,
        help=f"Needs the 36 feature columns and a '{TARGET}' column "
             f"({', '.join(CLASSES)}). Comma- or semicolon-separated.",
    )

# ---------------------------------------------------------------- load + validate
if use_bundled:
    source_name, raw = BUNDLED_CSV.name, pd.read_csv(BUNDLED_CSV)
elif uploaded is None:
    st.info("Upload a CSV in the sidebar, or tick **Use bundled test_data.csv**.")
    st.stop()
else:
    source_name = uploaded.name
    if not uploaded.getvalue().strip():
        st.error("The uploaded file is empty.")
        st.stop()
    try:
        # sep=None lets pandas detect ',' or ';' (the raw UCI file uses ';').
        raw = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
    except (csv.Error, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the file as CSV: {exc}")
        st.stop()

# Codes the one-hot encoder learned in training (identical in all five pipelines).
encoder = next(iter(models.values())).named_steps["prep"].named_transformers_["nominal"]
known = dict(zip(encoder.feature_names_in_, encoder.categories_))

try:
    X, y, infos, warnings = validate_upload(raw, known_categories=known)
except UploadError as exc:
    st.error(str(exc))
    st.stop()

for msg in infos:
    st.info(msg)
for msg in warnings:
    st.warning(msg)

tab_eval, tab_compare = st.tabs(["Evaluate selected model", "Compare all models"])

# ---------------------------------------------------------------- tab 1
with tab_eval:
    st.subheader(f"{model_name} on {source_name}")

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**Preview** ({len(X)} rows)")
        preview = X.assign(**{TARGET: y}).head(10)
        if st.toggle("Show code meanings instead of numbers", value=True):
            preview = decode(preview, dictionary)
        st.dataframe(preview, hide_index=True)
    with right:
        st.markdown("**True class counts**")
        counts = y.value_counts().reindex(CLASSES, fill_value=0).rename("rows")
        st.dataframe(counts, width="stretch")

    y_pred, m = evaluate(models[model_name], X, y)

    cards = st.columns(len(METRIC_NAMES))
    for card, name in zip(cards, METRIC_NAMES):
        help_text = m.get("auc_note") if name == "AUC" else None
        card.metric(name, fmt(m[name]), help=help_text, border=True)
    if m["AUC"] is None:
        st.caption(f"AUC n/a: {m['auc_note']}.")

    cm_col, report_col = st.columns(2)
    with cm_col:
        st.markdown("**Confusion matrix**")
        st.altair_chart(confusion_matrix_chart(y, y_pred, CLASSES), width="stretch")
    with report_col:
        st.markdown("**Classification report**")
        st.dataframe(classification_report_frame(y, y_pred, CLASSES))

    st.markdown(
        """
**Reading the matrix.** Rows are the true outcome, columns the prediction.
- *Dropout* row, predicted *Graduate*: at-risk students the model misses, the costliest error
  for a student-support office, because these students would get no outreach.
- *Enrolled* is the smallest class (about 18% of students) and sits between the other two:
  these students are neither finished nor gone, so models confuse it with both neighbours.
  Check its row in the report: a model can over-predict it (high recall, low precision) or
  almost ignore it (low recall). Either way macro F1 drops even when accuracy looks fine.
- *Graduate* predicted as *Dropout* costs less: an unneeded check-in with a student who was
  doing well.
"""
    )

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader(f"All five models on {source_name}")
    rows = {name: evaluate(model, X, y)[1] for name, model in models.items()}
    table = pd.DataFrame(rows).T[METRIC_NAMES]
    st.dataframe(table.map(fmt), width="stretch")

    st.markdown("**Macro F1 (primary criterion)**")
    f1 = table["F1"].astype(float).rename_axis("Model").reset_index()
    # horizontal=True swaps the axes, so x/y here still mean category/value.
    st.bar_chart(f1, x="Model", y="F1", horizontal=True, sort="-F1",
                 x_label="", y_label="Macro F1")

    st.markdown(best_model_sentence(table["F1"].astype(float)))
