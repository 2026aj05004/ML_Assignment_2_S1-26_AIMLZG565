"""Streamlit app: evaluate the saved classifiers on an uploaded test CSV.

    streamlit run app.py

The app only loads the pipelines in model/ and calls predict / predict_proba.
It never trains or refits anything.
"""

import csv
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data_loader import CLASSES, TARGET, UploadError, validate_upload
from metrics import (
    METRIC_NAMES,
    classification_report_frame,
    compute_metrics,
    confusion_matrix_figure,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
BUNDLED_CSV = ROOT / "test_data.csv"
REPO_URL = "https://github.com/2026aj05004/ML_Assignment_2_S1-26_AIMLZG565"
UCI_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

st.set_page_config(page_title="Student Outcome Classifier", layout="wide")


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


meta, models = load_models()

st.title("Student outcome classifier")
st.caption(
    "Predicts whether a university student will **drop out**, **still be enrolled** or "
    "**graduate**, and lets you check how five trained models perform on labelled data."
)

# ---------------------------------------------------------------- about (top of page)
with st.expander("About the dataset", expanded=True):
    data_col, class_col, use_col = st.columns([5, 3, 4], gap="large")
    with data_col:
        st.markdown(
            f"""
**What it is.** Records of **4,424 students** from a higher-education institution in
Portugal, across degrees such as nursing, management, design, social service and
agronomy ([UCI dataset #697]({UCI_URL}), CC BY 4.0).

**36 features per student**, in five groups:
- **Enrolment and background**: course, application mode, previous qualification,
  admission grade, age, nationality, gender, ...
- **Family**: each parent's qualification and occupation
- **Financial**: tuition fees up to date, debtor, scholarship holder
- **First-year results**: units enrolled, evaluated and approved, and the average grade,
  for each of the two semesters
- **Economy**: unemployment rate, inflation rate and GDP

Most columns are numeric codes (for example `Course = 9500` is Nursing). The nationality
column is spelled *Nacionality*, as in the original dataset, so raw UCI files work unchanged.
"""
        )
    with class_col:
        st.markdown("**What is predicted**: the outcome at the end of the normal course duration")
        st.dataframe(
            pd.DataFrame(
                {"students": [1421, 794, 2209], "share": ["32.1%", "17.9%", "49.9%"]},
                index=pd.Index(CLASSES, name="Outcome"),
            ),
            width="stretch",
        )
        st.caption(
            "*Enrolled* means still studying after the normal duration. It is the smallest "
            "class and the hardest to predict."
        )
    with use_col:
        st.markdown(
            """
**When the prediction is made.** The features include first- and second-semester
results, so this is a prediction **at the end of the first year**, not at admission.
That still leaves time for a student-support office to contact students at risk.

**Using this app**
1. Pick a model in the sidebar.
2. Keep the bundled `test_data.csv` (885 students held out from training), or upload
   your own labelled CSV with the same columns.
3. Read the metrics and confusion matrix, or compare all five models.
"""
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
        st.dataframe(X.assign(**{TARGET: y}).head(10), hide_index=True)
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
        st.pyplot(confusion_matrix_figure(y, y_pred, CLASSES), width="content")
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

    best = table["F1"].astype(float).idxmax()
    st.markdown(f"Best on this file by macro F1: **{best}** ({table.loc[best, 'F1']:.3f}).")
