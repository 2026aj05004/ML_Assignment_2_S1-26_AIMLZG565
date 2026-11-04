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

# Plain-language explanations, used for tooltips and the "How the metrics work" section.
METRIC_HELP = {
    "Accuracy": "Share of students whose outcome was predicted correctly. It can look good even "
                "when a model mostly ignores the small Enrolled group.",
    "AUC": "How well the model ranks students: 1 = perfect, 0.5 = no better than guessing. "
           "Computed for each outcome against the other two, then averaged. Needs all three "
           "outcomes in the file.",
    "Precision": "Of the students the model put in an outcome group, the share that really "
                 "belong there. Averaged over the three groups, each counting equally.",
    "Recall": "Of the students who really belong to an outcome group, the share the model "
              "found. Averaged over the three groups, each counting equally.",
    "F1": "A single score balancing precision and recall (their harmonic mean), averaged over "
          "the three groups. This is the main score used to compare the models.",
    "MCC": "Matthews correlation coefficient: agreement between predicted and actual outcomes, "
           "from -1 (always wrong) through 0 (no better than chance) to 1 (perfect). Stays "
           "fair when the groups have very different sizes.",
}

MODEL_HELP = {
    "Logistic Regression": "Gives each feature a weight and turns the weighted total into a "
                           "probability for each outcome. Simple, fast and easy to explain.",
    "Decision Tree": "Asks a sequence of yes/no questions about the features and predicts from "
                     "the answers. The trained tree's first question is \"approved 4 or fewer "
                     "units in semester 2?\"",
    "K-Nearest Neighbors": "Finds the students in the training data most similar to this one "
                           "and predicts the outcome most common among them.",
    "Naive Bayes (Gaussian)": "Uses probability rules, assuming every feature is independent "
                              "of the others and follows a bell curve within each outcome group. "
                              "Fast, but those assumptions do not hold well here.",
    "SVM (RBF kernel)": "A Support Vector Machine draws the boundary between outcome groups with "
                        "the widest possible gap. The RBF (radial basis function) kernel lets that "
                        "boundary curve, by judging how similar two students are from the "
                        "distance between their feature values.",
}

PARAM_HELP = {
    "C": "C: how closely the model fits the training data. Smaller values keep it simpler, "
         "which helps it generalise to new students.",
    "max_depth": "max_depth: the most yes/no questions the tree may ask in a row.",
    "min_samples_leaf": "min_samples_leaf: the fewest training students allowed at the end "
                        "of a branch.",
    "n_neighbors": "n_neighbors: how many similar students are consulted.",
    "weights": "weights = distance: closer (more similar) students count more.",
    "var_smoothing": "var_smoothing: a small amount added to every spread estimate, so rare "
                     "values do not produce extreme probabilities.",
}

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
    return "n/a" if value is None else f"{value:.4f}"


def best_model_sentence(f1: pd.Series) -> str:
    """Name the best model by macro F1, and say so when the runner-up is effectively tied.

    Scores differing by less than 0.001 are too close to rank, so they are called a tie.
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
    "Five machine-learning models predict whether a university student will **drop out**, "
    "**still be enrolled** or **graduate**. See how well each one does on real student records."
)

# ---------------------------------------------------------------- about (top of page)
with st.expander("About the dataset", expanded=True):
    data_col, class_col, use_col = st.columns([5, 4, 4], gap="large")
    with data_col:
        st.markdown(
            f"""
**4,424 students** at a higher-education institution in Portugal, one row each, with no
missing values ([UCI dataset #697]({UCI_URL}), CC BY 4.0).

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

*Enrolled* means still studying after the normal duration. It is the smallest group and
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
2. Keep the bundled test file (885 students the models never saw during training), or
   upload your own CSV with the same columns and a known outcome for each student.
3. Read the scores and the confusion matrix, or compare all five models.
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

with st.expander("How the scores work"):
    st.markdown(
        "All scores are computed on the students in the file you evaluate. Hover over the "
        "ⓘ next to any score for the same explanation.\n\n"
        + "\n".join(f"- **{name}**: {text}" for name, text in METRIC_HELP.items())
        + f"""

Precision, Recall and F1 are **macro averages**: each outcome group counts equally, however
many students it has. If a file has no students in one group, that group scores 0, so these
three cannot go above 0.667 with only two groups present.

**F1** was chosen as the main score before training, because accuracy alone would reward a
model that rarely predicts *Enrolled*.

The models were trained once by `train.py` (scikit-learn {meta['sklearn_version']}); this
app only loads them and never retrains. Source code: [{REPO_URL}]({REPO_URL})
"""
    )

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Choose a model")
    model_name = st.selectbox(
        "Model", list(models),
        help="All five were trained on the same students and tuned the same way.",
    )
    st.caption(MODEL_HELP[model_name])
    params = meta["best_params"][model_name]
    st.caption(
        "Tuned settings: " + ", ".join(f"`{k} = {v}`" for k, v in params.items()),
        help="Chosen automatically by testing a few values on the training students.\n\n"
             + "\n\n".join(PARAM_HELP[k] for k in params),
    )

    st.header("2. Provide test data")
    use_bundled = st.checkbox(
        "Use bundled test_data.csv", value=True,
        help="885 students (20% of the data) set aside before training, keeping the same mix "
             "of outcomes. The models never saw them, so this is a fair test.",
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

    # Headings (and the toggle) share one row so the two tables below start level.
    head_left, head_toggle, head_right = st.columns([2, 2, 2], vertical_alignment="center")
    head_left.markdown(f"**Preview**: first 10 of {len(X)} students")
    show_meanings = head_toggle.toggle("Show code meanings", value=True,
                                       help="Show labels such as *Nursing* instead of codes such as 9500.")
    head_right.markdown("**Actual outcomes** in this file")
    left, right = st.columns([4, 2])
    preview = X.assign(**{TARGET: y}).head(10)
    left.dataframe(decode(preview, dictionary) if show_meanings else preview, hide_index=True)
    counts = y.value_counts().reindex(CLASSES, fill_value=0).rename("Students")
    right.dataframe(counts.rename_axis("Outcome"), width="stretch")

    y_pred, m = evaluate(models[model_name], X, y)

    # Two rows of three so 4-decimal scores fit at laptop widths: overall scores first,
    # then the per-group (macro) averages.
    for row in (["Accuracy", "AUC", "MCC"], ["Precision", "Recall", "F1"]):
        for card, name in zip(st.columns(3), row):
            help_text = METRIC_HELP[name]
            if name == "AUC" and m["AUC"] is None:
                help_text += f"\n\nNot available here: {m['auc_note']}."
            card.metric(name, fmt(m[name]), help=help_text, border=True)
    if m["AUC"] is None:
        st.caption(f"AUC n/a: {m['auc_note']}.")

    cm_col, report_col = st.columns(2)
    with cm_col:
        st.markdown(
            "**Confusion matrix**",
            help="Each cell counts students by their actual outcome (row) and the model's "
                 "prediction (column). The diagonal holds the correct predictions.",
        )
        st.altair_chart(confusion_matrix_chart(y, y_pred, CLASSES), width="stretch")
    with report_col:
        st.markdown(
            "**Scores per outcome group**",
            help="precision, recall and f1-score for each group (see the score tooltips above); "
                 "support = number of students actually in that group. *macro avg* weights every "
                 "group equally (the cards above); *weighted avg* weights groups by size.",
        )
        st.dataframe(
            classification_report_frame(y, y_pred, CLASSES),
            column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                           for c in ["precision", "recall", "f1-score"]},
        )

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
    st.dataframe(
        table.map(fmt), width="stretch",
        column_config={n: st.column_config.TextColumn(n, help=METRIC_HELP[n]) for n in METRIC_NAMES},
    )

    st.markdown("**F1 by model** (the main score used to compare them)",
                help=METRIC_HELP["F1"])
    f1 = table["F1"].astype(float).rename_axis("Model").reset_index()
    # horizontal=True swaps the axes, so x/y here still mean category/value.
    st.bar_chart(f1, x="Model", y="F1", horizontal=True, sort="-F1",
                 x_label="", y_label="Macro F1")

    st.markdown(best_model_sentence(table["F1"].astype(float)))
