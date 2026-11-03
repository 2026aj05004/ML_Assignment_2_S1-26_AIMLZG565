"""Evaluation metrics shared by train.py, the notebook and app.py.

Definitions (also stated in the README and in the app):
- Accuracy: fraction of rows predicted correctly.
- AUC: macro-averaged one-vs-rest ROC AUC over the three classes, from predict_proba.
- Precision, Recall, F1: macro averages over the three classes (each class counts equally).
- MCC: Matthews correlation coefficient, multiclass form.
Primary comparison criterion: macro F1.
"""

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are saved or handed to Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba, classes) -> dict:
    """Return the six metrics as plain floats.

    y_proba must have one column per class, in the same order as `classes`
    (i.e. the fitted pipeline's classes_). If y_true does not contain every
    class, ROC AUC is undefined for the missing class: AUC is then None and
    "auc_note" explains why. The other five metrics are still computed over
    the full label set.
    """
    classes = list(classes)
    y_true = np.asarray(y_true)
    avg = dict(labels=classes, average="macro", zero_division=0)

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": None,
        "Precision": precision_score(y_true, y_pred, **avg),
        "Recall": recall_score(y_true, y_pred, **avg),
        "F1": f1_score(y_true, y_pred, **avg),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    present = [c for c in classes if c in set(y_true)]
    if len(present) == len(classes):
        result["AUC"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        )
    else:
        missing = [c for c in classes if c not in present]
        result["auc_note"] = (
            f"only {' and '.join(present) or 'no known class'} present "
            f"(missing: {', '.join(missing)}); one-vs-rest AUC needs every class"
        )

    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in result.items()}


def confusion_matrix_figure(y_true, y_pred, classes, title: str = ""):
    """Heatmap of the confusion matrix; rows = true class, columns = predicted."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def classification_report_frame(y_true, y_pred, classes) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support as a DataFrame."""
    report = classification_report(
        y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).T.round(3)
