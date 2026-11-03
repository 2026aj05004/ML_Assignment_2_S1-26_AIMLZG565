"""Download, cache and tidy the UCI "Predict Students' Dropout and Academic Success" dataset.

Shared by train.py, the notebook and app.py so that every part of the project
uses exactly the same column names and column groups.
"""

import io
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)

TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

# Integer-coded categories with no natural order -> one-hot encoded.
NOMINAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",  # spelling as in the original dataset
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Everything else: binary 0/1 flags and continuous values -> standardised.
# (Scaling a 0/1 flag is harmless and keeps the preprocessing to two groups.)
NUMERIC_COLS = [
    "Application order",
    "Daytime/evening attendance",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

FEATURE_COLS = NOMINAL_COLS + NUMERIC_COLS


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with tidy column names.

    The UCI file starts with a byte-order mark and has one header with a trailing
    tab ("Daytime/evening attendance\\t"). Removing the BOM, trimming whitespace and
    collapsing repeated spaces means a raw UCI export and our own test_data.csv
    end up with identical column names.
    """
    out = df.copy()
    out.columns = [
        re.sub(r"\s+", " ", str(c).replace("﻿", "")).strip() for c in out.columns
    ]
    return out


class UploadError(ValueError):
    """An uploaded CSV cannot be evaluated; the message is shown to the user."""


def validate_upload(df: pd.DataFrame, known_categories: dict | None = None):
    """Check an uploaded test CSV and return (X, y, infos, warnings).

    Raises UploadError for problems the user has to fix in the file. The
    saved pipelines have no imputer, so missing or non-numeric values are
    rejected rather than guessed.

    known_categories (column -> codes seen in training) lets us tell the user
    about category codes the one-hot encoder has never seen; it encodes them
    as all zeros, so the model still predicts but with less information.
    """
    if len(df) == 0:
        raise UploadError("The file has a header but no data rows.")

    df = normalise_columns(df)
    infos, warnings = [], []

    # e.g. "Target" and "Target " both become "Target" after cleanup
    dupes = sorted(set(df.columns[df.columns.duplicated()]))
    if dupes:
        raise UploadError(
            "These columns appear more than once (after trimming spaces): " + ", ".join(dupes)
        )

    if TARGET not in df.columns:
        raise UploadError(
            f"Column '{TARGET}' is missing. The app needs the true labels to compute metrics."
        )

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise UploadError(
            f"{len(missing)} required feature column(s) missing: " + ", ".join(missing)
        )

    extra = [c for c in df.columns if c not in FEATURE_COLS and c != TARGET]
    if extra:
        infos.append("Ignored extra column(s): " + ", ".join(extra))

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    # NaN = empty or non-numeric; +-inf comes from values like "inf" or 1e400.
    invalid = X.isna() | np.isinf(X)
    bad_cols = X.columns[invalid.any()].tolist()
    if bad_cols:
        n_rows = int(invalid.any(axis=1).sum())
        raise UploadError(
            f"{n_rows} row(s) have empty, non-numeric or infinite values in: "
            f"{', '.join(bad_cols)}. The models have no imputer, so please clean these "
            "rows first. (Numbers must use '.' as the decimal point.)"
        )

    if df[TARGET].isna().any():
        raise UploadError(f"{int(df[TARGET].isna().sum())} row(s) have an empty '{TARGET}'.")
    y = df[TARGET].astype(str).str.strip()
    unknown = sorted(set(y) - set(CLASSES))
    if unknown:
        raise UploadError(
            f"Unknown label(s) in '{TARGET}': {', '.join(unknown)}. "
            f"Expected only: {', '.join(CLASSES)}."
        )

    if known_categories:
        unseen = {c: int((~X[c].isin(known_categories[c])).sum()) for c in known_categories}
        unseen = {c: n for c, n in unseen.items() if n}
        if unseen:
            infos.append(
                "Category codes not seen in training (encoded as 'unknown', so the model "
                "uses the other features for these rows): "
                + ", ".join(f"{c} ({n} row(s))" for c, n in unseen.items())
            )

    present = [c for c in CLASSES if c in set(y)]
    if len(present) < len(CLASSES):
        absent = [c for c in CLASSES if c not in present]
        warnings.append(
            f"No rows with label {', '.join(absent)}: AUC is shown as n/a. Precision, "
            "Recall and F1 are still macro averages over all three classes, so the absent "
            f"class counts as 0 and their maximum here is {len(present) / len(CLASSES):.3f}."
        )

    return X, y, infos, warnings


def load_raw(cache_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load the full dataset (4,424 rows), downloading the UCI zip on first use."""
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / "data.csv"
    if not csv_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # certifi's CA bundle: python.org builds of Python on macOS ship without
        # root certificates, so the default context fails to verify UCI's HTTPS.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, timeout=60, context=context) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_path.write_bytes(archive.read("data.csv"))

    # The file is ';'-separated, not ','.
    df = normalise_columns(pd.read_csv(csv_path, sep=";", encoding="utf-8-sig"))

    expected = set(FEATURE_COLS + [TARGET])
    if set(df.columns) != expected:
        raise ValueError(
            f"Unexpected columns. Missing: {sorted(expected - set(df.columns))}; "
            f"extra: {sorted(set(df.columns) - expected)}"
        )
    return df
