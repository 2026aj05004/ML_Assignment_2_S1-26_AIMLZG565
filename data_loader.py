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
