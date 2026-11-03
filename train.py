"""Train the five classifiers and save everything the app needs.

    python train.py

Steps: download data -> stratified 80/20 split -> write test_data.csv ->
tune each pipeline with 5-fold CV on the training split (scoring = macro F1) ->
evaluate the refit best pipeline once on the hold-out -> save model/*.joblib,
model/metadata.json and model/metrics.json.
"""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import CLASSES, FEATURE_COLS, NOMINAL_COLS, NUMERIC_COLS, TARGET, load_raw
from metrics import METRIC_NAMES, compute_metrics

SEED = 42
TEST_SIZE = 0.2
MODEL_DIR = Path("model")
TEST_CSV = Path("test_data.csv")

# name -> (file stem, estimator, hyper-parameter grid)
# Class weighting is used where the estimator supports it; KNN and Gaussian NB have no such option.
MODELS = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        {"clf__C": [0.1, 1, 10]},
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"clf__max_depth": [4, 6, 8, None], "clf__min_samples_leaf": [1, 5, 20]},
    ),
    "K-Nearest Neighbors": (
        "knn",
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes (Gaussian)": (
        "naive_bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
    ),
    "SVM (RBF kernel)": (
        "svm",
        # An SVC outputs margins, not probabilities. Platt (sigmoid) calibration on
        # cross-validated margins gives the predict_proba that AUC needs; ensemble=False
        # keeps a single SVC refit on all training rows. (This replaces SVC(probability=True),
        # deprecated in scikit-learn 1.9.) No random_state: fitting an SVC this way is
        # deterministic, and the calibration folds are unshuffled.
        CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced"),
            method="sigmoid", ensemble=False,
        ),
        {"clf__estimator__C": [0.5, 1, 5]},
    ),
}


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode nominal codes, standardise everything else.

    sparse_output=False and sparse_threshold=0 force a dense array:
    GaussianNB raises a TypeError on sparse input.
    """
    return ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
            ("numeric", StandardScaler(), NUMERIC_COLS),
        ],
        sparse_threshold=0,
    )


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + classifier, so each saved file is self-contained."""
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def split_data(df):
    X, y = df[FEATURE_COLS], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def markdown_table(all_metrics: dict) -> str:
    lines = ["| Model | " + " | ".join(METRIC_NAMES) + " |",
             "|---" * (len(METRIC_NAMES) + 1) + "|"]
    for name, m in all_metrics.items():
        cells = ["n/a" if m[k] is None else f"{m[k]:.4f}" for k in METRIC_NAMES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df = load_raw()
    X_train, X_test, y_train, y_test = split_data(df)

    # The hold-out is saved for the app and never used during tuning.
    X_test.assign(**{TARGET: y_test}).to_csv(TEST_CSV, index=False)
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)} -> {TEST_CSV}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_metrics, best_params, files = {}, {}, {}

    for name, (stem, estimator, grid) in MODELS.items():
        search = GridSearchCV(build_pipeline(estimator), grid, scoring="f1_macro", cv=cv,
                              refit=True, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        classes = [str(c) for c in model.classes_]
        assert classes == CLASSES, classes  # predict_proba columns follow this order

        y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred, model.predict_proba(X_test), classes)
        m["cv_f1_macro"] = float(search.best_score_)
        all_metrics[name] = m
        best_params[name] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        files[name] = f"{stem}.joblib"
        joblib.dump(model, MODEL_DIR / files[name])

        print(f"\n{name}: best {best_params[name]}, CV macro F1 = {search.best_score_:.4f}")
        print(f"Confusion matrix (rows = actual {classes}):")
        print(confusion_matrix(y_test, y_pred, labels=classes))

    metadata = {
        "dataset": "UCI #697 Predict Students' Dropout and Academic Success",
        "target": TARGET,
        "classes": CLASSES,
        "feature_columns": FEATURE_COLS,
        "nominal_columns": NOMINAL_COLS,
        "numeric_columns": NUMERIC_COLS,
        "model_files": files,
        "best_params": best_params,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "tuning_scoring": "f1_macro",
        "primary_metric": "F1",
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")

    print("\nHold-out metrics (macro averages; primary criterion = F1):\n")
    print(markdown_table(all_metrics))


if __name__ == "__main__":
    main()
