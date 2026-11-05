# Predicting student dropout and academic success

ML Assignment 2: five classifiers compared on one public dataset, with an interactive
Streamlit app that evaluates the saved models on an uploaded test file.

- **Live app:** <https://2026aj05004-aimlzg565-assignment-2.streamlit.app/>
- **Repository:** <https://github.com/2026aj05004/ML_Assignment_2_S1-26_AIMLZG565>

---

## a. Problem statement

A higher-education institution wants to know, for each student, which of three outcomes they
are heading towards at the end of the normal duration of their course:

| Class | Meaning |
|---|---|
| **Dropout** | left the course |
| **Enrolled** | still studying after the normal duration (neither graduated nor dropped out) |
| **Graduate** | finished the course |

This is a **three-class classification** problem.

**Prediction point.** The features include the student's results for the **first and second
semesters**, so a prediction can only be made **at the end of the first academic year**, not
at admission. A student-support office can still contact students predicted to drop out while
two or more years of the course remain.

## b. Dataset description

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset #697, *Predict Students' Dropout and Academic Success* |
| Rows | 4,424 students (one row each), no missing values |
| Features | 36 (all numeric codes or numbers) + `Target` |
| Classes | Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |
| Licence | CC BY 4.0 |

The data comes from a higher-education institution in Portugal and covers undergraduate degrees
such as agronomy, design, education, nursing, journalism, management and technologies.
The 36 features fall into five groups:

- **Enrolment and background** (known at admission): marital status, application mode and order,
  course, attendance, previous qualification and its grade, nationality, admission grade, age,
  gender, and flags for displaced, international and special-needs students.
- **Family**: mother's and father's qualification and occupation.
- **Financial**: debtor, tuition fees up to date, scholarship holder.
- **Academic progress** (first year): for each semester, curricular units credited, enrolled,
  evaluated, approved and without evaluation, and the average grade.
- **Macro-economic**: unemployment rate, inflation rate and GDP.

Nine features are **integer codes for unordered categories** (for example `Course = 9500` is
Nursing). Enrolled is the smallest class and, for every model, the hardest to predict.

Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students'
Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5MC89>

## c. GitHub repository

<https://github.com/2026aj05004/ML_Assignment_2_S1-26_AIMLZG565>

```
app.py                Streamlit app (entry point on Streamlit Community Cloud)
train.py              data -> split -> tune 5 pipelines -> save model/, test_data.csv
data_loader.py        UCI download, column names and groups, upload checks
metrics.py            the six metrics and the classification report
model/                5 saved pipelines (*.joblib), metadata.json, metrics.json
data/                 feature_dictionary.json: UCI column descriptions and code meanings
test_data.csv         20% hold-out test set (885 rows, 36 features + Target)
notebooks/            EDA and training walkthrough (runs in any Jupyter environment)
requirements.txt      pinned packages for the app and the notebook
```

## d. Models used

### Method

1. **Split.** A stratified 80/20 split (`random_state=42`) gives 3,539 training rows and an
   885-row hold-out, saved as `test_data.csv` and **not used during tuning**.
2. **Preprocessing.** Inside every pipeline, the nine category codes are one-hot encoded and the
   other 27 columns are standardised, so each saved pipeline takes the raw columns directly.
3. **Tuning.** `GridSearchCV` with a small grid per model and the same
   `StratifiedKFold(5, shuffle=True, random_state=42)` for all five, scored by macro F1 on the
   training split. `class_weight="balanced"` is used where supported (Logistic Regression,
   Decision Tree, SVM).
4. **Evaluation.** Each tuned pipeline is evaluated **once** on the hold-out.

| Model | Grid searched | Chosen |
|---|---|---|
| Logistic Regression | `C` in {0.1, 1, 10} | `C=0.1` |
| Decision Tree | `max_depth` in {4, 6, 8, None} x `min_samples_leaf` in {1, 5, 20} | `max_depth=6`, `min_samples_leaf=1` |
| K-Nearest Neighbors | `n_neighbors` in {5, 11, 21} x `weights` in {uniform, distance} | `k=5`, `distance` |
| Naive Bayes (Gaussian) | `var_smoothing` in {1e-9, 1e-6, 1e-3} | `1e-3` |
| SVM (RBF kernel) | `C` in {0.5, 1, 5} | `C=1` |

### Metrics

All metrics are computed on the hold-out (885 students):

- **Accuracy**: share of students whose outcome is predicted correctly.
- **AUC**: how well the predicted probabilities rank students, as the macro average of the
  one-vs-rest ROC AUC for each class (0.5 = chance, 1 = perfect).
- **Precision, Recall, F1**: **macro** averages, so the small Enrolled class counts as much as Graduate.
- **MCC**: Matthews correlation coefficient (multiclass form); 0 = chance, 1 = perfect.

**Primary comparison criterion: macro F1**, fixed before training, because accuracy alone would
reward a model that rarely predicts Enrolled.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7367 | **0.8838** | **0.7094** | **0.7158** | **0.7014** | 0.5937 |
| Decision Tree | 0.6486 | 0.8240 | 0.6660 | 0.6580 | 0.6314 | 0.4929 |
| K-Nearest Neighbors | 0.7096 | 0.8013 | 0.6494 | 0.6130 | 0.6216 | 0.5147 |
| Naive Bayes (Gaussian) | 0.5842 | 0.7468 | 0.5499 | 0.5427 | 0.5395 | 0.3504 |
| SVM (RBF kernel) | **0.7480** | 0.8644 | 0.7022 | 0.7041 | 0.7012 | **0.5957** |

(Source: `model/metrics.json`, written by `train.py`. Best value in each column in bold.)

### Observations

| ML Model Name | Observation about model performance |
|------|------------------------------|
| Logistic Regression | Logistic Regression achieved the highest macro F1 (0.7014) and AUC (0.8838). It identified 106 of 159 Enrolled students, although precision for this class was only 0.43. |
| Decision Tree | The Decision Tree had the highest Enrolled recall (0.71) but the second-lowest accuracy (0.6486). It labelled 332 students as Enrolled, including 134 graduates, so Graduate recall fell to 0.65. |
| K-Nearest Neighbors | KNN performed better on accuracy (0.7096) than on macro F1 (0.6216), reflecting its weak Enrolled recall (0.29, 46 of 159). It also predicted 65 of 284 dropouts as Graduate, the most of any model. |
| Naive Bayes (Gaussian) | Gaussian Naive Bayes scored lowest on every metric (macro F1 0.5395, MCC 0.3504). Its assumption of independent, normally distributed features does not hold here: 1st- and 2nd-semester units approved correlate at 0.90, and most inputs are 0/1 one-hot columns. |
| SVM (RBF kernel) | The SVM had the highest accuracy (0.7480) and MCC (0.5957) and the best Dropout recall (0.74, 210 of 284). Its AUC (0.8644) was lower than Logistic Regression's, so its probabilities ranked students less well. |
| **Overall winner** | **Logistic Regression**: highest macro F1 and AUC, and the simplest model. The SVM's macro F1 (0.7012) was almost identical on this test set, with slightly higher accuracy and MCC. |

**Conclusion.** The macro F1 scores of Logistic Regression and the SVM were almost identical on
this test set (difference 0.0002), so the choice rests on the other criteria: Logistic Regression
has the higher AUC and is simpler to explain. All five models found **Enrolled** the hardest
class: the first-year data separates "will finish" from "will leave" much better than it
separates "still studying" from either.

## The Streamlit app

`app.py` loads the five saved pipelines from `model/` (it never retrains):

- **Upload** a test CSV with the 36 feature columns and `Target`, or tick the box to use the
  bundled `test_data.csv` (the hold-out). Files with problems are rejected with a message.
- **Choose a model** from the dropdown in the sidebar.
- **Results**: the six metrics, the confusion matrix and the classification report for that
  model, plus a **Compare all models** tab ranking all five on the same file by macro F1.

Use `test_data.csv` for a fair evaluation: a full UCI download also contains the training rows.

## Running locally

Python 3.12 with the pinned versions (a `.joblib` model only loads reliably with the
scikit-learn version that saved it, 1.9.1):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # uses the committed models
python train.py               # optional: re-download the data and retrain (about 1 minute)
```

`train.py` is deterministic: re-running it reproduces `model/metrics.json` and `test_data.csv`.
