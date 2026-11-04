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
semesters**. A prediction can therefore only be made **at the end of the first academic year**,
not at admission. That is still useful: a student-support office can contact students
predicted to drop out while two or more years of the course remain.

## b. Dataset description

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset #697, *Predict Students' Dropout and Academic Success* |
| Rows | 4,424 students (one row each), no missing values |
| Features | 36 (all numeric codes or numbers) + `Target` |
| Classes | Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |
| Licence | CC BY 4.0 |

The data comes from a higher-education institution in Portugal and covers students in undergraduate
degrees such as agronomy, design, education, nursing, journalism, management, social service
and technologies.

The 36 features fall into five groups:

- **Enrolment and background** (known at admission): marital status, application mode and order,
  course, daytime/evening attendance, previous qualification and its grade, nationality,
  admission grade, age at enrolment, displaced, international, educational special needs, gender.
- **Family**: mother's and father's qualification and occupation.
- **Financial**: debtor, tuition fees up to date, scholarship holder.
- **Academic progress** (first year): for each of the two semesters, curricular units credited,
  enrolled, evaluated, approved, without evaluation, and the average grade.
- **Macro-economic**: unemployment rate, inflation rate and GDP.

Nine of the features are **integer codes for categories with no order** (for example `Course = 9500`
is Nursing): marital status, application mode, course, previous qualification, nationality, and
the four parent qualification and occupation columns. These are one-hot encoded. The other 27
features are standardised, including the 0/1 flags (scaling a 0/1 flag does no harm).

**Class imbalance.** Enrolled is the smallest class (17.9%) and also the hardest. These students are
"in between": still studying, but not on track to finish on time. It is the class every model
handles worst (see below).

Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students'
Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5MC89>

## c. GitHub repository

<https://github.com/2026aj05004/ML_Assignment_2_S1-26_AIMLZG565>

```
app.py                Streamlit app (entry point on Streamlit Community Cloud)
train.py              data -> split -> tune 5 pipelines -> save model/, test_data.csv
data_loader.py        UCI download, column names and groups, upload checks
metrics.py            six metrics, confusion-matrix figure, classification report
model/                5 saved pipelines (*.joblib), metadata.json, metrics.json
data/                 feature_dictionary.json: UCI column descriptions and code meanings
test_data.csv         20% hold-out test set (885 rows, 36 features + Target)
notebooks/            EDA and training walkthrough
requirements.txt      pinned packages the app needs
requirements-dev.txt  adds Jupyter for the notebook
```

## d. Models used

### Method

1. **Split.** A stratified 80/20 train/test split (`random_state=42`) gives 3,539 training rows
   and an 885-row hold-out. The hold-out is saved as `test_data.csv` and is **not used during tuning**.
2. **Preprocessing.** The same steps inside every pipeline: one-hot encoding of the nine nominal
   codes (`handle_unknown="ignore"`) and standardisation of the other 27 columns, giving 236 inputs.
   The output is a dense matrix because Gaussian Naive Bayes does not accept sparse input.
   Because preprocessing lives inside each saved pipeline, the app needs no separate step.
3. **Tuning.** `GridSearchCV` with a small grid per model and the same seeded
   `StratifiedKFold(5, shuffle=True, random_state=42)` for all five, scored by macro F1 on the
   training split only. `class_weight="balanced"` is used where the estimator supports it
   (Logistic Regression, Decision Tree, SVM). KNN and Gaussian NB have no such option. So the
   comparison is between five *complete configured pipelines*, not bare algorithms.
4. **Evaluation.** Each tuned pipeline is evaluated **once** on the hold-out.

| Model | Grid searched | Chosen |
|---|---|---|
| Logistic Regression | `C` in {0.1, 1, 10} | `C=0.1` |
| Decision Tree | `max_depth` in {4, 6, 8, None} x `min_samples_leaf` in {1, 5, 20} | `max_depth=6`, `min_samples_leaf=1` |
| K-Nearest Neighbors | `n_neighbors` in {5, 11, 21} x `weights` in {uniform, distance} | `k=5`, `distance` |
| Naive Bayes (Gaussian) | `var_smoothing` in {1e-9, 1e-6, 1e-3} | `1e-3` |
| SVM (RBF kernel) | `C` in {0.5, 1, 5} | `C=1` |

The SVM's probabilities (needed for AUC) come from sigmoid (Platt) calibration on
cross-validated decision values (`CalibratedClassifierCV(SVC(...), ensemble=False)`).
This replaces `SVC(probability=True)`, which is deprecated in scikit-learn 1.9.

### Metrics

All metrics are computed on the hold-out (885 students):

- **Accuracy**: share of students whose outcome is predicted correctly.
- **AUC**: macro-averaged one-vs-rest ROC AUC over the three classes, from `predict_proba`.
- **Precision, Recall, F1**: **macro** averages. Each class counts equally, so the small Enrolled
  class matters as much as Graduate.
- **MCC**: Matthews correlation coefficient (multiclass form). 0 means no better than chance; 1 means perfect.

**Primary comparison criterion: macro F1**, fixed before training. Accuracy alone would reward
a model that rarely predicts Enrolled.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7367 | **0.8838** | **0.7094** | **0.7158** | **0.7014** | 0.5937 |
| Decision Tree | 0.6486 | 0.8240 | 0.6660 | 0.6580 | 0.6314 | 0.4929 |
| K-Nearest Neighbors | 0.7096 | 0.8013 | 0.6494 | 0.6130 | 0.6216 | 0.5147 |
| Naive Bayes (Gaussian) | 0.5842 | 0.7468 | 0.5499 | 0.5427 | 0.5395 | 0.3504 |
| SVM (RBF kernel) | **0.7480** | 0.8644 | 0.7022 | 0.7041 | 0.7012 | **0.5957** |

(Source: `model/metrics.json`, written by `train.py`. Best value in each column in bold.)

Hold-out confusion matrices (rows = actual, columns = predicted, order Dropout / Enrolled / Graduate):

| Model | Dropout row | Enrolled row | Graduate row |
|---|---|---|---|
| Logistic Regression | 195 / 65 / 24 | 25 / 106 / 28 | 16 / 75 / 351 |
| Decision Tree | 175 / 85 / 24 | 20 / 113 / 26 | 22 / 134 / 286 |
| K-Nearest Neighbors | 185 / 34 / 65 | 32 / 46 / 81 | 18 / 27 / 397 |
| Naive Bayes (Gaussian) | 153 / 94 / 37 | 25 / 66 / 68 | 62 / 82 / 298 |
| SVM (RBF kernel) | 210 / 44 / 30 | 31 / 87 / 41 | 17 / 60 / 365 |

### Observations

| ML Model Name | Observation about model performance |
|------|------------------------------|
| Logistic Regression | Best macro F1 (0.701) and best AUC (0.884), so its probabilities rank students best. Class weighting makes it predict Enrolled generously: Enrolled recall 0.67 but precision only 0.43 (140 Dropout or Graduate students labelled Enrolled). It misses only 24 of 284 dropouts as Graduate (8%), joint fewest with the Decision Tree. Its 5-fold CV F1 (0.714) is close to the hold-out F1, so the result is stable. |
| Decision Tree | A depth-6 tree, the most over-predicting of Enrolled: it labels 332 students Enrolled, including 134 actual graduates. Graduate recall therefore falls to 0.65 and accuracy to 0.649, the second-lowest. Class weighting plus coarse axis-aligned splits buys Enrolled recall (0.71, the highest) at a large cost elsewhere. |
| K-Nearest Neighbors | Third-highest accuracy (0.710) but second-lowest macro F1 (0.622). This is the clearest case of accuracy misleading. With no class weighting, the majority Graduate class wins the neighbour votes: Enrolled recall is only 0.29 (46 of 159), with 81 Enrolled students predicted Graduate, while Graduate recall is 0.90. It also labels 65 dropouts (23%) as Graduate, the worst outcome for a support office. Distances are also computed over 236 inputs, 209 of them one-hot category columns, which likely dilutes the academic-progress signal. |
| Naive Bayes (Gaussian) | Lowest on every metric (F1 0.540, MCC 0.350). Its two assumptions are both broken here. Features are far from independent given the class: 1st- and 2nd-semester units approved correlate at 0.90, units enrolled at 0.94. And the 209 one-hot 0/1 columns are modelled as Gaussians. The largest smoothing value in the grid (1e-3) was chosen, which tempers the extreme likelihoods from near-constant one-hot columns but does not fix the assumptions. It confuses all three classes, including 62 graduates predicted as Dropout. |
| SVM (RBF kernel) | Effectively tied with Logistic Regression on macro F1 (0.7012 vs 0.7014), with the highest accuracy (0.748) and MCC (0.596). It catches the most dropouts (recall 0.74, 210 of 284) and over-predicts Enrolled less than the other weighted models (Enrolled precision 0.46). Its AUC (0.864) is lower than Logistic Regression's, so its calibrated probabilities rank students less well. Its saved file is also much larger (4.5 MB vs 14 KB). |

**Conclusion.** On the primary criterion, Logistic Regression and the SVM are tied (a 0.0002
difference in macro F1 is not meaningful on 885 rows). **Logistic Regression is the preferred
model**: it has the better AUC and CV F1, ties for the fewest dropouts mislabelled as Graduate
(24 of 284), and is far simpler and smaller. The trade-off is that the SVM is slightly more accurate overall and flags
more of the actual dropouts. All five models find **Enrolled** the hardest class (per-class F1
0.33 to 0.52). The first-year data separates "will finish" from "will leave" much better than
it separates "still studying" from either.

## The Streamlit app

`app.py` loads the five saved pipelines from `model/` and **never retrains**. It provides:

- a **dataset overview** at the top (why the data exists, the five feature groups, the
  outcome classes, when the prediction is made) and a **feature dictionary** explaining every
  column and code, taken from the UCI documentation
- a **CSV upload** (or a checkbox to use the bundled `test_data.csv`)
- a **model dropdown**, with a plain-language description of each model (including what the
  SVM's RBF kernel does) and of its tuned settings
- the **six metrics** for the chosen model, each with a hover tooltip (ⓘ) explaining it for
  non-specialists
- the **confusion matrix** and **classification report**, with an option to show code
  meanings (e.g. *Nursing*) instead of numbers in the data preview
- a **Compare all models** tab showing all five models on the same file, ranked by macro F1

Uploads are checked before evaluation, and each problem gets a clear message: empty file,
missing `Target` or feature columns, non-numeric, empty or infinite cells, empty or unknown labels.
If a file lacks one of the three classes, AUC is shown as *n/a* (one-vs-rest AUC needs every
class). Precision, Recall and F1 stay macro averages over all three classes, so the absent class
counts as 0: on a two-class file their maximum is 0.667, and the app says so. Category codes
never seen in training (the hold-out itself has a few rare ones) are reported too; the one-hot
encoder treats them as unknown and the model uses the remaining features. Both `,` and `;` separators work, so a
raw UCI download can be uploaded directly. The app has light and dark themes; it follows the
visitor's system setting, and the theme can be changed in the Settings menu (top right). Note that a full UCI download contains the
training rows too, so its scores are optimistic. Use `test_data.csv` for a fair evaluation.

## Running locally

Python 3.12 (the Streamlit Community Cloud default). The pinned versions matter: a `.joblib`
model only loads reliably with the scikit-learn version that saved it (1.9.1).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # uses the committed models
python train.py               # optional: re-download the data and retrain (about 1 minute)
```

`train.py` is deterministic: re-running it reproduces `model/metrics.json` and `test_data.csv`
exactly. The raw dataset is downloaded to `data/raw/` and is not committed.
