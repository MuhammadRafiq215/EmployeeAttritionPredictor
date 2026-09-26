# EmployeeAttritionPredictor 🔍

**Predicting employee attrition with interpretable machine learning — built for HR decision-making, not just accuracy leaderboards.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

Employee turnover is expensive — recruiting, onboarding, and lost productivity
can cost far more than retaining an existing employee. This project builds an
end-to-end machine learning pipeline that predicts which employees are at risk
of leaving, using an HR dataset of 800 employees (demographics, compensation,
tenure, and work-condition attributes).

Rather than optimizing for accuracy alone — which is misleading on this kind
of imbalanced dataset (~83% stay, ~17% leave) — this project focuses on the
metrics that actually matter for an HR early-warning system: **recall,
precision, F1-score, and ROC-AUC**, alongside a transparent, business-facing
interpretation of *why* the model flags certain employees as at risk.

## ✨ Key Features

- **Full EDA pipeline** — structure, summary statistics, missing-value and
  outlier checks (IQR method), and class-imbalance analysis.
- **Clean preprocessing** — ordinal encoding for ordered categories, one-hot
  encoding for nominal categories, stratified 80/20 train-test split.
- **Two complementary models** — Logistic Regression (interpretable baseline)
  and Random Forest (non-linear, feature-importance ranking), both trained
  with class-balanced weighting to address the imbalance.
- **Rigorous evaluation** — accuracy, precision, recall, F1-score, confusion
  matrices, and ROC/AUC curves, with an explicit discussion of *why accuracy
  alone is misleading* on imbalanced targets.
- **Feature interpretation** — Random Forest importances and Logistic
  Regression coefficients translated into concrete HR recommendations.
- **Fully reproducible** — a single executed Jupyter notebook with all code,
  outputs, and visualizations included.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.681 | 0.306 | **0.704** | 0.427 | **0.736** |
| Random Forest | **0.844** | **0.625** | 0.185 | 0.286 | 0.649 |

> **Takeaway:** Random Forest scores higher on raw accuracy but catches only
> ~18% of employees who actually leave. Logistic Regression trades some
> precision for far higher recall (70%) and the better ROC-AUC — making it the
> more useful model for an early-warning use case, and a clear illustration of
> why accuracy is the wrong metric to optimize for on imbalanced data.

**Top predictors** (agreed on by both models): `MonthlyIncome` and
`YearsAtCompany` — both negatively associated with attrition risk — along with
`OverTime`, `TrainingHoursLastYear`, `Age`, `DistanceFromHome_km`, and
department (`IT` / `Sales` / `R&D` show elevated risk).

## 🗂️ Repository Structure

```
.
├── employee_attrition_analysis.ipynb   # Full analysis notebook (code + outputs)
├── Attrition_Analysis_Report.md        # 1–2 page summary report
├── employee_attrition_dataset.csv      # Dataset (800 employees, 15 features)
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/EmployeeAttritionPredictor.git
cd EmployeeAttritionPredictor
pip install -r requirements.txt
```

### Usage

```bash
jupyter notebook employee_attrition_analysis.ipynb
```

Run all cells top to bottom to reproduce the full analysis, or open the
notebook to explore the pre-computed outputs and plots directly.

## 🧰 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` ·
`Jupyter`

## 🔍 Methodology

1. **Data Exploration & Preprocessing** — structure/type inspection, missing
   value and IQR-based outlier checks, class-imbalance analysis, categorical
   encoding, stratified train/test split.
2. **Model Building** — Logistic Regression and Random Forest, both trained
   with `class_weight="balanced"`.
3. **Model Evaluation** — full classification report, confusion matrices,
   ROC curves and AUC, with a dedicated discussion of imbalanced-metric
   interpretation.
4. **Feature Importance & Interpretation** — Random Forest importances and
   Logistic Regression coefficients, translated into HR-actionable insights.
5. **Conclusion** — model comparison, recommendation, and limitations.

## ⚠️ Limitations & Future Work

- Dataset size (800 rows, ~137 attrition cases) is modest — more data would
  improve reliability.
- Class imbalance was handled via `class_weight="balanced"` only; **SMOTE**
  and other resampling strategies are worth comparing.
- Findings are correlational, not causal.
- Planned improvements: hyperparameter tuning (`GridSearchCV`), Gradient
  Boosting / XGBoost / SVM comparisons, and SHAP-based interpretability.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE)
file for details.

## 🙋 Author

Built as a data science portfolio project demonstrating an end-to-end
classification workflow: EDA → preprocessing → modeling → imbalanced-data
evaluation → business interpretation.
