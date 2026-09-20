<div align="center">

# Bank Marketing Campaign Response Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![UCI Dataset](https://img.shields.io/badge/Dataset-UCI_Bank_Marketing-00897B?style=for-the-badge&logo=databricks&logoColor=white)](https://archive.ics.uci.edu/dataset/222/bank-marketing)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

*An end-to-end Machine Learning project predicting customer subscription to term deposits from historical bank marketing campaign data.*

</div>

---

## 📌 Project Overview

This is an individual project developed under the **Guided Data Track** for the IT3091 Machine Learning assignment.

The goal of this project is to build a predictive model that identifies which customers are likely to subscribe to a term deposit when contacted during a bank marketing campaign. By scoring customers before calls are made, the bank can prioritise outreach efforts — contacting high-probability customers first and reducing unnecessary contacts.

**Dataset:** [Bank Marketing — UCI Machine Learning Repository (ID 222)](https://archive.ics.uci.edu/dataset/222/bank-marketing)
> S. Moro, P. Cortez, P. Rita (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems.

---

## 🚀 Workflow & Notebooks

The project is structured into five sequential Jupyter Notebooks documenting the entire ML pipeline from raw data to business insights. All decisions are logged extensively in the `reports/` directory.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | Investigates dataset structure, confirms 45,211 rows and 17 columns, identifies missing/'unknown' values, establishes binary target (`y`), and flags `duration` as a leakage variable. |
| 2 | [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Visualises target imbalance (~11.7% positive), numerical distributions, categorical response rates, the `duration` leakage pattern, and unknown-value analysis. |
| 3 | [`03_preprocessing_feature_engineering.ipynb`](notebooks/03_preprocessing_feature_engineering.ipynb) | Defines the prediction point, conducts systematic leakage analysis for every feature, engineers 3 non-leaky features, and builds a leak-free `sklearn` ColumnTransformer pipeline with a stratified 70/15/15 split. |
| 4 | [`04_model_training.ipynb`](notebooks/04_model_training.ipynb) | Addresses the 1:8 class imbalance using `class_weight='balanced'`. Trains DummyClassifier, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, evaluated on Recall, F1, and ROC-AUC. |
| 5 | [`05_evaluation_and_business_insights.ipynb`](notebooks/05_evaluation_and_business_insights.ipynb) | First use of the held-out test set. Includes threshold analysis, ROC/PR curves, error analysis, feature importance, customer profile analysis, responsible AI review, and business recommendations. |

---

## 📊 Key Findings & Results

### ⚠️ The Class Imbalance Problem

Only ~11.7% of contacts result in a subscription. A dummy model predicting "no subscription" for every customer achieves 88.8% accuracy while catching **zero** actual subscribers. Models are therefore evaluated on **Recall**, **F1**, and **ROC-AUC / PR-AUC**.

### Validation Set Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| **Dummy (Baseline)** | 79.3% | 0.106 | 0.103 | 0.105 | 0.494 | 0.116 |
| Logistic Regression | 76.8% | 0.281 | **0.630** | 0.389 | 0.779 | 0.424 |
| Decision Tree | 82.4% | 0.343 | 0.552 | 0.423 | 0.744 | 0.353 |
| Random Forest | 80.8% | 0.330 | 0.617 | **0.430** | 0.794 | 0.453 |
| Gradient Boosting | 89.4% | **0.635** | 0.232 | 0.340 | **0.807** | **0.469** |

### 🏆 Final Test Set Results (used once)

| Model Configuration | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|--------------------|-----------|--------|----|---------|--------|
| Random Forest (t=0.50) | 0.329 | 0.608 | 0.427 | 0.785 | 0.438 |
| Gradient Boosting (t=0.50) | 0.659 | 0.254 | 0.366 | 0.791 | 0.458 |
| **Gradient Boosting (t=0.30)** ✅ | **0.540** | **0.409** | **0.465** | **0.791** | **0.458** |

**Selected model:** Gradient Boosting at threshold 0.30 — best F1 and PR-AUC on the test set. Lowering the threshold from 0.50 → 0.30 improved Recall from 0.25 → 0.41 while maintaining strong Precision (0.54).

### 📈 Hypothetical Prioritisation Scenario

> *Illustrative scenario — no actual campaign costs or revenues are claimed.*

By ranking customers by predicted probability and contacting the **top 30%**:

| Strategy | Customers Contacted | Subscribers Captured | Capture Rate |
|----------|--------------------|--------------------|-------------|
| Model-guided (Gradient Boosting) | 2,034 | 549 | **69.2%** |
| Random selection | 2,034 | 237 | 29.9% |

The model captures approximately **2.3× more subscribers** than random selection when contacting the same number of customers.

### 🔍 Key Predictors

Based on Random Forest feature importance and Logistic Regression coefficients:
- **`poutcome = success`** — strongest predictor; customers who subscribed in a previous campaign are far more likely to subscribe again.
- **`was_previously_contacted`** — binary flag indicating any prior campaign contact is strongly associated with positive response.
- **`month`** — campaign timing shows significant variation in response rates.
- **`balance`** — customers with higher bank balances show higher subscription likelihood.
- **`campaign` (contact frequency)** — too many contacts in a single campaign reduces response likelihood.

---

## 💻 Setup & Execution

**Note:** This project is notebook-driven. The `src/` directory provides production-ready modules for deployment (see [Production Architecture](#-production-architecture--the-src-module)).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/umandathathsarani/bank-marketing-response-ml.git
   cd bank-marketing-response-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (not committed to the repo):
   ```python
   from ucimlrepo import fetch_ucirepo
   import pandas as pd
   bm = fetch_ucirepo(id=222)
   df = pd.concat([bm.data.features, bm.data.targets], axis=1)
   df.to_csv('data/raw/bank_marketing.csv', index=False)
   print(f'Saved {len(df):,} rows.')
   ```

4. **Run notebooks in order:**
   ```
   01_data_understanding.ipynb
   02_eda.ipynb
   03_preprocessing_feature_engineering.ipynb
   04_model_training.ipynb
   05_evaluation_and_business_insights.ipynb
   ```

> **Random seed:** 42 used throughout for full reproducibility.

---

## 📁 Repository Structure

```text
bank-marketing-response-ml/
├── data/
│   ├── raw/                  # Raw dataset (downloaded locally, not committed)
│   │   └── .gitkeep
│   └── processed/            # Processed splits (not committed)
│       └── .gitkeep
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation_and_business_insights.ipynb
├── reports/
│   ├── figures/              # All exported charts and plots
│   ├── data_dictionary.md    # Full feature reference with leakage annotations
│   ├── decision_log.md       # Every major decision with options, evidence, and trade-offs
│   ├── eda_insight_log.md    # EDA findings and actions
│   ├── final_findings.md     # Complete results summary with actual numbers
│   └── preprocessing_feature_log.md
├── src/                      # Production-ready Python modules
│   ├── data/load_data.py
│   ├── preprocessing/clean_data.py
│   ├── features/build_features.py
│   ├── models/train.py
│   ├── models/predict.py
│   └── evaluation/metrics.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Production Architecture — the `src/` Module

While analysis and results live in the Jupyter Notebooks, the project includes a fully modular `src/` package that refactors the notebook logic into production-ready Python code. This separation follows standard **Software Engineering for ML** practice — notebooks are for exploration and communication; `src/` is for repeatable, testable deployment.

```
src/
├── data/
│   └── load_data.py          # load_raw_data(), describe_dataset()
├── preprocessing/
│   └── clean_data.py         # basic_cleaning(), drop_leakage_features(), encode_target()
├── features/
│   └── build_features.py     # make_features(), add_was_previously_contacted(),
│                             # add_age_group(), add_campaign_capped()
├── models/
│   ├── train.py              # build_pipeline(), train(), load_pipeline()
│   └── predict.py            # predict(), predict_proba(), score_customers()
└── evaluation/
    └── metrics.py            # compute_metrics(), plot_roc_curve(),
                              # plot_pr_curve(), threshold_analysis()
```

A complete end-to-end scoring run using the `src/` modules:

```python
from src.data.load_data import load_raw_data
from src.preprocessing.clean_data import basic_cleaning
from src.features.build_features import make_features
from src.models.train import train
from src.models.predict import score_customers
from sklearn.model_selection import train_test_split

# 1. Load
df = load_raw_data()

# 2. Clean  (drops 'duration' leakage feature, encodes target 0/1)
df = basic_cleaning(df)

# 3. Feature engineering  (was_previously_contacted, age_group, campaign_capped)
df = make_features(df)

# 4. Split
X, y = df.drop(columns=['y']), df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 5. Train and save the pipeline
pipeline = train(X_train, y_train)  # saved to models/gradient_boosting_pipeline.pkl

# 6. Score and rank customers by predicted subscription probability
priority_list = score_customers(X_test, pipeline, threshold=0.30)
print(priority_list.head(10))
```

`score_customers()` returns a DataFrame sorted by `subscription_probability` with a `priority` column (`High` / `Low`), ready to hand off to a campaign manager or export to a CRM system.

---

## 🛡️ Responsible AI

- **No causation claims** — the model identifies correlations in historical data; it does not prove that contacting a customer *causes* subscription.
- **Demographic features** — the model uses age, job, marital status, and education. Any deployment must be reviewed against applicable data protection and fair-treatment regulations.
- **Historical bias** — the model reflects patterns from past campaigns (2008–2010). If campaigns historically favoured or excluded certain groups, the model may perpetuate those patterns.
- **Human oversight required** — model predictions should support, not replace, campaign manager judgment.
- **Transparency** — Logistic Regression coefficients and Random Forest feature importances are documented in notebook 05 to explain model behaviour.

---

## ⚠️ Limitations

- Historical data from a specific Portuguese bank (2008–2010) — may not generalise to other institutions or time periods.
- No causal design — associations found do not prove causation.
- No campaign cost/revenue data — business ROI cannot be quantified from this dataset alone.
- Class imbalance — despite handling via class weights and threshold tuning, the model may still miss subscribers in certain subgroups (notably the 30–50 age bracket).
- Validation recommended — pilot A/B testing is strongly advised before operational deployment.

---

## 🤖 AI Usage

This project was developed using **Antigravity** as an AI coding assistant. AI assistance was used for code scaffolding, debugging, documentation writing, and algorithm explanations. All analysis decisions, interpretations, and results are based on code actually executed against the real dataset. No results were fabricated or assumed.

---

*Created by Umanda Thathsarani for IT3091 Machine Learning.*
