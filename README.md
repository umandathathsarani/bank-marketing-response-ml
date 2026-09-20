# Bank Marketing Campaign Response Prediction

Machine learning project using bank marketing data to predict campaign response and support data-driven customer prioritisation.

---

## Overview

This project investigates whether historical customer and campaign data can predict whether a customer will subscribe to a term deposit when contacted by a bank's marketing team. It is built as a complete, professional, reproducible ML project following a business-first methodology.

## Business Problem

A Portuguese bank runs direct marketing campaigns via telephone calls to sell term deposits. The campaign results in a subscription (`yes`) or no subscription (`no`). The bank wants to improve campaign effectiveness by identifying customers who are more likely to respond positively — allowing campaign resources to be prioritised using historical evidence rather than contacting customers indiscriminately.

## Objective

> Predict whether a customer will subscribe to a term deposit (`y = yes`) using historical customer attributes and previous campaign information that is available **before** a campaign call is made.

## Dataset

| Item | Value |
|------|-------|
| **Name** | Bank Marketing |
| **Source** | [UCI Machine Learning Repository — ID 222](https://archive.ics.uci.edu/dataset/222/bank-marketing) |
| **Citation** | S. Moro, P. Cortez, P. Rita (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems. |
| **Rows** | 45,211 |
| **Features** | 17 original (16 used after leakage exclusion) + 3 engineered |
| **Target** | `y` — subscribed to term deposit (yes / no) |
| **Positive class** | ~11.7% (5,289 / 45,211) |

## Prediction Point

The model is designed to make predictions **before** a customer is contacted in a new campaign round. Only features available at that moment are used. The `duration` variable (call length) is excluded because it is only known after a call has ended.

## Methodology

| Step | Notebook |
|------|----------|
| Data Understanding | `01_data_understanding.ipynb` |
| Exploratory Data Analysis | `02_eda.ipynb` |
| Preprocessing & Feature Engineering | `03_preprocessing_feature_engineering.ipynb` |
| Model Training | `04_model_training.ipynb` |
| Evaluation & Business Insights | `05_evaluation_and_business_insights.ipynb` |

### Key methodological decisions
- `duration` excluded — **target leakage** (only known after call)
- `'unknown'` category values retained as explicit categories
- **Stratified 70/15/15 split** — preserves class imbalance across sets
- **`class_weight='balanced'`** applied to address class imbalance
- **Test set used once** — after all model and threshold decisions

## Models

| Model | Role |
|-------|------|
| DummyClassifier | Naive baseline |
| Logistic Regression | Interpretable baseline |
| Decision Tree | Rule-based interpretable model |
| Random Forest | Ensemble — best F1 at default threshold |
| **Gradient Boosting** | **Final selected model** — best ROC-AUC, PR-AUC |

## Results

### Validation Set

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| DummyClassifier *(baseline)* | 0.106 | 0.103 | 0.105 | 0.494 | 0.116 |
| Logistic Regression | 0.281 | 0.630 | 0.389 | 0.779 | 0.424 |
| Decision Tree | 0.343 | 0.552 | 0.423 | 0.744 | 0.353 |
| Random Forest | 0.330 | 0.617 | 0.430 | 0.794 | 0.453 |
| Gradient Boosting | 0.635 | 0.232 | 0.340 | 0.807 | 0.469 |

### Test Set (Final)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Random Forest (t=0.50) | 0.329 | 0.608 | 0.427 | 0.785 | 0.438 |
| Gradient Boosting (t=0.50) | 0.659 | 0.254 | 0.366 | 0.791 | 0.458 |
| **Gradient Boosting (t=0.30)** | **0.540** | **0.409** | **0.465** | **0.791** | **0.458** |

## Findings

- All models substantially outperform the naive baseline.
- Gradient Boosting at threshold 0.30 achieves the best F1 (0.465) on the test set.
- When contacting the **top 30% of customers** ranked by model score, the model captures **69.2% of actual subscribers** vs. 29.9% for random selection — approximately **2.3× improvement**.
- `poutcome = success` (previous campaign outcome) is the strongest predictor of positive response.
- The 60+ age group has the highest positive response rate and best model Recall.

## Business Recommendations

1. Deploy Gradient Boosting at threshold 0.30 for campaign prioritisation.
2. Prioritise customers with a successful previous campaign outcome.
3. Target the top 20–30% of customers by predicted probability.
4. Limit contact frequency per customer per campaign.
5. Validate with an A/B pilot study before full operational use.
6. Maintain human review of all model-generated contact lists.

## Responsible AI

- The model uses demographic attributes (age, job, marital status, education). Deployment must be reviewed against applicable data protection and fair treatment regulations.
- This model identifies **correlations**, not causal effects. It does not prove that contacting a customer causes them to subscribe.
- Historical data from 2008–2010 may not reflect current customer behaviour.
- Human oversight is required — the model supports decisions, not replaces them.

## Limitations

- Historical data from a specific Portuguese bank and time period — generalisation is not guaranteed.
- No causal claims — model identifies associations, not causes.
- No campaign cost or revenue data — ROI cannot be quantified.
- Potential historical campaign bias — past targeting may have under-served certain groups.

## Reproducibility

```bash
# 1. Clone the repository
git clone https://github.com/umandathathsarani/bank-marketing-response-ml.git
cd bank-marketing-response-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (not committed to repo)
python -c "
from ucimlrepo import fetch_ucirepo
import pandas as pd
bm = fetch_ucirepo(id=222)
df = pd.concat([bm.data.features, bm.data.targets], axis=1)
df.to_csv('data/raw/bank_marketing.csv', index=False)
print(f'Saved {len(df):,} rows.')
"

# 4. Run notebooks in order
#    01_data_understanding.ipynb
#    02_eda.ipynb
#    03_preprocessing_feature_engineering.ipynb
#    04_model_training.ipynb
#    05_evaluation_and_business_insights.ipynb
```

**Random seed:** 42 used throughout for reproducibility.

## Production Architecture — the `src/` Module

While the analysis and results live in the Jupyter Notebooks, the project also includes a fully modular `src/` package that
refactors the notebook logic into production-ready Python code. This separation follows standard **Software Engineering for ML** practice — notebooks are for exploration and communication; `src/` is for repeatable, testable deployment.

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

A complete end-to-end scoring run using the `src/` modules looks like this:

```python
from src.data.load_data import load_raw_data
from src.preprocessing.clean_data import basic_cleaning
from src.features.build_features import make_features
from src.models.train import train
from src.models.predict import score_customers
from sklearn.model_selection import train_test_split

# 1. Load
df = load_raw_data()

# 2. Clean  (drops 'duration' leakage feature, encodes target)
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

# 6. Score new customers  (sorted by predicted subscription probability)
priority_list = score_customers(X_test, pipeline, threshold=0.30)
print(priority_list.head(10))
```

The `score_customers()` function returns a DataFrame sorted by `subscription_probability`,
with a human-readable `priority` column (`High` / `Low`), ready to hand off to a campaign manager or export to a CRM system.

## AI Usage

This project was developed using **Antigravity** as an AI coding assistant. AI assistance was used for:
- Code scaffolding and notebook generation
- Debugging and refactoring
- Documentation writing
- Algorithm explanations

All analysis decisions, interpretations, and results are based on code actually executed against the real dataset. No results were fabricated or assumed.
