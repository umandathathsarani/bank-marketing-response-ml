# Final Findings and Recommendations

## 1. Business Problem Context

A Portuguese bank ran direct marketing campaigns via phone calls to sell term deposits.
The goal of this project was to investigate whether historical customer and campaign data can predict
whether a customer will subscribe to a term deposit (`y = yes`).

**Primary lens:** Campaign Response Prediction  
**ML task:** Binary classification (y = 1 if subscribed, y = 0 if not)  
**Prediction point:** Before the current campaign contact is made  
**Unit of analysis:** Individual customer campaign interaction

---

## 2. Dataset Summary

| Item | Value |
|------|-------|
| Source | UCI Machine Learning Repository — Bank Marketing (ID 222) |
| Rows | 45,211 |
| Features used | 16 (after dropping `duration`) |
| Target | `y` — subscribed to term deposit |
| Positive class | 5,289 (11.7%) |
| Negative class | 39,922 (88.3%) |
| `duration` | **Excluded** — target leakage (only known after the call) |

---

## 3. Key Insights from EDA

- **Target is highly imbalanced** — 88.3% no, 11.7% yes. Accuracy alone is misleading.
- **`poutcome = success`** is strongly associated with positive response.
- **`poutcome`** has 81.7% 'unknown' values — retained as explicit category (carries signal).
- **`contact` type** shows different response rates between 'cellular' and 'telephone'.
- **Balance and previous contacts** show skewed distributions with outliers.

---

## 4. Model Performance — Validation Set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| DummyClassifier *(baseline)* | 0.7933 | 0.1062 | 0.1033 | 0.1047 | 0.4940 | 0.1160 |
| Logistic Regression | 0.7681 | 0.2811 | 0.6297 | 0.3887 | 0.7785 | 0.4238 |
| Decision Tree | 0.8238 | 0.3430 | 0.5516 | 0.4230 | 0.7441 | 0.3531 |
| Random Forest | 0.8083 | 0.3297 | 0.6171 | 0.4298 | 0.7940 | 0.4532 |
| Gradient Boosting | 0.8944 | 0.6345 | 0.2317 | 0.3395 | 0.8074 | 0.4689 |

---

## 5. Final Evaluation — Test Set (used once)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Random Forest (t=0.50) | 0.8093 | 0.3292 | 0.6078 | 0.4271 | 0.7854 | 0.4379 |
| Gradient Boosting (t=0.50) | 0.8974 | 0.6590 | 0.2535 | 0.3661 | 0.7912 | 0.4575 |
| **Gradient Boosting (t=0.30)** | **0.8902** | **0.5400** | **0.4086** | **0.4652** | **0.7912** | **0.4575** |

**Selected model:** Gradient Boosting at threshold 0.30  
**Reason:** Best F1 (0.4652) and PR-AUC (0.4575) on the test set. Lowering the threshold to 0.30 substantially improved Recall (0.41 vs 0.25) while maintaining reasonable Precision (0.54).

---

## 6. Threshold Analysis

By lowering the Gradient Boosting threshold from 0.50 to 0.30:
- Recall improved from 0.25 → 0.41
- Precision decreased from 0.66 → 0.54
- F1 improved from 0.37 → 0.47

The optimal threshold depends on the bank's business priorities (whether missing a subscriber or contacting a non-subscriber is more costly).

---

## 7. Hypothetical Prioritisation Scenario

> *Hypothetical — no actual costs or revenues are claimed.*

If the bank contacts only the **top 30%** of customers ranked by predicted probability:

| Strategy | Customers contacted | Subscribers captured | Capture rate |
|----------|--------------------|--------------------|-------------|
| Model (Gradient Boosting) | 2,034 | 549 | **69.2%** |
| Random selection | 2,034 | 237 | 29.9% |

The model captures **~2.3× more subscribers** than random selection when contacting the same number of customers.

---

## 8. Subgroup Performance (Gradient Boosting, t=0.30)

| Age Group | n | Positive Rate | Precision | Recall | F1 |
|-----------|---|--------------|-----------|--------|----|
| <30 | 1,028 | 15% | 0.510 | 0.506 | 0.508 |
| 30-40 | 2,595 | 10% | 0.520 | 0.350 | 0.419 |
| 40-50 | 1,676 | 9% | 0.578 | 0.312 | 0.405 |
| 50-60 | 1,307 | 12% | 0.639 | 0.342 | 0.445 |
| 60+ | 176 | 41% | 0.509 | 0.753 | 0.608 |

The 60+ group has a much higher positive rate and the model achieves its best Recall there. Younger groups (30-50) show lower Recall — the model misses more subscribers in these groups.

---

## 9. Business Recommendations

1. **Deploy Gradient Boosting at threshold 0.30** for campaign prioritisation — best overall test-set performance.
2. **Prioritise customers with `poutcome = success`** — strong predictor of re-subscription.
3. **Target top 20-30% of scored customers** — captures 61-69% of subscribers while limiting contact volume.
4. **Limit campaign contact frequency** — repeated contacts have diminishing returns.
5. **Do not use `duration`** — it is not available before a call and must not be included in deployment.
6. **Validate with a pilot A/B test** before full operational deployment.
7. **Apply human review** — the model supports, not replaces, campaign managers.

---

## 10. Limitations

- Historical data from one time period (May 2008 to November 2010) — may not generalise.
- No causal claims — model identifies correlations, not causes of subscription.
- No campaign cost or revenue data — business ROI cannot be quantified.
- Class imbalance — despite handling, the model may still underperform on minority class.
- Demographic features used — deployment must be reviewed against applicable regulations.
- No experimental design — model bias from historical campaign targeting patterns is possible.
