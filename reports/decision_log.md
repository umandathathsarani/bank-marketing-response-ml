# Decision Log

---

## Decision: Primary Lens
**Options Considered:** Customer segmentation, churn prediction, campaign response prediction  
**Selected Option:** Campaign Response Prediction  
**Reason:** The dataset contains a direct campaign outcome variable (`y`). The bank's stated goal is to improve campaign effectiveness by identifying likely responders.  
**Evidence:** Dataset target `y` encodes whether a customer subscribed after the campaign contact.  
**Trade-offs:** Does not address causation — model identifies associations, not what causes a customer to subscribe.  
**Date:** 2026-09-20

---

## Decision: Target Definition
**Options Considered:** Binary classification, multi-class, regression on probability  
**Selected Option:** Binary classification — `y` encoded as 0 (no) / 1 (yes)  
**Reason:** The target is naturally binary and maps directly to the business decision (contact vs. do not prioritise).  
**Evidence:** Target column `y` has exactly two values: 'yes', 'no'.  
**Trade-offs:** Loses any nuance in degree of interest.  
**Date:** 2026-09-20

---

## Decision: Prediction Point
**Options Considered:** During call, after call, before call  
**Selected Option:** Before the current campaign contact is made  
**Reason:** For the model to support campaign prioritisation, it must operate before the bank decides which customers to contact.  
**Evidence:** Features like `duration` (only known after the call) must be excluded.  
**Trade-offs:** Limits available information — post-contact features cannot be used.  
**Date:** 2026-09-20

---

## Decision: Leakage Exclusion — `duration`
**Options Considered:** Keep with warning, drop  
**Selected Option:** Drop `duration` entirely  
**Reason:** Call duration is only measurable after the conversation ends. Including it would mean the model uses information unavailable at decision time, making the model useless in deployment.  
**Evidence:** UCI dataset documentation explicitly warns: "this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed."  
**Trade-offs:** Slight reduction in apparent model performance, but the model is now honest and deployable.  
**Date:** 2026-09-20

---

## Decision: Treatment of 'unknown' Values
**Options Considered:** Drop rows, impute with mode, retain as explicit category  
**Selected Option:** Retain 'unknown' as an explicit category  
**Reason:** `poutcome` has 81.7% unknowns — dropping or imputing would destroy most of this column. Analysis showed 'unknown' categories have different response rates from known categories, indicating predictive signal.  
**Evidence:** Computed in `02_eda.ipynb` — response rates differ between 'unknown' and 'known' groups for all affected columns.  
**Trade-offs:** Adds an extra category level; OneHotEncoder handles this transparently.  
**Date:** 2026-09-20

---

## Decision: Train / Validation / Test Split
**Options Considered:** 80/20, 70/15/15, cross-validation only  
**Selected Option:** 70% train | 15% validation | 15% test, stratified  
**Reason:** Dataset is large enough (45,211 rows) for a held-out test set. Stratification preserves the ~11.7% positive class ratio across all splits.  
**Evidence:** Confirmed in `03_preprocessing_feature_engineering.ipynb` — positive rate consistent across splits.  
**Trade-offs:** Slightly smaller training set vs. 80/20, but protects the test set from repeated evaluation.  
**Date:** 2026-09-20

---

## Decision: Engineered Features
**Options Considered:** Use raw features only, engineer domain-informed features  
**Selected Option:** Add 3 engineered features: `was_previously_contacted`, `age_group`, `campaign_capped`  
**Reason:** `pdays=-1` is a special sentinel meaning "never contacted" — binary encoding makes this explicit. Age binning aids interpretability. Capping campaign contacts reduces outlier influence.  
**Evidence:** Feature analysis in `03_preprocessing_feature_engineering.ipynb`.  
**Trade-offs:** Adds complexity; all features are non-leaky and reproducible.  
**Date:** 2026-09-20

