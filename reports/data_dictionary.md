# Data Dictionary

## Source
Bank Marketing Dataset — UCI Machine Learning Repository (ID 222)  
S. Moro, P. Cortez, P. Rita (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems.

## Original Features

| Variable | Meaning | Type | Role | Missing / Unknown | Prediction Availability | Notes |
| -------- | ------- | ---- | ---- | ----------------- | ----------------------- | ----- |
| `age` | Client age in years | Integer | Feature | None | ✅ Pre-contact | Stable CRM attribute |
| `job` | Type of job | Categorical (12 values) | Feature | 288 NaN / 'unknown' category | ✅ Pre-contact | 'unknown' retained as explicit category |
| `marital` | Marital status | Categorical | Feature | None | ✅ Pre-contact | Values: married, single, divorced |
| `education` | Education level | Categorical | Feature | 1,857 NaN / 'unknown' | ✅ Pre-contact | 'unknown' retained |
| `default` | Has credit in default? | Binary (yes/no) | Feature | None | ✅ Pre-contact | |
| `balance` | Average yearly bank balance (€) | Integer | Feature | None | ✅ Pre-contact | Right-skewed; outliers present |
| `housing` | Has housing loan? | Binary (yes/no) | Feature | None | ✅ Pre-contact | |
| `loan` | Has personal loan? | Binary (yes/no) | Feature | None | ✅ Pre-contact | |
| `contact` | Contact communication type | Categorical | Feature | 13,020 NaN / 'unknown' | ✅ Pre-contact | Contact type decided before calling; 'unknown' retained |
| `day_of_week` | Last contact day of month | Integer | Feature | None | ✅ Pre-contact | Campaign scheduling information |
| `month` | Last contact month | Categorical | Feature | None | ✅ Pre-contact | Campaign scheduling information |
| `duration` | Last contact duration (seconds) | Integer | **DROPPED** | None | ❌ **Post-contact** | **Leakage — only known after call ends. Excluded from all models.** |
| `campaign` | Contacts during this campaign | Integer | Feature | None | ✅ Pre-contact | Includes last contact |
| `pdays` | Days since last previous campaign contact (-1 = never) | Integer | Feature | None | ✅ Pre-contact | -1 encodes 'never contacted' |
| `previous` | Contacts before this campaign | Integer | Feature | None | ✅ Pre-contact | |
| `poutcome` | Outcome of previous campaign | Categorical | Feature | 36,959 'unknown' (81.7%) | ✅ Pre-contact | 'unknown' retained — majority of rows |
| `y` | Subscribed to term deposit? | Binary (yes/no) | **Target** | None | N/A | Encoded: yes=1, no=0 |

## Engineered Features

| Variable | Meaning | Type | Derived From | Leakage Risk | Notes |
| -------- | ------- | ---- | ------------ | ------------ | ----- |
| `was_previously_contacted` | Binary flag — was client contacted in a previous campaign? | Binary (0/1) | `pdays` | None | pdays != -1 → 1 |
| `age_group` | Binned age for interpretability | Ordinal category | `age` | None | Bins: <30, 30-40, 40-50, 50-60, 60+ |
| `campaign_capped` | Campaign contact count capped at 95th percentile | Integer | `campaign` | None | Reduces outlier influence |

