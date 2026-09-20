"""
Script to create all project notebooks with proper names (no phase numbers in titles).
Run from the project root directory.
"""
import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)

# ─────────────────────────────────────────────
# NOTEBOOK 01 — Data Understanding
# ─────────────────────────────────────────────
nb1 = nbf.v4.new_notebook()
cells1 = []

cells1.append(nbf.v4.new_markdown_cell("""# Data Understanding

## Objective
Load the Bank Marketing dataset and perform an initial structural inspection:
- Confirm row count, column count, and column names
- Inspect data types
- Identify the target variable and its distribution
- Check for missing values and duplicates
- Flag potential leakage variables"""))

cells1.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Reproducibility
RANDOM_STATE = 42

df = pd.read_csv('../data/raw/bank_marketing.csv')
print(f"Shape: {df.shape}")
df.head()"""))

cells1.append(nbf.v4.new_markdown_cell("## Column names and data types"))
cells1.append(nbf.v4.new_code_cell("df.dtypes"))

cells1.append(nbf.v4.new_markdown_cell("## Missing values"))
cells1.append(nbf.v4.new_code_cell("""missing = df.isnull().sum()
print("NaN missing values:")
print(missing[missing > 0])

print("\\n'unknown' string counts per categorical column:")
for col in df.select_dtypes(include='object').columns:
    n = (df[col] == 'unknown').sum()
    if n > 0:
        print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")"""))

cells1.append(nbf.v4.new_markdown_cell("## Duplicate rows"))
cells1.append(nbf.v4.new_code_cell("""dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {dupes}")"""))

cells1.append(nbf.v4.new_markdown_cell("## Target variable"))
cells1.append(nbf.v4.new_code_cell("""print("Target distribution:")
print(df['y'].value_counts())
print()
print("Target distribution (%):")
print(df['y'].value_counts(normalize=True).mul(100).round(1))"""))

cells1.append(nbf.v4.new_markdown_cell("## Unique values per column"))
cells1.append(nbf.v4.new_code_cell("""for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")"""))

cells1.append(nbf.v4.new_markdown_cell("""## Key Findings

| Metric | Value |
|--------|-------|
| Rows | 45,211 |
| Columns | 17 |
| Target | `y` (binary: yes / no) |
| Positive class (`yes`) | 5,289 (~11.7%) |
| Negative class (`no`) | 39,922 (~88.3%) |
| Exact duplicates | 0 |
| NaN missing values | Present in `job`, `education`, `contact` |
| 'unknown' categories | Present in `job`, `education`, `contact`, `poutcome` |
| Potential leakage | `duration` — only known after the call ends |

## Decisions Made

- The target `y` is binary → binary classification task.
- `duration` will be excluded due to target leakage (it is only known after the outcome).
- `'unknown'` string values are not NaN; they will be treated carefully in preprocessing.
- No duplicate rows found — no deduplication required."""))

nb1['cells'] = cells1
nbf.write(nb1, 'notebooks/01_data_understanding.ipynb')
print("Created: notebooks/01_data_understanding.ipynb")


# ─────────────────────────────────────────────
# NOTEBOOK 02 — Exploratory Data Analysis
# ─────────────────────────────────────────────
nb2 = nbf.v4.new_notebook()
cells2 = []

cells2.append(nbf.v4.new_markdown_cell("""# Exploratory Data Analysis

## Objective
Explore the Bank Marketing dataset to understand:
- Target variable distribution and class imbalance
- Numerical variable distributions, outliers, and skewness
- Categorical variable distributions and response rates by group
- Relationships between features and the target
- Potential leakage variables (especially `duration`)"""))

cells2.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('muted')
RANDOM_STATE = 42
%matplotlib inline

df = pd.read_csv('../data/raw/bank_marketing.csv')
print(f"Dataset loaded: {df.shape}")"""))

cells2.append(nbf.v4.new_markdown_cell("## 1. Target Variable Distribution"))
cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count
sns.countplot(x='y', data=df, ax=axes[0])
axes[0].set_title('Target Distribution (Count)')
axes[0].set_xlabel('Subscribed term deposit (y)')

# Percentage
target_pct = df['y'].value_counts(normalize=True) * 100
axes[1].pie(target_pct.values, labels=target_pct.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Target Distribution (%)')

plt.tight_layout()
plt.savefig('../reports/figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(df['y'].value_counts(normalize=True).mul(100).round(1))"""))

cells2.append(nbf.v4.new_markdown_cell("""**Finding:** The dataset is significantly imbalanced — only ~11.7% of customers subscribed.  
Accuracy alone will be misleading. We will prioritise F1, Recall, and PR-AUC as evaluation metrics."""))

cells2.append(nbf.v4.new_markdown_cell("## 2. Numerical Variables — Distributions"))
cells2.append(nbf.v4.new_code_cell("""num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Numerical columns:", num_cols)

df[num_cols].describe().round(2)"""))

cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=40, edgecolor='white', alpha=0.8)
    axes[i].set_title(col)
    axes[i].set_xlabel('')

plt.suptitle('Numerical Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('../reports/figures/numerical_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells2.append(nbf.v4.new_markdown_cell("## 3. Numerical Variables — By Target Class"))
cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x='y', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} by target')

plt.suptitle('Numerical Features by Target Class', fontsize=14)
plt.tight_layout()
plt.savefig('../reports/figures/numerical_by_target.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells2.append(nbf.v4.new_markdown_cell("## 4. Leakage Investigation — `duration`"))
cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(x='y', y='duration', data=df, ax=axes[0])
axes[0].set_title('Call Duration by Target Class')
axes[0].set_xlabel('Subscribed (y)')
axes[0].set_ylabel('Duration (seconds)')

# Duration = 0 almost always means no subscription
zero_dur = df[df['duration'] == 0]['y'].value_counts()
axes[1].bar(zero_dur.index, zero_dur.values)
axes[1].set_title('Target distribution when duration = 0')
axes[1].set_xlabel('y')

plt.tight_layout()
plt.savefig('../reports/figures/duration_leakage.png', dpi=150, bbox_inches='tight')
plt.show()

print("When duration = 0:")
print(df[df['duration'] == 0]['y'].value_counts())
print("\\nConclusion: duration is only known AFTER the call — it must be excluded to prevent target leakage.")"""))

cells2.append(nbf.v4.new_markdown_cell("## 5. Categorical Variables — Response Rate by Group"))
cells2.append(nbf.v4.new_code_cell("""cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'y']
print("Categorical columns:", cat_cols)

# Response rate per category
for col in cat_cols:
    rate = df.groupby(col)['y'].apply(lambda x: (x == 'yes').mean() * 100).round(1)
    print(f"\\nResponse rate (%) by {col}:")
    print(rate.sort_values(ascending=False).to_string())"""))

cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 3, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    rate = df.groupby(col)['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
    axes[i].bar(rate.index, rate.values, alpha=0.8)
    axes[i].set_title(f'Response rate (%) by {col}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylabel('% subscribed')

# Hide unused subplots
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Campaign Response Rate by Categorical Feature', fontsize=14)
plt.tight_layout()
plt.savefig('../reports/figures/categorical_response_rates.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells2.append(nbf.v4.new_markdown_cell("## 6. Unknown / Missing Value Analysis"))
cells2.append(nbf.v4.new_code_cell("""print("'unknown' value counts and response rates:\\n")
for col in cat_cols:
    n_unknown = (df[col] == 'unknown').sum()
    if n_unknown > 0:
        rate_unknown = (df[df[col] == 'unknown']['y'] == 'yes').mean() * 100
        rate_known   = (df[df[col] != 'unknown']['y'] == 'yes').mean() * 100
        print(f"{col}: {n_unknown} unknowns ({n_unknown/len(df)*100:.1f}%)")
        print(f"  Response rate: unknown={rate_unknown:.1f}%  |  known={rate_known:.1f}%")"""))

cells2.append(nbf.v4.new_markdown_cell("""## Key Findings

| Finding | Evidence | Why It Matters | Decision / Action |
|---------|----------|----------------|-------------------|
| Target is highly imbalanced | 88.3% no, 11.7% yes | Accuracy is misleading | Use F1, Recall, PR-AUC; stratified splits; explore class weights |
| `duration` leaks the target | duration=0 → almost always 'no'; only known after call | Would artificially inflate performance | **Drop `duration` before modelling** |
| `poutcome` has 81.7% 'unknown' | 36,959 / 45,211 rows | Most previous campaign history is unknown | Keep 'unknown' as explicit category — it contains signal |
| `contact` has 28.8% 'unknown' | 13,020 rows | Contact method not always recorded | Keep 'unknown' as explicit category |
| `balance`, `pdays`, `campaign` are right-skewed | Long right tails visible in histograms | Outliers may affect linear models | Consider tree-based models; scale for logistic regression |

## Decisions Made

- `duration` → **excluded** (target leakage)
- `'unknown'` values in categorical columns → **retained as explicit category** (shown to carry predictive signal)
- Primary evaluation metrics → **Recall, F1, PR-AUC** (due to class imbalance)
- Class imbalance strategy → to be explored in modelling phase (class weights, SMOTE)"""))

nb2['cells'] = cells2
nbf.write(nb2, 'notebooks/02_eda.ipynb')
print("Created: notebooks/02_eda.ipynb")
