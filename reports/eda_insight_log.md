# EDA Insight Log

| Finding | Evidence | Why It Matters | Decision/Action |
| ------- | -------- | -------------- | --------------- |
| Target Imbalance | 'no' (88.3%), 'yes' (11.7%) | Accuracy will be misleading. Need proper metrics (F1, PR-AUC, etc.) | Use stratified splits and explore resampling or class weights. |
| Missing / Unknown values | 'job', 'education', 'contact', 'poutcome' have 'unknown' strings | Could contain predictive signal or need imputation | Investigate if 'unknown' category is predictive, otherwise impute/drop. |
| Target Leakage | 'duration' shows distinct differences but is only known after call | Including it would artificially inflate performance and create target leakage | Drop 'duration' feature before model training. |
| Numerical Skewness | 'balance', 'campaign', 'pdays' are highly right-skewed | Outliers or heavy tails can affect linear models | Explore scaling or non-linear models (tree-based). |
