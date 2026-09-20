from ucimlrepo import fetch_ucirepo 
import pandas as pd
import json

print("Fetching dataset...")
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

print(bank_marketing.metadata)

# Create a full dataframe to save
df = pd.concat([X, y], axis=1)

# Save to raw
df.to_csv("data/raw/bank_marketing.csv", index=False)
print("Data saved to data/raw/bank_marketing.csv")

# Data Understanding outputs
stats = {
    "row_count": len(df),
    "column_count": len(df.columns),
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "target_distribution": df['y'].value_counts(dropna=False).to_dict() if 'y' in df.columns else None,
    "missing_values": df.isnull().sum().to_dict(),
    "duplicates": df.duplicated().sum(),
}
with open("data/raw/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=4)
print("Stats saved to data/raw/dataset_stats.json")
