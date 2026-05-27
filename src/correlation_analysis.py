import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/dataset.csv")

# Convert categorical columns
categorical_cols = [
    "location_sharing",
    "password_reuse",
    "cookie_acceptance",
    "os_update_frequency",
    "risk_level"
]

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# Correlation with risk_score
correlations = df.corr()["risk_score"].abs().sort_values(ascending=False)

# Remove self-correlation
correlations = correlations.drop("risk_score")

top_10 = correlations.head(10)

print("\n🔍 Top 10 Risk Features\n")
print(top_10)

# Save results
top_10.to_csv("data/processed/top_10_risk_features.csv")

print("\n✅ Saved: data/processed/top_10_risk_features.csv")