import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/dataset.csv")

print("📊 Original Shape:", df.shape)

# -----------------------------
# Handle Missing Values
# -----------------------------
df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("✅ Missing values handled")

# -----------------------------
# Handle Outliers using IQR
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    if col != "risk_score":

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

print("✅ Outliers handled")

# Save cleaned dataset
df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("💾 Saved: data/processed/cleaned_dataset.csv")
print("📊 Final Shape:", df.shape)