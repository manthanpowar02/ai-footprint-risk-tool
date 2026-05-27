import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/dataset.csv")

# Create processed directory
os.makedirs("data/processed", exist_ok=True)

# Set style
sns.set(style="darkgrid")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Risk Level Distribution
sns.countplot(
    data=df,
    x="risk_level",
    palette="viridis",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Risk Level Distribution")

# 2. Location Sharing vs Risk Score
sns.boxplot(
    data=df,
    x="location_sharing",
    y="risk_score",
    palette="magma",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Location Sharing vs Risk Score")

# 3. Social Media Accounts Distribution
sns.histplot(
    df["social_media_accounts"],
    bins=10,
    kde=True,
    color="skyblue",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Social Media Accounts Distribution")

# 4. Correlation Heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()

# Save plot
plt.savefig(
    "data/processed/eda_plots.png",
    dpi=300,
    bbox_inches="tight"
)

print("✅ EDA complete")
print("📊 Saved: data/processed/eda_plots.png")