
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load dataset
df = pd.read_csv("cs-training.csv")

# 2. Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

print("Dataset loaded successfully!")

# 3. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 4. Handle missing values using median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
    df["MonthlyIncome"].median()
)

df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].median()
)

# 5. Check missing values again
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 6. Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# 7. Class distribution
print("\nCredit Risk Class Distribution:")
print(df["SeriousDlqin2yrs"].value_counts())

# 8. Plot class distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="SeriousDlqin2yrs"
)

plt.title("Credit Risk Class Distribution")
plt.xlabel("Default Status")
plt.ylabel("Number of Customers")

plt.savefig("class_distribution.png")
plt.show()

# 9. Correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Credit Risk Correlation Heatmap")

plt.savefig("correlation_heatmap.png")
plt.show()

# 10. Save cleaned dataset
df.to_csv("cleaned_credit_data.csv", index=False)

print("\nDATA PREPROCESSING COMPLETED!")
print("Cleaned dataset saved successfully!")
