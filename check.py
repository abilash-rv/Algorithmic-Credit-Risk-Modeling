import os
import pandas as pd

# Check dataset
file = "cs-training.csv"

if os.path.exists(file):
    print("Dataset found!")

    df = pd.read_csv(file)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

else:
    print("Dataset not found!")