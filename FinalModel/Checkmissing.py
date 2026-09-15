"""
Check missing values in the Kanpur 15-minute CSV.
The CSV is located in the FinalModel folder.
"""

import os
import pandas as pd


# --------------------------------------------------
# CSV PATH
# --------------------------------------------------
CSV_PATH = r"D:\SIH26\AWS-AnomalyDetection\FinalModel\incompass_kanpur_July_August_2018_15min.csv"


# --------------------------------------------------
# CHECK FILE EXISTS
# --------------------------------------------------
if not os.path.exists(CSV_PATH):
    print("ERROR: CSV file not found!")
    print()
    print("Expected file:")
    print(CSV_PATH)
    raise SystemExit(1)


# --------------------------------------------------
# READ CSV
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print(f"File: {CSV_PATH}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()


# --------------------------------------------------
# CHECK THE SENSOR COLUMNS
# --------------------------------------------------
sensor_columns = [
    "temperature",
    "pressure",
    "humidity"
]

print("Missing values:")
print("-" * 60)

for col in sensor_columns:

    if col in df.columns:

        n_missing = df[col].isna().sum()
        percentage = (n_missing / len(df)) * 100

        print(
            f"{col}: {n_missing} missing "
            f"({percentage:.2f}%)"
        )

    else:
        print(f"{col}: COLUMN NOT FOUND")


# --------------------------------------------------
# CHECK ALL COLUMNS
# --------------------------------------------------
print()
print("All columns:")
print("-" * 60)

for col in df.columns:

    n_missing = df[col].isna().sum()
    percentage = (n_missing / len(df)) * 100

    print(
        f"{col}: {n_missing} missing "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print()
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)