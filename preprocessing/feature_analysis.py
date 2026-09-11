import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATASET_PATH = PROJECT_ROOT / "dataset" / "encoded.csv"

print("=" * 70)
print("5G IDS - FEATURE EXTRACTION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

if not DATASET_PATH.exists():
    print("\nERROR: encoded.csv not found!")
    print("\nExpected location:")
    print(DATASET_PATH)
    exit()

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n" + "=" * 70)
print("COLUMN / FEATURE NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# Save feature names
output_file = Path(__file__).parent / "feature_names.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for i, column in enumerate(df.columns, start=1):
        f.write(f"{i:3}. {column}\n")

print("\n" + "=" * 70)
print("Saved feature names to:")
print(output_file)
print("=" * 70)