import pandas as pd

df = pd.read_csv("dataset/encoded.csv")

print("=" * 60)
print("GRAPH COLUMN INSPECTION")
print("=" * 60)

print("\nColumns:")
for col in df.columns:
    print(col)