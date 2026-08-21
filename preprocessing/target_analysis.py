import pandas as pd

# Load dataset
df = pd.read_csv("dataset/encoded.csv")

print("="*60)
print("TARGET COLUMN ANALYSIS")
print("="*60)

print("\nUnique values in Label:")
print(df["Label"].value_counts())

print("\nUnique values in Attack Type:")
print(df["Attack Type"].value_counts())

print("\nUnique values in Attack Tool:")
print(df["Attack Tool"].value_counts())