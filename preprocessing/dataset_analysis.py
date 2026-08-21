import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/encoded.csv")

print("=" * 60)
print("5G-NIDD DATASET ANALYSIS")
print("=" * 60)

# Dataset Shape
print("\nDataset Shape")
print(df.shape)

# First Five Rows
print("\nFirst Five Rows")
print(df.head())

# Column Names
print("\nColumns")
print(df.columns.tolist())

# Data Types
print("\nData Types")
print(df.dtypes)

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print(df.duplicated().sum())

# Basic Statistics
print("\nStatistics")
print(df.describe())

print("\nAnalysis Completed Successfully.")