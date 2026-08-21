import pandas as pd

df = pd.read_csv("dataset/encoded.csv")

print("=" * 60)

for column in df.columns:

    print(f"\nFeature : {column}")

    print("Datatype :", df[column].dtype)

    print("Unique :", df[column].nunique())

    print("Missing :", df[column].isnull().sum())