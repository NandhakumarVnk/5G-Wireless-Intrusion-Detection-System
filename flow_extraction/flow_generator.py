import pandas as pd
import os

print("=" * 60)
print("FLOW GENERATION MODULE")
print("=" * 60)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("dataset/encoded.csv")

print(f"Dataset Shape : {df.shape}")

# -------------------------------------------------
# Remove Only Unnecessary Columns
# -------------------------------------------------

remove_columns = [
    "Unnamed: 0",
    "Seq"
]

flow_columns = [col for col in df.columns if col not in remove_columns]

flow_df = df[flow_columns].copy()

# -------------------------------------------------
# Create Flow ID
# -------------------------------------------------

flow_df.insert(0, "Flow_ID", range(1, len(flow_df) + 1))

print("\nFlow Dataset")

print("Rows :", len(flow_df))

print("Columns :", len(flow_df.columns))

print("\nLast Columns")

print(flow_df.columns[-5:])

# -------------------------------------------------
# Save
# -------------------------------------------------

os.makedirs("outputs", exist_ok=True)

flow_df.to_csv(
    "outputs/flow_records.csv",
    index=False
)

print("\nFlow Dataset Saved Successfully.")

print("Location : outputs/flow_records.csv")

print("=" * 60)