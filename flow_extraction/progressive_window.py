import pandas as pd
import os

# ---------------------------------------
# Load Flow Records
# ---------------------------------------

flow_df = pd.read_csv("outputs/flow_records.csv")

print("=" * 60)
print("PROGRESSIVE WINDOW GENERATION")
print("=" * 60)

window_sizes = [5, 10, 20, 50]

os.makedirs("outputs/progressive_windows", exist_ok=True)

for window in window_sizes:

    window_list = []

    # Non-overlapping windows
    for start in range(0, len(flow_df), window):

        end = start + window

        temp = flow_df.iloc[start:end].copy()

        # Ignore incomplete windows
        if len(temp) == window:

            temp["Window_ID"] = (start // window) + 1

            temp["Window_Size"] = window

            window_list.append(temp)

    result = pd.concat(window_list, ignore_index=True)

    filename = f"outputs/progressive_windows/window_{window}.csv"

    result.to_csv(filename, index=False)

    print(f"Window Size {window}")
    print(f"Number of Windows : {result['Window_ID'].nunique()}")
    print(f"Saved : {filename}\n")

print("Progressive Window Generation Completed Successfully.")