import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/Encoded.csv"

TOP_N = 50

# Your dataset has more than 1.2 million rows.
# Sampling makes the analysis practical on a local system.
MAX_SAMPLES = 300000


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 80)
print("5G-NIDD TOP-50 FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape :", df.shape)


# ============================================================
# 2. CHECK LABEL
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 80)

print(
    df["Label"].value_counts(
        dropna=False
    )
)

print("\nUnique Label Values:")
print(
    df["Label"].unique()
)


# ============================================================
# 3. CREATE BENIGN / MALICIOUS TARGET
# ============================================================

label = df["Label"].copy()


if label.dtype == "object":

    label_clean = (
        label
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = label_clean.map({

        "benign": 0,
        "normal": 0,

        "malicious": 1,
        "attack": 1

    })

else:

    y = pd.to_numeric(
        label,
        errors="coerce"
    )


# ============================================================
# 4. FALLBACK USING ATTACK TYPE
# ============================================================

if (
    y.notna().sum() == 0
    or
    y.nunique(dropna=True) < 2
):

    print("\nWARNING:")
    print(
        "Label column does not contain two usable classes."
    )

    print(
        "Creating Benign/Malicious target using Attack Type."
    )

    attack_type = (

        df["Attack Type"]
        .astype(str)
        .str.strip()
        .str.lower()

    )

    y = np.where(

        attack_type.isin([
            "",
            "nan",
            "none",
            "benign",
            "normal"
        ]),

        0,

        1

    )

    y = pd.Series(
        y,
        index=df.index
    )


# ============================================================
# 5. REMOVE INVALID TARGET ROWS
# ============================================================

valid_rows = y.notna()

df = df.loc[
    valid_rows
].copy()

y = pd.Series(
    y.loc[valid_rows],
    index=df.index
).astype(int)


# ============================================================
# 6. FINAL TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("FINAL BENIGN / MALICIOUS DISTRIBUTION")
print("=" * 80)

target_distribution = (

    y.value_counts()
    .sort_index()
    .rename({
        0: "Benign",
        1: "Malicious"
    })

)

print(
    target_distribution
)


print(
    "\nNumber of Target Classes :",
    y.nunique()
)


# ============================================================
# 7. VERIFY TWO CLASSES
# ============================================================

if y.nunique() < 2:

    raise ValueError(

        "\nERROR: Only one target class is available.\n"
        "Feature importance cannot be calculated.\n"
        "Check the Label / Attack Type columns."

    )


# ============================================================
# 8. REMOVE NON-FEATURE COLUMNS
# ============================================================

EXCLUDE_COLUMNS = [

    "Label",
    "Attack Type",
    "Attack Tool",

    "Seq",

    "Window_ID",
    "Window_Size",

    "Flow_ID",

    "Unnamed: 0"

]


feature_columns = [

    col

    for col in df.columns

    if col not in EXCLUDE_COLUMNS

]


print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(
    "Candidate Features :",
    len(feature_columns)
)


# ============================================================
# 9. CREATE FEATURE MATRIX
# ============================================================

X = df[
    feature_columns
].copy()


# ============================================================
# 10. CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 11. HANDLE INFINITE VALUES
# ============================================================

X = X.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan

)


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

missing_before = X.isna().sum().sum()

print(
    "\nMissing values before filling :",
    missing_before
)

X = X.fillna(0)

missing_after = X.isna().sum().sum()

print(
    "Missing values after filling  :",
    missing_after
)


# ============================================================
# 13. REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [

    col

    for col in X.columns

    if X[col].nunique(
        dropna=False
    ) <= 1

]


print(
    "\nConstant Features Removed :",
    len(constant_features)
)


if constant_features:

    print(
        "Removed Features:"
    )

    for col in constant_features:

        print(
            " -",
            repr(col)
        )


X = X.drop(
    columns=constant_features
)


print(
    "\nFeatures Used for Classification :",
    X.shape[1]
)


# ============================================================
# 14. STRATIFIED SAMPLING
# ============================================================

print("\n" + "=" * 80)
print("SAMPLING")
print("=" * 80)

if len(X) > MAX_SAMPLES:

    X_sample, _, y_sample, _ = (

        train_test_split(

            X,
            y,

            train_size=MAX_SAMPLES,

            random_state=42,

            stratify=y

        )

    )

else:

    X_sample = X
    y_sample = y


print(
    "Original Samples :",
    len(X)
)

print(
    "Samples Used     :",
    len(X_sample)
)


# ============================================================
# 15. SAMPLE CLASS DISTRIBUTION
# ============================================================

print("\nSample Target Distribution:")

print(

    y_sample
    .value_counts()
    .sort_index()
    .rename({
        0: "Benign",
        1: "Malicious"
    })

)


# ============================================================
# 16. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

rf = RandomForestClassifier(

    n_estimators=200,

    max_depth=20,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


print(
    "Training started..."
)


rf.fit(

    X_sample,

    y_sample

)


print(
    "Random Forest training completed."
)


# ============================================================
# 17. CALCULATE FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING FEATURE IMPORTANCE")
print("=" * 80)


importance_df = pd.DataFrame({

    "Feature":
        X_sample.columns,

    "Importance":
        rf.feature_importances_

})


# ============================================================
# 18. SORT MOST IMPORTANT → LEAST IMPORTANT
# ============================================================

importance_df = (

    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)

)


# ============================================================
# 19. ADD RANK
# ============================================================

importance_df.insert(

    0,

    "Rank",

    range(
        1,
        len(importance_df) + 1
    )

)


# ============================================================
# 20. PRINT ALL FEATURES
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE FEATURE IMPORTANCE RANKING")
print("=" * 80)

print(

    importance_df[
        [
            "Rank",
            "Feature",
            "Importance"
        ]
    ].to_string(
        index=False
    )

)


# ============================================================
# 21. TOP 50 FEATURES
# ============================================================

top50 = (

    importance_df
    .head(TOP_N)
    .copy()

)


print("\n" + "=" * 80)
print("TOP 50 MOST IMPORTANT FEATURES")
print("=" * 80)


print(

    f"{'Rank':<8}"
    f"{'Feature':<20}"
    f"{'Importance':<15}"

)

print(
    "-" * 80
)


for _, row in top50.iterrows():

    print(

        f"{int(row['Rank']):<8}"
        f"{str(row['Feature']):<20}"
        f"{row['Importance']:<15.8f}"

    )


# ============================================================
# 22. SAVE COMPLETE FEATURE RANKING
# ============================================================

importance_df.to_csv(

    "feature_importance_all_features.csv",

    index=False

)


# ============================================================
# 23. SAVE TOP 50
# ============================================================

top50.to_csv(

    "top50_feature_importance.csv",

    index=False

)


print("\nSaved:")
print(
    "feature_importance_all_features.csv"
)

print(
    "top50_feature_importance.csv"
)


# ============================================================
# 24. PREPARE TOP 50 FOR PLOT
# ============================================================

plot_df = (

    top50
    .sort_values(
        by="Importance",
        ascending=True
    )

)


# ============================================================
# 25. CREATE TOP-50 DIAGRAM
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TOP-50 FEATURE IMPORTANCE DIAGRAM")
print("=" * 80)


plt.figure(
    figsize=(15, 18)
)


bars = plt.barh(

    plot_df["Feature"],

    plot_df["Importance"]

)


# ============================================================
# 26. ADD NUMERICAL IMPORTANCE VALUES
# ============================================================

max_importance = (

    plot_df["Importance"].max()

)


for bar in bars:

    width = bar.get_width()

    # Prevent label overlap when importance is very small
    if max_importance > 0:

        offset = (
            max_importance * 0.01
        )

    else:

        offset = 0.001

    plt.text(

        width + offset,

        bar.get_y()
        +
        bar.get_height() / 2,

        f"{width:.4f}",

        va="center",

        fontsize=9

    )


# ============================================================
# 27. GRAPH LABELS
# ============================================================

plt.xlabel(

    "Random Forest Feature Importance",

    fontsize=14

)


plt.ylabel(

    "Network Traffic Features",

    fontsize=14

)


plt.title(

    "Top 50 5G-NIDD Features for\n"
    "Benign vs Malicious Classification",

    fontsize=18,

    fontweight="bold",

    pad=15

)


plt.xticks(
    fontsize=11
)

plt.yticks(
    fontsize=10
)


# ============================================================
# 28. GRID
# ============================================================

plt.grid(

    axis="x",

    linestyle="--",

    alpha=0.35

)


# ============================================================
# 29. LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# 30. SAVE DIAGRAM
# ============================================================

plt.savefig(

    "top50_feature_importance.png",

    dpi=400,

    bbox_inches="tight"

)


plt.show()


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 80)

print(
    "Dataset               : 5G-NIDD encoded.csv"
)

print(
    "Classification        : Benign vs Malicious"
)

print(
    "Features Analyzed     :",
    X.shape[1]
)

print(
    "Top Features Printed  : 50"
)

print(
    "Ranking File          : "
    "feature_importance_all_features.csv"
)

print(
    "Top-50 File           : "
    "top50_feature_importance.csv"
)

print(
    "Diagram               : "
    "top50_feature_importance.png"
)

print("=" * 80)