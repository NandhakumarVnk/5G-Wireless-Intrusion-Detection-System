# ============================================================
# HTSTCL-GNN FINAL EVALUATION & THESIS-READY VISUALIZATION
# ============================================================

import os
import sys
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from tqdm.auto import tqdm


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_DIR = "/kaggle/working"

DATASET_DIR = "/kaggle/input/datasets/nandhunk07/5g-ids"

SEQUENCE_DATASET_PATH = os.path.join(
    DATASET_DIR,
    "graph_sequences.pt"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "saved_models",
    "htstcl_gnn_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "evaluation_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. MODEL CONFIGURATION
# MUST MATCH train.py
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32

INPUT_DIM = 91

HIDDEN_DIM = 128

EMBEDDING_DIM = 128

NUM_CLASSES = 2

GRU_LAYERS = 2

DROPOUT = 0.3

NUM_WORKERS = 0


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. PROJECT IMPORT
# ============================================================

sys.path.insert(
    0,
    PROJECT_DIR
)

from models.htstcl_gnn import HTSTCL_GNN


# ============================================================
# 4. REPRODUCIBILITY
# MUST MATCH TRAINING
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# 5. DISPLAY CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("HTSTCL-GNN FINAL EVALUATION")
print("=" * 70)

print(f"Device        : {DEVICE}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Random Seed   : {RANDOM_SEED}")
print(f"Dataset Path  : {SEQUENCE_DATASET_PATH}")
print(f"Model Path    : {MODEL_PATH}")
print(f"Output Folder : {OUTPUT_DIR}")


# ============================================================
# 6. VERIFY FILES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING REQUIRED FILES")
print("=" * 70)

required_files = [

    os.path.join(PROJECT_DIR, "models", "__init__.py"),

    os.path.join(PROJECT_DIR, "models", "spatial_encoder.py"),

    os.path.join(PROJECT_DIR, "models", "temporal_encoder.py"),

    os.path.join(PROJECT_DIR, "models", "hierarchical_fusion.py"),

    os.path.join(PROJECT_DIR, "models", "htstcl_gnn.py"),

    MODEL_PATH,

    SEQUENCE_DATASET_PATH
]


for file_path in required_files:

    if os.path.exists(file_path):

        print(f"✓ FOUND : {file_path}")

    else:

        raise FileNotFoundError(
            f"\n✗ MISSING : {file_path}"
        )


# ============================================================
# 7. LOAD GRAPH SEQUENCE DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING GRAPH SEQUENCE DATASET")
print("=" * 70)


full_dataset = torch.load(
    SEQUENCE_DATASET_PATH,
    weights_only=False
)


print(
    f"Total sequences: {len(full_dataset)}"
)


sample = full_dataset[0]

print(
    f"Sample type: {type(sample)}"
)

print(
    f"Available keys: {sample.keys()}"
)

print(
    f"Graphs per sequence: {len(sample['graphs'])}"
)


# ============================================================
# 8. LABEL FUNCTION
# ============================================================

def get_label(sample):

    label = sample["label"]

    if torch.is_tensor(label):

        label = label.item()

    return int(label)


# ============================================================
# 9. EXTRACT LABELS
# ============================================================

print("\nExtracting labels...")


all_labels = np.array(
    [
        get_label(sample)
        for sample in full_dataset
    ]
)


print(
    "Full dataset distribution:"
)

print(
    Counter(all_labels.tolist())
)


# ============================================================
# 10. RECREATE EXACT TRAIN / VALIDATION / TEST SPLIT
#
# MUST MATCH train.py:
#
# 70% Training
# 15% Validation
# 15% Testing
#
# RANDOM_SEED = 42
# ============================================================

all_indices = np.arange(
    len(full_dataset)
)


train_indices, temp_indices = train_test_split(

    all_indices,

    test_size=0.30,

    random_state=RANDOM_SEED,

    stratify=all_labels
)


temp_labels = all_labels[
    temp_indices
]


val_indices, test_indices = train_test_split(

    temp_indices,

    test_size=0.50,

    random_state=RANDOM_SEED,

    stratify=temp_labels
)


train_dataset = Subset(
    full_dataset,
    train_indices.tolist()
)


val_dataset = Subset(
    full_dataset,
    val_indices.tolist()
)


test_dataset = Subset(
    full_dataset,
    test_indices.tolist()
)


# ============================================================
# 11. VERIFY SPLIT
# ============================================================

train_labels = all_labels[
    train_indices
]

val_labels = all_labels[
    val_indices
]

test_labels = all_labels[
    test_indices
]


print("\n" + "=" * 70)
print("RECREATED DATASET SPLIT")
print("=" * 70)

print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")


print("\nTRAIN DISTRIBUTION")
print(
    Counter(train_labels.tolist())
)


print("\nVALIDATION DISTRIBUTION")
print(
    Counter(val_labels.tolist())
)


print("\nTEST DISTRIBUTION")
print(
    Counter(test_labels.tolist())
)


# ============================================================
# 12. COLLATE FUNCTION
# MUST MATCH train.py
# ============================================================

def collate_fn(batch):

    sequences = []

    labels = []

    for sample in batch:

        sequences.append(
            sample["graphs"]
        )

        labels.append(
            get_label(sample)
        )


    labels = torch.tensor(
        labels,
        dtype=torch.long
    )


    return (
        sequences,
        labels
    )


# ============================================================
# 13. CREATE VALIDATION AND TEST DATALOADERS
# ============================================================

PIN_MEMORY = (
    DEVICE.type == "cuda"
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    collate_fn=collate_fn
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    collate_fn=collate_fn
)


print("\n" + "=" * 70)
print("DATALOADERS CREATED")
print("=" * 70)

print(
    f"Validation batches : {len(val_loader)}"
)

print(
    f"Test batches       : {len(test_loader)}"
)


# ============================================================
# 14. CREATE MODEL
# MUST MATCH TRAINING ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("CREATING HTSTCL-GNN MODEL")
print("=" * 70)


model = HTSTCL_GNN(

    input_dim=INPUT_DIM,

    hidden_dim=HIDDEN_DIM,

    embedding_dim=EMBEDDING_DIM,

    num_classes=NUM_CLASSES,

    gru_layers=GRU_LAYERS,

    dropout=DROPOUT

).to(
    DEVICE
)


# ============================================================
# 15. LOAD EXISTING TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print(
    f"✓ Model loaded successfully"
)


print(
    f"Best Epoch              : {checkpoint['epoch']}"
)


print(
    f"Best Validation Loss    : "
    f"{checkpoint['val_loss']:.6f}"
)


print(
    f"Best Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# 16. GENERIC EVALUATION FUNCTION
#
# Returns:
# - Accuracy
# - Precision
# - Recall
# - F1 Score
# - Classification Report
# - Confusion Matrix
# - Probabilities for ROC and PR curves
# ============================================================

def evaluate_model(
    loader,
    dataset_name
):

    model.eval()


    true_labels = []

    predicted_labels = []

    malicious_probabilities = []


    print("\n" + "=" * 70)

    print(
        f"EVALUATING {dataset_name.upper()} SET"
    )

    print("=" * 70)


    progress_bar = tqdm(

        loader,

        desc=f"Evaluating {dataset_name}"
    )


    with torch.no_grad():

        for batch_sequences, labels in progress_bar:


            labels = labels.to(

                DEVICE,

                non_blocking=True
            )


            with torch.amp.autocast(

                device_type=DEVICE.type,

                enabled=(DEVICE.type == "cuda")
            ):

                output = model(
                    batch_sequences
                )


                logits = output[
                    "logits"
                ]


            probabilities = torch.softmax(

                logits,

                dim=1
            )


            predictions = torch.argmax(

                logits,

                dim=1
            )


            true_labels.extend(

                labels.cpu().numpy()
            )


            predicted_labels.extend(

                predictions.cpu().numpy()
            )


            malicious_probabilities.extend(

                probabilities[:, 1]
                .cpu()
                .float()
                .numpy()
            )


    true_labels = np.array(
        true_labels
    )


    predicted_labels = np.array(
        predicted_labels
    )


    malicious_probabilities = np.array(
        malicious_probabilities
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(

        true_labels,

        predicted_labels
    )


    precision = precision_score(

        true_labels,

        predicted_labels,

        pos_label=1,

        zero_division=0
    )


    recall = recall_score(

        true_labels,

        predicted_labels,

        pos_label=1,

        zero_division=0
    )


    f1 = f1_score(

        true_labels,

        predicted_labels,

        pos_label=1,

        zero_division=0
    )


    cm = confusion_matrix(

        true_labels,

        predicted_labels,

        labels=[0, 1]
    )


    report = classification_report(

        true_labels,

        predicted_labels,

        labels=[0, 1],

        target_names=[

            "Benign",

            "Malicious"

        ],

        digits=4,

        zero_division=0
    )


    results = {

        "dataset": dataset_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "confusion_matrix": cm,

        "classification_report": report,

        "true_labels": true_labels,

        "predicted_labels": predicted_labels,

        "malicious_probabilities": malicious_probabilities
    }


    return results


# ============================================================
# 17. EVALUATE VALIDATION SET
# ============================================================

validation_results = evaluate_model(

    val_loader,

    "Validation"
)


# ============================================================
# 18. EVALUATE TEST SET
# ============================================================

test_results = evaluate_model(

    test_loader,

    "Test"
)


# ============================================================
# 19. PRINT RESULTS FUNCTION
# ============================================================

def print_results(results):

    print("\n" + "=" * 70)

    print(
        f"{results['dataset'].upper()} RESULTS"
    )

    print("=" * 70)


    print(
        f"Accuracy  : "
        f"{results['accuracy'] * 100:.2f}%"
    )


    print(
        f"Precision : "
        f"{results['precision']:.4f}"
    )


    print(
        f"Recall    : "
        f"{results['recall']:.4f}"
    )


    print(
        f"F1 Score  : "
        f"{results['f1_score']:.4f}"
    )


    print("\nClassification Report\n")


    print(
        results[
            "classification_report"
        ]
    )


    print(
        "Confusion Matrix"
    )


    print(
        results[
            "confusion_matrix"
        ]
    )


# ============================================================
# 20. DISPLAY VALIDATION RESULTS
# ============================================================

print_results(
    validation_results
)


# ============================================================
# 21. DISPLAY TEST RESULTS
# ============================================================

print_results(
    test_results
)


# ============================================================
# 22. SAVE CLASSIFICATION REPORTS
# ============================================================

validation_report_path = os.path.join(

    OUTPUT_DIR,

    "validation_classification_report.txt"
)


test_report_path = os.path.join(

    OUTPUT_DIR,

    "test_classification_report.txt"
)


with open(

    validation_report_path,

    "w"

) as file:

    file.write(

        validation_results[
            "classification_report"
        ]
    )


with open(

    test_report_path,

    "w"

) as file:

    file.write(

        test_results[
            "classification_report"
        ]
    )


# ============================================================
# 23. CONFUSION MATRIX VISUALIZATION FUNCTION
# ============================================================

def plot_confusion_matrix(
    results,
    filename
):

    cm = results[
        "confusion_matrix"
    ]


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    image = ax.imshow(
        cm,
        interpolation="nearest"
    )


    plt.colorbar(
        image,
        ax=ax
    )


    class_names = [

        "Benign",

        "Malicious"
    ]


    ax.set(

        xticks=np.arange(
            len(class_names)
        ),

        yticks=np.arange(
            len(class_names)
        ),

        xticklabels=class_names,

        yticklabels=class_names,

        ylabel="Actual Class",

        xlabel="Predicted Class",

        title=(
            f"{results['dataset']} "
            f"Confusion Matrix"
        )
    )


    threshold = (

        cm.max()

        /

        2
    )


    for i in range(

        cm.shape[0]
    ):

        for j in range(

            cm.shape[1]
        ):

            ax.text(

                j,

                i,

                format(
                    cm[i, j],
                    "d"
                ),

                ha="center",

                va="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),

                fontsize=16,

                fontweight="bold"
            )


    fig.tight_layout()


    save_path = os.path.join(

        OUTPUT_DIR,

        filename
    )


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    print(
        f"✓ Saved: {save_path}"
    )


# ============================================================
# 24. GENERATE CONFUSION MATRICES
# ============================================================

plot_confusion_matrix(

    validation_results,

    "validation_confusion_matrix.png"
)


plot_confusion_matrix(

    test_results,

    "test_confusion_matrix.png"
)


# ============================================================
# 25. ROC CURVE
# TEST SET
# ============================================================

true_labels = test_results[
    "true_labels"
]

malicious_probabilities = test_results[
    "malicious_probabilities"
]


fpr, tpr, _ = roc_curve(

    true_labels,

    malicious_probabilities,

    pos_label=1
)


roc_auc = auc(
    fpr,
    tpr
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    fpr,

    tpr,

    linewidth=2,

    label=(
        f"HTSTCL-GNN "
        f"(AUC = {roc_auc:.4f})"
    )
)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve - HTSTCL-GNN"
)


plt.legend(
    loc="lower right"
)


plt.grid(
    True,
    alpha=0.3
)


roc_path = os.path.join(

    OUTPUT_DIR,

    "roc_curve.png"
)


plt.savefig(

    roc_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    f"✓ Saved: {roc_path}"
)


# ============================================================
# 26. PRECISION-RECALL CURVE
# TEST SET
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(

    true_labels,

    malicious_probabilities,

    pos_label=1
)


average_precision = average_precision_score(

    true_labels,

    malicious_probabilities
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    recall_curve,

    precision_curve,

    linewidth=2,

    label=(
        f"HTSTCL-GNN "
        f"(AP = {average_precision:.4f})"
    )
)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve - HTSTCL-GNN"
)


plt.legend(
    loc="lower left"
)


plt.grid(
    True,
    alpha=0.3
)


pr_path = os.path.join(

    OUTPUT_DIR,

    "precision_recall_curve.png"
)


plt.savefig(

    pr_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    f"✓ Saved: {pr_path}"
)


# ============================================================
# 27. CLASS DISTRIBUTION VISUALIZATION
#
# Shows full dataset distribution
# and Train / Validation / Test distribution
# ============================================================

split_names = [

    "Train",

    "Validation",

    "Test"
]


benign_counts = [

    int(np.sum(
        train_labels == 0
    )),

    int(np.sum(
        val_labels == 0
    )),

    int(np.sum(
        test_labels == 0
    ))
]


malicious_counts = [

    int(np.sum(
        train_labels == 1
    )),

    int(np.sum(
        val_labels == 1
    )),

    int(np.sum(
        test_labels == 1
    ))
]


x = np.arange(
    len(split_names)
)


width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - width / 2,

    benign_counts,

    width,

    label="Benign"
)


plt.bar(

    x + width / 2,

    malicious_counts,

    width,

    label="Malicious"
)


plt.xticks(

    x,

    split_names
)


plt.ylabel(
    "Number of Samples"
)


plt.title(
    "Class Distribution Across Dataset Splits"
)


plt.legend()


for i, value in enumerate(
    benign_counts
):

    plt.text(

        i - width / 2,

        value,

        str(value),

        ha="center",

        va="bottom"
    )


for i, value in enumerate(
    malicious_counts
):

    plt.text(

        i + width / 2,

        value,

        str(value),

        ha="center",

        va="bottom"
    )


class_distribution_path = os.path.join(

    OUTPUT_DIR,

    "class_distribution.png"
)


plt.savefig(

    class_distribution_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    f"✓ Saved: {class_distribution_path}"
)


# ============================================================
# 28. FINAL METRICS SUMMARY
# VALIDATION VS TEST
# ============================================================

metrics_df = pd.DataFrame(

    {

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"
        ],

        "Validation": [

            validation_results[
                "accuracy"
            ],

            validation_results[
                "precision"
            ],

            validation_results[
                "recall"
            ],

            validation_results[
                "f1_score"
            ],

        ],

        "Test": [

            test_results[
                "accuracy"
            ],

            test_results[
                "precision"
            ],

            test_results[
                "recall"
            ],

            test_results[
                "f1_score"
            ],

        ]
    }
)


metrics_csv_path = os.path.join(

    OUTPUT_DIR,

    "final_metrics.csv"
)


metrics_df.to_csv(

    metrics_csv_path,

    index=False
)


print("\nFINAL METRICS TABLE")

print(
    metrics_df
)


# ============================================================
# 29. FINAL METRICS SUMMARY VISUALIZATION
# ============================================================

x = np.arange(
    len(
        metrics_df["Metric"]
    )
)


width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - width / 2,

    metrics_df[
        "Validation"
    ],

    width,

    label="Validation"
)


plt.bar(

    x + width / 2,

    metrics_df[
        "Test"
    ],

    width,

    label="Test"
)


plt.xticks(

    x,

    metrics_df[
        "Metric"
    ]
)


plt.ylim(
    0,
    1.05
)


plt.ylabel(
    "Score"
)


plt.title(
    "HTSTCL-GNN Performance Summary"
)


plt.legend()


for i, value in enumerate(
    metrics_df[
        "Validation"
    ]
):

    plt.text(

        i - width / 2,

        value + 0.01,

        f"{value:.4f}",

        ha="center"
    )


for i, value in enumerate(
    metrics_df[
        "Test"
    ]
):

    plt.text(

        i + width / 2,

        value + 0.01,

        f"{value:.4f}",

        ha="center"
    )


metrics_plot_path = os.path.join(

    OUTPUT_DIR,

    "final_metrics_summary.png"
)


plt.savefig(

    metrics_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    f"✓ Saved: {metrics_plot_path}"
)


# ============================================================
# 30. SAVE COMPLETE NUMERICAL RESULTS
# ============================================================

complete_results = {

    "model": "HTSTCL-GNN",

    "best_epoch": checkpoint["epoch"],

    "best_validation_loss": checkpoint["val_loss"],

    "best_validation_accuracy": checkpoint["val_accuracy"],

    "validation": {

        "accuracy": float(
            validation_results["accuracy"]
        ),

        "precision": float(
            validation_results["precision"]
        ),

        "recall": float(
            validation_results["recall"]
        ),

        "f1_score": float(
            validation_results["f1_score"]
        ),

        "confusion_matrix": validation_results[
            "confusion_matrix"
        ].tolist()
    },

    "test": {

        "accuracy": float(
            test_results["accuracy"]
        ),

        "precision": float(
            test_results["precision"]
        ),

        "recall": float(
            test_results["recall"]
        ),

        "f1_score": float(
            test_results["f1_score"]
        ),

        "roc_auc": float(
            roc_auc
        ),

        "average_precision": float(
            average_precision
        ),

        "confusion_matrix": test_results[
            "confusion_matrix"
        ].tolist()
    }
}


results_path = os.path.join(

    OUTPUT_DIR,

    "complete_evaluation_results.pt"
)


torch.save(

    complete_results,

    results_path
)


# ============================================================
# 31. SAVE HUMAN-READABLE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(

    OUTPUT_DIR,

    "final_evaluation_summary.txt"
)


with open(

    summary_path,

    "w"

) as file:


    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        "HTSTCL-GNN FINAL EVALUATION SUMMARY\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    file.write(
        f"Best Epoch: "
        f"{checkpoint['epoch']}\n"
    )


    file.write(
        f"Best Validation Loss: "
        f"{checkpoint['val_loss']:.6f}\n"
    )


    file.write(
        f"Best Validation Accuracy: "
        f"{checkpoint['val_accuracy']:.2f}%\n\n"
    )


    file.write(
        "VALIDATION RESULTS\n"
    )

    file.write(
        "-" * 40 + "\n"
    )

    file.write(
        f"Accuracy  : "
        f"{validation_results['accuracy'] * 100:.2f}%\n"
    )

    file.write(
        f"Precision : "
        f"{validation_results['precision']:.4f}\n"
    )

    file.write(
        f"Recall    : "
        f"{validation_results['recall']:.4f}\n"
    )

    file.write(
        f"F1 Score  : "
        f"{validation_results['f1_score']:.4f}\n\n"
    )


    file.write(
        "TEST RESULTS\n"
    )

    file.write(
        "-" * 40 + "\n"
    )

    file.write(
        f"Accuracy  : "
        f"{test_results['accuracy'] * 100:.2f}%\n"
    )

    file.write(
        f"Precision : "
        f"{test_results['precision']:.4f}\n"
    )

    file.write(
        f"Recall    : "
        f"{test_results['recall']:.4f}\n"
    )

    file.write(
        f"F1 Score  : "
        f"{test_results['f1_score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}\n"
    )

    file.write(
        f"Avg Precision: "
        f"{average_precision:.4f}\n\n"
    )


    file.write(
        "TEST CONFUSION MATRIX\n"
    )

    file.write(
        str(
            test_results[
                "confusion_matrix"
            ]
        )
    )


print(
    f"✓ Saved: {summary_path}"
)


# ============================================================
# 32. FINAL IDS RESULTS
# ============================================================

test_cm = test_results[
    "confusion_matrix"
]


TN = int(
    test_cm[0, 0]
)

FP = int(
    test_cm[0, 1]
)

FN = int(
    test_cm[1, 0]
)

TP = int(
    test_cm[1, 1]
)


print("\n" + "=" * 70)
print("FINAL IDS RESULTS")
print("=" * 70)

print(
    f"True Negatives  (Correct Benign Detection)    : {TN}"
)

print(
    f"False Positives (Benign → Malicious)          : {FP}"
)

print(
    f"False Negatives (Malicious → Benign)          : {FN}"
)

print(
    f"True Positives  (Correct Malicious Detection) : {TP}"
)


print("\nMODEL PERFORMANCE")

print(
    f"Validation Accuracy : "
    f"{validation_results['accuracy'] * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_results['accuracy'] * 100:.2f}%"
)

print(
    f"Test Precision      : "
    f"{test_results['precision']:.4f}"
)

print(
    f"Test Recall         : "
    f"{test_results['recall']:.4f}"
)

print(
    f"Test F1 Score       : "
    f"{test_results['f1_score']:.4f}"
)

print(
    f"ROC-AUC             : "
    f"{roc_auc:.4f}"
)


print("\n" + "=" * 70)
print("ALL EVALUATION RESULTS GENERATED SUCCESSFULLY")
print("=" * 70)


print("\nOutput files:")


for file_name in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        os.path.join(
            OUTPUT_DIR,
            file_name
        )
    )