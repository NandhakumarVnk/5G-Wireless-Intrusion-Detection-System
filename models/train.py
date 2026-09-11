
import os
import sys
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from tqdm.auto import tqdm


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = "/kaggle/working"

DATASET_DIR = (
    "/kaggle/input/datasets/nandhunk07/5g-ids"
)

SEQUENCE_DATASET_PATH = os.path.join(
    DATASET_DIR,
    "graph_sequences.pt"
)

SAVE_DIR = os.path.join(
    PROJECT_DIR,
    "saved_models"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(
    0,
    PROJECT_DIR
)

from models.htstcl_gnn import HTSTCL_GNN


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32

MAX_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

INPUT_DIM = 91

HIDDEN_DIM = 128

EMBEDDING_DIM = 128

NUM_CLASSES = 2

GRU_LAYERS = 2

DROPOUT = 0.3


# IMPORTANT:
# Kaggle previously produced mmap / shared-memory errors.
NUM_WORKERS = 0


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(
    RANDOM_SEED
)


# ============================================================
# CUDA OPTIMIZATION
# ============================================================

if DEVICE.type == "cuda":

    torch.backends.cudnn.benchmark = True


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("HTSTCL-GNN FAST & STABLE TRAINING")
print("=" * 70)

print(f"Device             : {DEVICE}")
print(f"Batch Size         : {BATCH_SIZE}")
print(f"Maximum Epochs     : {MAX_EPOCHS}")
print(f"Learning Rate      : {LEARNING_RATE}")
print(f"Weight Decay       : {WEIGHT_DECAY}")
print(f"Dropout            : {DROPOUT}")
print(f"Early Stop Patience: {PATIENCE}")
print(f"DataLoader Workers : {NUM_WORKERS}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading graph sequences dataset...")

if not os.path.exists(
    SEQUENCE_DATASET_PATH
):

    raise FileNotFoundError(
        f"Dataset not found:\n"
        f"{SEQUENCE_DATASET_PATH}"
    )


full_dataset = torch.load(
    SEQUENCE_DATASET_PATH,
    weights_only=False
)


print(
    f"Total sequences: "
    f"{len(full_dataset)}"
)


# ============================================================
# DATASET VERIFICATION
# ============================================================

sample = full_dataset[0]

print("\n" + "=" * 70)
print("VERIFYING DATASET")
print("=" * 70)

print(
    "Sample type:",
    type(sample)
)

print(
    "Available keys:",
    sample.keys()
)

print(
    "Graphs per sequence:",
    len(sample["graphs"])
)

print(
    "First label:",
    sample["label"]
)


# ============================================================
# LABEL FUNCTION
# ============================================================

def get_label(sample):

    label = sample["label"]

    if torch.is_tensor(label):

        label = label.item()

    return int(label)


# ============================================================
# EXTRACT LABELS
# ============================================================

print("\nExtracting labels...")

all_labels = np.array(
    [
        get_label(sample)
        for sample in full_dataset
    ]
)


print(
    "Full dataset distribution:",
    Counter(all_labels.tolist())
)


# ============================================================
# STRATIFIED SPLIT
#
# 70% Training
# 15% Validation
# 15% Testing
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
# VERIFY SPLIT
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
print("DATASET SPLIT")
print("=" * 70)

print(
    f"Train      : "
    f"{len(train_dataset)}"
)

print(
    f"Validation : "
    f"{len(val_dataset)}"
)

print(
    f"Test       : "
    f"{len(test_dataset)}"
)


print("\nTRAIN DISTRIBUTION")

print(
    Counter(
        train_labels.tolist()
    )
)


print("\nVALIDATION DISTRIBUTION")

print(
    Counter(
        val_labels.tolist()
    )
)


print("\nTEST DISTRIBUTION")

print(
    Counter(
        test_labels.tolist()
    )
)


# ============================================================
# COLLATE FUNCTION
#
# Model receives:
#
# [
#   [graph1, graph2, graph3, graph4, graph5],
#   [graph1, graph2, graph3, graph4, graph5],
#   ...
# ]
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
# DATALOADERS
#
# NUM_WORKERS=0 avoids Kaggle mmap error.
# ============================================================

PIN_MEMORY = (
    DEVICE.type == "cuda"
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    collate_fn=collate_fn
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
    f"Train batches      : "
    f"{len(train_loader)}"
)

print(
    f"Validation batches : "
    f"{len(val_loader)}"
)

print(
    f"Test batches       : "
    f"{len(test_loader)}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(

    train_labels,

    minlength=NUM_CLASSES
)


class_weights = (

    len(train_labels)

    /

    (

        NUM_CLASSES

        *

        class_counts
    )
)


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float32,

    device=DEVICE
)


print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"Benign     : "
    f"{class_counts[0]}"
)

print(
    f"Malicious  : "
    f"{class_counts[1]}"
)

print(
    "\nClass Weights:",
    class_weights
)


# ============================================================
# CREATE MODEL
# ============================================================

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


print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=2
)


# ============================================================
# AUTOMATIC MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress_bar = tqdm(

        train_loader,

        desc=f"Training Epoch {epoch}",

        leave=True
    )


    for batch_sequences, labels in progress_bar:


        labels = labels.to(

            DEVICE,

            non_blocking=True
        )


        optimizer.zero_grad(

            set_to_none=True
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

            loss = criterion(

                logits,

                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )


        scaler.step(
            optimizer
        )


        scaler.update()


        running_loss += (
            loss.item()
            *
            labels.size(0)
        )


        predictions = torch.argmax(

            logits,

            dim=1
        )


        correct += (

            predictions == labels

        ).sum().item()


        total += labels.size(0)


        accuracy = (

            100.0

            *

            correct

            /

            total
        )


        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{accuracy:.2f}%"
        )


    epoch_loss = (

        running_loss

        /

        total
    )


    epoch_accuracy = (

        100.0

        *

        correct

        /

        total
    )


    return (

        epoch_loss,

        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(epoch):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    true_labels = []

    predicted_labels = []


    progress_bar = tqdm(

        val_loader,

        desc=f"Validation Epoch {epoch}",

        leave=True
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


                loss = criterion(

                    logits,

                    labels
                )


            running_loss += (

                loss.item()

                *

                labels.size(0)
            )


            predictions = torch.argmax(

                logits,

                dim=1
            )


            correct += (

                predictions == labels

            ).sum().item()


            total += labels.size(0)


            true_labels.extend(

                labels.cpu().numpy()
            )


            predicted_labels.extend(

                predictions.cpu().numpy()
            )


            accuracy = (

                100.0

                *

                correct

                /

                total
            )


            progress_bar.set_postfix(

                loss=f"{loss.item():.4f}",

                acc=f"{accuracy:.2f}%"
            )


    epoch_loss = (

        running_loss

        /

        total
    )


    epoch_accuracy = (

        100.0

        *

        correct

        /

        total
    )


    return (

        epoch_loss,

        epoch_accuracy,

        true_labels,

        predicted_labels
    )


# ============================================================
# HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],

    "learning_rate": []
}


# ============================================================
# BEST MODEL
# ============================================================

best_val_loss = float("inf")

best_epoch = 0

epochs_without_improvement = 0


best_model_path = os.path.join(

    SAVE_DIR,

    "htstcl_gnn_best.pth"
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(

    1,

    MAX_EPOCHS + 1
):


    print("\n" + "=" * 70)

    print(
        f"EPOCH {epoch}/{MAX_EPOCHS}"
    )

    print("=" * 70)


    # TRAIN

    train_loss, train_accuracy = train_one_epoch(
        epoch
    )


    # VALIDATION

    (

        val_loss,

        val_accuracy,

        val_true,

        val_pred

    ) = validate(
        epoch
    )


    # UPDATE LEARNING RATE

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    # SAVE HISTORY

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["learning_rate"].append(
        current_lr
    )


    # RESULTS

    print("\nEpoch Results")

    print(
        f"Train Loss       : "
        f"{train_loss:.6f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss  : "
        f"{val_loss:.6f}"
    )

    print(
        f"Validation Acc   : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate    : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:


        best_val_loss = val_loss

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "val_loss": val_loss,

                "val_accuracy": val_accuracy,

                "history": history

            },

            best_model_path
        )


        print(
            "\n🏆 New Best Model Saved!"
        )


    else:


        epochs_without_improvement += 1


        print(

            f"\nNo validation improvement: "

            f"{epochs_without_improvement}/{PATIENCE}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:


        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(

    best_model_path,

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
    f"Best Epoch              : "
    f"{checkpoint['epoch']}"
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
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_loss_total = 0.0

true_labels = []

predicted_labels = []


progress_bar = tqdm(

    test_loader,

    desc="Testing"
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


            loss = criterion(

                logits,

                labels
            )


        test_loss_total += (

            loss.item()

            *

            labels.size(0)
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


# ============================================================
# TEST METRICS
# ============================================================

test_loss = (

    test_loss_total

    /

    len(test_dataset)
)


test_accuracy = (

    accuracy_score(

        true_labels,

        predicted_labels

    )

    *

    100
)


precision, recall, f1, _ = (

    precision_recall_fscore_support(

        true_labels,

        predicted_labels,

        average="binary",

        pos_label=1,

        zero_division=0
    )
)


cm = confusion_matrix(

    true_labels,

    predicted_labels
)


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print(

    f"\nTest Loss     : "

    f"{test_loss:.6f}"
)


print(

    f"Test Accuracy : "

    f"{test_accuracy:.2f}%"
)


print(

    f"Precision     : "

    f"{precision:.4f}"
)


print(

    f"Recall        : "

    f"{recall:.4f}"
)


print(

    f"F1 Score      : "

    f"{f1:.4f}"
)


print("\nClassification Report")


print(

    classification_report(

        true_labels,

        predicted_labels,

        target_names=[

            "Benign",

            "Malicious"

        ],

        zero_division=0
    )
)


print("\nConfusion Matrix")

print(cm)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_model_path = os.path.join(

    SAVE_DIR,

    "htstcl_gnn_final.pth"
)


torch.save(

    {

        "model_state_dict": model.state_dict(),

        "best_epoch": checkpoint["epoch"],

        "best_validation_loss": checkpoint["val_loss"],

        "best_validation_accuracy": checkpoint["val_accuracy"],

        "test_loss": test_loss,

        "test_accuracy": test_accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "confusion_matrix": cm,

        "history": history

    },

    final_model_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)

print(
    f"Best Epoch          : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Validation Acc : "
    f"{checkpoint['val_accuracy']:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_accuracy:.2f}%"
)

print(
    f"Precision           : "
    f"{precision:.4f}"
)

print(
    f"Recall              : "
    f"{recall:.4f}"
)

print(
    f"F1 Score            : "
    f"{f1:.4f}"
)

print("\nSaved Models:")

print(
    best_model_path
)

print(
    final_model_path
)

print("=" * 70)