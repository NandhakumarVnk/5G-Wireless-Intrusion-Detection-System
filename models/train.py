import os
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from htstcl_gnn import HTSTCL_GNN
from graph_augmentation import GraphAugmentation
from contrastive_loss import ContrastiveLoss
from sequence_dataloader import SequenceDataset, collate_fn

# ==========================================================
# HTSTCL-GNN TRAINING
# ==========================================================

print("=" * 60)
print("HTSTCL-GNN TRAINING")
print("=" * 60)

# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 16
EPOCHS = 20

LEARNING_RATE = 0.001

LAMBDA_CONTRASTIVE = 0.10

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print("Device :", DEVICE)

# ==========================================================
# Reproducibility
# ==========================================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==========================================================
# Load Sequence Dataset
# ==========================================================

dataset = torch.load(

    "outputs/sequence_dataset/graph_sequences.pt",

    weights_only=False

)

print("\nTotal Temporal Sequences :", len(dataset))

# ==========================================================
# Shuffle Dataset
# ==========================================================

random.shuffle(dataset)

# ==========================================================
# Train / Test Split
# ==========================================================

train_size = int(

    TRAIN_RATIO *

    len(dataset)

)

train_data = dataset[:train_size]

test_data = dataset[train_size:]

print("Training Sequences :", len(train_data))
print("Testing Sequences  :", len(test_data))

# ==========================================================
# Compute Class Weights
# ==========================================================

benign_count = sum(

    sample["label"] == 0

    for sample in train_data

)

malicious_count = sum(

    sample["label"] == 1

    for sample in train_data

)

print("\n")
print("=" * 60)
print("TRAINING DATASET DISTRIBUTION")
print("=" * 60)

print("Benign Sequences    :", benign_count)
print("Malicious Sequences :", malicious_count)

total = benign_count + malicious_count

class_weights = torch.tensor(

    [

        total / (2 * benign_count),

        total / (2 * malicious_count)

    ],

    dtype=torch.float32,

    device=DEVICE

)

print("\nClass Weights")

print(class_weights)

# ==========================================================
# Create Dataset Objects
# ==========================================================

train_dataset = SequenceDataset(

    train_data

)

test_dataset = SequenceDataset(

    test_data

)

# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn

)

# ==========================================================
# Show Sample
# ==========================================================

sample = train_dataset[0]

print("\n")
print("=" * 60)
print("FIRST TRAINING SAMPLE")
print("=" * 60)

print("Sequence Label      :", sample["label"])
print("Graphs in Sequence  :", len(sample["graphs"]))
print("Malicious Graphs    :", sample["malicious_graphs"])
print("Benign Graphs       :", sample["benign_graphs"])
print("Malicious Nodes     :", sample["malicious_nodes"])
print("Benign Nodes        :", sample["benign_nodes"])

# ==========================================================
# Create HTSTCL-GNN
# ==========================================================

model = HTSTCL_GNN(

    input_dim=91,

    hidden_dim=128,

    embedding_dim=128,

    num_classes=2,

    gru_layers=2,

    dropout=0.3

).to(DEVICE)

print("\nHTSTCL-GNN Created Successfully")

# ==========================================================
# Graph Augmentation
# ==========================================================

augmentor = GraphAugmentation()

print("Graph Augmentation Ready")

# ==========================================================
# Loss Functions
# ==========================================================

classification_loss_fn = nn.CrossEntropyLoss(

    weight=class_weights

)

contrastive_loss_fn = ContrastiveLoss(

    temperature=0.5

)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3

)

# ==========================================================
# Save Directory
# ==========================================================

os.makedirs(

    "saved_models",

    exist_ok=True

)

best_loss = float("inf")
# ==========================================================
# Training
# ==========================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_contrastive_loss = 0.0

    total_correct = 0
    total_samples = 0

    print("\n")
    print("-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    # ======================================================
    # Train over mini-batches
    # ======================================================

    for batch in train_loader:

        batch_view1 = []
        batch_view2 = []
        targets = []

        # --------------------------------------------------
        # Create two augmented views
        # --------------------------------------------------

        for sample in batch:

            graph_sequence = sample["graphs"]

            label = sample["label"]

            # Generate two graph augmentations
            view1, view2 = augmentor(graph_sequence)

            # Move graphs to device
            view1 = [

                graph.to(DEVICE)

                for graph in view1

            ]

            view2 = [

                graph.to(DEVICE)

                for graph in view2

            ]

            batch_view1.append(

                {

                    "graphs": view1,

                    "label": label

                }

            )

            batch_view2.append(

                {

                    "graphs": view2,

                    "label": label

                }

            )

            targets.append(label)

        # --------------------------------------------------
        # Target Tensor
        # --------------------------------------------------

        targets = torch.tensor(

            targets,

            dtype=torch.long,

            device=DEVICE

        )

        optimizer.zero_grad()

        # --------------------------------------------------
        # Forward Pass
        # --------------------------------------------------

        output1 = model(batch_view1)

        output2 = model(batch_view2)

        logits = output1["logits"]

        projection1 = output1["projection"]

        projection2 = output2["projection"]

        # --------------------------------------------------
        # Classification Loss
        # --------------------------------------------------

        classification_loss = classification_loss_fn(

            logits,

            targets

        )

        # --------------------------------------------------
        # Contrastive Loss
        # --------------------------------------------------

        contrastive_loss = contrastive_loss_fn(

            projection1,

            projection2

        )

        # --------------------------------------------------
        # Combined Loss
        # --------------------------------------------------

        loss = (

            classification_loss +

            LAMBDA_CONTRASTIVE *

            contrastive_loss

        )
                # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        loss.backward()

        # Gradient Clipping

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        total_loss += loss.item()

        total_classification_loss += classification_loss.item()

        total_contrastive_loss += contrastive_loss.item()

        predictions = torch.argmax(

            logits,

            dim=1

        )

        total_correct += (

            predictions == targets

        ).sum().item()

        total_samples += targets.size(0)

    # ======================================================
    # Epoch Statistics
    # ======================================================

    average_loss = (

        total_loss /

        len(train_loader)

    )

    average_classification_loss = (

        total_classification_loss /

        len(train_loader)

    )

    average_contrastive_loss = (

        total_contrastive_loss /

        len(train_loader)

    )

    train_accuracy = (

        total_correct /

        total_samples

    ) * 100

    print(f"Classification Loss : {average_classification_loss:.4f}")

    print(f"Contrastive Loss    : {average_contrastive_loss:.4f}")

    print(f"Total Loss          : {average_loss:.4f}")

    print(f"Training Accuracy   : {train_accuracy:.2f}%")

    # ======================================================
    # Learning Rate Scheduler
    # ======================================================

    scheduler.step(

        average_loss

    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate       : {current_lr:.6f}")

    # ======================================================
    # Save Best Model
    # ======================================================

    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(

            model.state_dict(),

            "saved_models/htstcl_gnn_best.pth"

        )

        print("Best Model Updated")
        # ==========================================================
# Evaluation
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# ----------------------------------------------------------
# Load Best Model
# ----------------------------------------------------------

model.load_state_dict(

    torch.load(

        "saved_models/htstcl_gnn_best.pth",

        map_location=DEVICE

    )

)

model.eval()

correct = 0
total = 0

true_labels = []
predicted_labels = []

evaluation_loss = 0.0

with torch.no_grad():

    for batch in test_loader:

        batch_sequences = []
        targets = []

        # --------------------------------------------------
        # Move graphs to device
        # --------------------------------------------------

        for sample in batch:

            graphs = [

                graph.to(DEVICE)

                for graph in sample["graphs"]

            ]

            batch_sequences.append(

                {

                    "graphs": graphs,

                    "label": sample["label"]

                }

            )

            targets.append(

                sample["label"]

            )

        targets = torch.tensor(

            targets,

            dtype=torch.long,

            device=DEVICE

        )

        # --------------------------------------------------
        # Forward Pass
        # --------------------------------------------------

        output = model(

            batch_sequences

        )

        logits = output["logits"]

        # --------------------------------------------------
        # Evaluation Loss
        # --------------------------------------------------

        loss = classification_loss_fn(

            logits,

            targets

        )

        evaluation_loss += loss.item()

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        predictions = torch.argmax(

            logits,

            dim=1

        )

        correct += (

            predictions == targets

        ).sum().item()

        total += targets.size(0)

        true_labels.extend(

            targets.cpu().tolist()

        )

        predicted_labels.extend(

            predictions.cpu().tolist()

        )

# ==========================================================
# Evaluation Statistics
# ==========================================================

test_loss = (

    evaluation_loss /

    len(test_loader)

)

test_accuracy = (

    correct /

    total

) * 100

print(f"\nTest Loss     : {test_loss:.4f}")

print(f"Test Accuracy : {test_accuracy:.2f}%")

# ==========================================================
# Prediction Distribution
# ==========================================================

predicted_benign = predicted_labels.count(0)

predicted_malicious = predicted_labels.count(1)

print("\nPrediction Distribution")

print("-----------------------------")

print("Predicted Benign    :", predicted_benign)

print("Predicted Malicious :", predicted_malicious)
# ==========================================================
# Precision / Recall / F1 Score
# ==========================================================

tp = fp = tn = fn = 0

for gt, pred in zip(

    true_labels,

    predicted_labels

):

    if gt == 1 and pred == 1:

        tp += 1

    elif gt == 0 and pred == 1:

        fp += 1

    elif gt == 0 and pred == 0:

        tn += 1

    elif gt == 1 and pred == 0:

        fn += 1

precision = tp / (tp + fp + 1e-8)

recall = tp / (tp + fn + 1e-8)

f1 = (

    2 *

    precision *

    recall

) / (

    precision +

    recall +

    1e-8

)

# ==========================================================
# Classification Report
# ==========================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix")

print("-" * 35)

print(f"True Positive  (TP): {tp}")

print(f"False Positive (FP): {fp}")

print(f"True Negative  (TN): {tn}")

print(f"False Negative (FN): {fn}")

# ==========================================================
# Save Final Model
# ==========================================================

torch.save(

    model.state_dict(),

    "saved_models/htstcl_gnn_final.pth"

)

# ==========================================================
# Final Summary
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Model Name           : HTSTCL-GNN")

print("Architecture         : Hierarchical Spatio-Temporal Contrastive Learning GNN")

print("Spatial Encoder      : Graph Attention Network (GAT)")

print("Temporal Encoder     : GRU")

print("Fusion Module        : Hierarchical Attention Fusion")

print("Projection Head      : Enabled")

print("Graph Augmentation   : Enabled")

print("Contrastive Learning : NT-Xent (InfoNCE)")

print("Classification Loss  : Weighted CrossEntropyLoss")

print("Sequence Length      : 5")

print(f"Batch Size           : {BATCH_SIZE}")

print("Embedding Dimension  : 128")

print(f"Epochs               : {EPOCHS}")

print(f"Learning Rate        : {LEARNING_RATE}")

print(f"Contrastive Lambda   : {LAMBDA_CONTRASTIVE}")

print("\n")

print(f"Training Samples     : {len(train_dataset)}")

print(f"Testing Samples      : {len(test_dataset)}")

print(f"Test Loss            : {test_loss:.4f}")

print(f"Final Accuracy       : {test_accuracy:.2f}%")

print(f"Precision            : {precision:.4f}")

print(f"Recall               : {recall:.4f}")

print(f"F1 Score             : {f1:.4f}")

print("\nPrediction Distribution")

print("-----------------------------")

print(f"Predicted Benign     : {predicted_benign}")

print(f"Predicted Malicious  : {predicted_malicious}")

print("\nSaved Models")

print("-----------------------------")

print("Best Model  : saved_models/htstcl_gnn_best.pth")

print("Final Model : saved_models/htstcl_gnn_final.pth")

print("=" * 60)