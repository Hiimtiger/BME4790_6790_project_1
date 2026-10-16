
from pathlib import Path
import csv
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =====================================
# CLASS CONFIGURATION
# =====================================

CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature Granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]


# =====================================
# 1. CALCULATE METRICS
# =====================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics from
    ground-truth labels and predicted labels.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(CLASS_NAMES)))

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


# =====================================
# 2. EVALUATE A MODEL
# =====================================

@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    criterion=None,
):
    """
    Evaluate a trained CNN on a DataLoader.

    Returns:
        metrics: dictionary of overall metrics
        y_true: ground-truth labels
        y_pred: predicted labels
    """

    model.eval()

    all_true = []
    all_pred = []

    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        predictions = torch.argmax(logits, dim=1)

        if criterion is not None:
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

        total_samples += images.size(0)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    metrics = calculate_metrics(all_true, all_pred)

    if criterion is not None:
        metrics["loss"] = total_loss / total_samples

    return metrics, all_true, all_pred


# =====================================
# 3. SAVE CLASSIFICATION REPORT
# =====================================

def save_classification_report(
    y_true,
    y_pred,
    save_dir,
):
    """
    Save per-class precision, recall, F1,
    and support as a CSV file.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    output_path = save_dir / "classification_report.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1_score",
            "support",
        ])

        for class_name in CLASS_NAMES:
            row = report[class_name]

            writer.writerow([
                class_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

    print(f"Saved classification report: {output_path}")


# =====================================
# 4. SAVE CONFUSION MATRIX
# =====================================

def save_confusion_matrix(
    y_true,
    y_pred,
    save_dir,
):
    """
    Generate and save a confusion matrix figure.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
        xticks_rotation=45,
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    fig.tight_layout()

    output_path = save_dir / "confusion_matrix.png"

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    # Also save numerical values
    np.savetxt(
        save_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    print(f"Saved confusion matrix: {output_path}")


# =====================================
# 5. SAVE TRAINING HISTORY
# =====================================

def save_training_history(history, save_dir):
    """
    Save training/validation metrics and
    generate learning-curve figures.

    Expected history:
        {
            "train_loss": [...],
            "val_loss": [...],
            "train_accuracy": [...],
            "val_accuracy": [...],
        }
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    required_keys = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    ]

    for key in required_keys:
        if key not in history:
            raise KeyError(f"Missing history key: {key}")

    n_epochs = len(history["train_loss"])

    if not all(
        len(history[key]) == n_epochs
        for key in required_keys
    ):
        raise ValueError(
            "All history lists must have equal lengths."
        )

    epochs = np.arange(1, n_epochs + 1)

    # Save history CSV
    csv_path = save_dir / "history.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
        ])

        for i in range(n_epochs):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_accuracy"][i],
                history["val_accuracy"][i],
            ])

    # Plot loss
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, history["train_loss"], label="Training")
    ax.plot(epochs, history["val_loss"], label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        save_dir / "loss_curve.png",
        dpi=300,
    )
    plt.close(fig)

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, history["train_accuracy"], label="Training")
    ax.plot(epochs, history["val_accuracy"], label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        save_dir / "accuracy_curve.png",
        dpi=300,
    )
    plt.close(fig)

    print(f"Saved training history: {csv_path}")


# =====================================
# 6. SAVE OVERALL TEST RESULTS
# =====================================

def save_evaluation_results(
    metrics,
    y_true,
    y_pred,
    save_dir,
):
    """
    Save all test evaluation outputs.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics
    with open(
        save_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    save_classification_report(
        y_true,
        y_pred,
        save_dir,
    )

    # Confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        save_dir,
    )

    print("\nEvaluation results:")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")


# =====================================
# 7. TEST METRIC FUNCTIONS
# =====================================

if __name__ == "__main__":

    # Example labels for testing the evaluator
    y_true = [0, 1, 2, 3, 4, 5, 6, 7]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 6]

    metrics = calculate_metrics(y_true, y_pred)

    save_evaluation_results(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        save_dir="results/evaluator_test",
    )

    sample_history = {
        "train_loss": [2.1, 1.7, 1.3, 0.9],
        "val_loss": [2.2, 1.8, 1.5, 1.2],
        "train_accuracy": [20, 40, 65, 80],
        "val_accuracy": [18, 35, 58, 72],
    }

    save_training_history(
        sample_history,
        save_dir="results/evaluator_test",
    )

    print("\nEvaluator test complete!")
