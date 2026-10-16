
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.evaluator import (
    CLASS_NAMES,
    evaluate_model,
    save_training_history,
)
from utils.gradcam import generate_gradcam_examples


# =====================================
# 1. REPRODUCIBILITY
# =====================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 2. TRAIN ONE EPOCH
# =====================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch=None,
    num_epochs=None,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = 100.0 * total_correct / total_samples

    return epoch_loss, epoch_accuracy


# =====================================
# 3. VALIDATE ONE EPOCH
# =====================================

@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    epoch=None,
    num_epochs=None,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = 100.0 * total_correct / total_samples

    return epoch_loss, epoch_accuracy


# =====================================
# 4. FULL TRAINING FUNCTION
# =====================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    model_name,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.0,
    patience=10,
    optimizer_name="adam",
    gradcam_every=0,
    gradcam_indices=None,
    seed=42,
):
    """
    Train a CNN using a single tqdm bar across all epochs.

    Best checkpoint:
        Lowest validation loss.

    Early stopping:
        Stop after 'patience' consecutive epochs
        without validation loss improvement.

    gradcam_every:
        0 = disabled
        5 = save Grad-CAM snapshots every 5 epochs
    """

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")

    if patience < 1:
        raise ValueError("patience must be at least 1")

    set_seed(seed)

    model = model.to(device)

    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("saved_model")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / f"{model_name}_best.pth"
    )

    criterion = nn.CrossEntropyLoss()

    # =================================
    # OPTIMIZER
    # =================================

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            "optimizer_name must be 'adam' or 'sgd'"
        )

    # =================================
    # TRAINING HISTORY
    # =================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 55)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {num_params:,}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Maximum epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print("=" * 55)

    start_time = time.perf_counter()

    # =================================
    # SINGLE TQDM PROGRESS BAR
    # =================================

    with tqdm(
        total=num_epochs,
        desc=f"Epoch 0/{num_epochs}",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    ) as progress_bar:

        for epoch in range(1, num_epochs + 1):

            # -------------------------
            # Training
            # -------------------------

            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            # -------------------------
            # Validation
            # -------------------------

            val_loss, val_acc = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            # -------------------------
            # Record history
            # -------------------------

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # -------------------------
            # Save best checkpoint
            # -------------------------

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "model_name": model_name,
                        "num_conv_layers": model.num_conv_layers,
                        "num_classes": model.num_classes,
                    },
                    checkpoint_path,
                )

            else:
                epochs_without_improvement += 1

            # -------------------------
            # Optional Grad-CAM
            # -------------------------

            if (
                gradcam_every > 0
                and epoch % gradcam_every == 0
            ):
                if gradcam_indices is None:
                    selected_indices = [0, 10, 20, 30]
                else:
                    selected_indices = gradcam_indices

                selected_indices = [
                    idx for idx in selected_indices
                    if 0 <= idx < len(val_loader.dataset)
                ]

                if selected_indices:
                    generate_gradcam_examples(
                        model=model,
                        dataset=val_loader.dataset,
                        device=device,
                        mean=val_loader.dataset.mean,
                        std=val_loader.dataset.std,
                        class_names=CLASS_NAMES,
                        save_dir=results_dir / "gradcam",
                        epoch=epoch,
                        indices=selected_indices,
                    )

            # -------------------------
            # Update ONE progress bar
            # -------------------------

            progress_bar.set_description(
                f"Epoch {epoch}/{num_epochs}",
                refresh=False,
            )

            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "train_acc": f"{train_acc:.1f}%",
                    "val_acc": f"{val_acc:.1f}%",
                },
                refresh=False,
            )

            progress_bar.update(1)

            # -------------------------
            # Early stopping
            # -------------------------

            if epochs_without_improvement >= patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch}"
                )
                break

    total_training_seconds = (
        time.perf_counter() - start_time
    )

    # =================================
    # SAVE TRAINING HISTORY AND PLOTS
    # =================================

    save_training_history(
        history=history,
        save_dir=results_dir,
    )

    # =================================
    # RESTORE BEST CHECKPOINT
    # =================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # =================================
    # EVALUATE BEST VALIDATION MODEL
    # =================================

    best_val_metrics, _, _ = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        criterion=criterion,
    )

    # =================================
    # SAVE TRAINING SUMMARY
    # =================================

    training_summary = {
        "model_name": model_name,
        "num_conv_layers": model.num_conv_layers,
        "trainable_parameters": num_params,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": train_loader.batch_size,
        "max_epochs": num_epochs,
        "patience": patience,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_macro_f1": best_val_metrics["macro_f1"],
        "epochs_completed": len(history["train_loss"]),
        "training_time_seconds": total_training_seconds,
        "checkpoint_path": str(checkpoint_path),
    }

    with open(
        results_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            training_summary,
            f,
            indent=2,
        )

    # =================================
    # FINAL SUMMARY
    # =================================

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print(f"Model: {model_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Best validation accuracy: "
        f"{best_val_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"Best validation macro F1: "
        f"{best_val_metrics['macro_f1']:.4f}"
    )
    print(
        f"Total training time: "
        f"{total_training_seconds / 60:.2f} minutes"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {results_dir}")
    print("=" * 55)

    return model, history, training_summary
