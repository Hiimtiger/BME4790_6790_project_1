
from pathlib import Path
import json
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from utils.model import BloodCellCNN
from utils.gradcam import (
    GradCAM,
    denormalize_image,
    save_gradcam_figure,
)
from utils.evaluator import CLASS_NAMES


# =====================================
# 1. INFERENCE SETTINGS
# =====================================

# Select which trained CNN to use:
# 2, 3, or 4
NUM_CONV_LAYERS = 4

# Path to the image you want to classify
IMAGE_PATH = "bloodcells_processed/test/BA_000001.jpg"

# Folder containing saved checkpoints
CHECKPOINT_DIR = "saved_model"

# Normalization statistics from training
NORMALIZATION_PATH = (
    "bloodcells_processed/normalization.json"
)

# Output folder
OUTPUT_DIR = "results/inference"

# Generate Grad-CAM visualization?
GENERATE_GRADCAM = True


# =====================================
# 2. LOAD NORMALIZATION STATISTICS
# =====================================

def load_normalization(path):

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = stats["mean"]
    std = stats["std"]

    return mean, std


# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

def load_trained_model(
    num_conv_layers,
    checkpoint_dir,
    device,
):

    checkpoint_path = (
        Path(checkpoint_dir)
        / f"cnn_{num_conv_layers}_layer_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Verify architecture matches checkpoint
    saved_depth = checkpoint["num_conv_layers"]

    if saved_depth != num_conv_layers:
        raise ValueError(
            f"Checkpoint has {saved_depth} layers, "
            f"but requested {num_conv_layers}."
        )

    model = BloodCellCNN(
        num_conv_layers=num_conv_layers,
        num_classes=8,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# =====================================
# 4. LOAD AND NORMALIZE IMAGE
# =====================================

def load_image(image_path, mean, std):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    with Image.open(image_path) as img:
        image = img.convert("RGB")

        if image.size != (363, 360):
            raise ValueError(
                f"Expected image size 360x363 (H x W), "
                f"but received {image.height}x{image.width}."
            )

        image_tensor = transform(image)

    return image_tensor


# =====================================
# 5. PREDICT CLASS PROBABILITIES
# =====================================

@torch.no_grad()
def predict(model, input_tensor):

    model.eval()

    logits = model(input_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predicted_class = int(
        probabilities.argmax(dim=1).item()
    )

    confidence = float(
        probabilities[0, predicted_class].item()
    )

    return (
        predicted_class,
        confidence,
        probabilities[0].cpu().numpy(),
    )


# =====================================
# 6. SAVE PROBABILITY BAR CHART
# =====================================

def save_probability_plot(
    probabilities,
    save_path,
):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        CLASS_NAMES,
        probabilities * 100,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, 100)

    ax.set_xlabel("Predicted Probability (%)")
    ax.set_title("Blood Cell Classification Probabilities")

    for i, probability in enumerate(probabilities):
        ax.text(
            probability * 100 + 0.5,
            i,
            f"{probability * 100:.1f}%",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =====================================
# 7. MAIN INFERENCE PIPELINE
# =====================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = Path(IMAGE_PATH)

    print("\n" + "=" * 55)
    print("BLOOD CELL INFERENCE")
    print("=" * 55)

    print(f"Device: {device}")
    print(f"Model depth: {NUM_CONV_LAYERS} layers")
    print(f"Image: {image_path}")

    # -------------------------
    # Load normalization
    # -------------------------

    mean, std = load_normalization(
        NORMALIZATION_PATH
    )

    # -------------------------
    # Load model
    # -------------------------

    model, checkpoint = load_trained_model(
        num_conv_layers=NUM_CONV_LAYERS,
        checkpoint_dir=CHECKPOINT_DIR,
        device=device,
    )

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

    # -------------------------
    # Load image
    # -------------------------

    image_tensor = load_image(
        image_path=image_path,
        mean=mean,
        std=std,
    )

    input_tensor = (
        image_tensor.unsqueeze(0).to(device)
    )

    # -------------------------
    # Predict
    # -------------------------

    (
        predicted_class,
        confidence,
        probabilities,
    ) = predict(model, input_tensor)

    print("\nPREDICTION")
    print("-" * 55)

    print(
        f"Predicted class: "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        print(
            f"  {class_name:<24} "
            f"{probability * 100:6.2f}%"
        )

    # -------------------------
    # Save prediction results
    # -------------------------

    image_id = image_path.stem
    model_id = f"cnn_{NUM_CONV_LAYERS}_layer"

    results = {
        "image_path": str(image_path),
        "model": model_id,
        "checkpoint_epoch": checkpoint["epoch"],
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            name: float(prob)
            for name, prob in zip(
                CLASS_NAMES,
                probabilities,
            )
        },
    }

    json_path = (
        output_dir
        / f"{model_id}_{image_id}_prediction.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    # -------------------------
    # Save probability plot
    # -------------------------

    plot_path = (
        output_dir
        / f"{model_id}_{image_id}_probabilities.png"
    )

    save_probability_plot(
        probabilities,
        plot_path,
    )

    # -------------------------
    # Generate Grad-CAM
    # -------------------------

    if GENERATE_GRADCAM:

        gradcam = GradCAM(model)

        try:
            heatmap, _ = gradcam.generate(
                input_tensor,
                target_class=predicted_class,
            )
        finally:
            gradcam.remove_hooks()

        original_image = denormalize_image(
            image_tensor,
            mean,
            std,
        )

        gradcam_path = (
            output_dir
            / f"{model_id}_{image_id}_gradcam.png"
        )

        save_gradcam_figure(
            image=original_image,
            heatmap=heatmap,
            true_class=predicted_class,
            predicted_class=predicted_class,
            save_path=gradcam_path,
            class_names=CLASS_NAMES,
        )

        print(
            f"\nGrad-CAM saved: {gradcam_path}"
        )

    print(f"Prediction saved: {json_path}")
    print(f"Probability plot saved: {plot_path}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
