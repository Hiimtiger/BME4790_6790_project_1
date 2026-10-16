
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =====================================
# 1. GRAD-CAM CLASS
# =====================================

class GradCAM:
    """
    Grad-CAM for BloodCellCNN.

    Uses the final convolutional layer returned by
    model.get_last_conv_layer().

    Supports 2-, 3-, and 4-layer CNNs.
    """

    def __init__(self, model):
        self.model = model

        self.activations = None
        self.gradients = None

        target_layer = model.get_last_conv_layer()

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM for one image.

        Args:
            input_tensor: shape [1, 3, H, W]
            target_class: integer class index.
                If None, use predicted class.

        Returns:
            heatmap: numpy array [H, W] in [0, 1]
            predicted_class: predicted class index
        """

        self.model.eval()

        # Grad-CAM needs gradient computation.
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)

            logits = self.model(input_tensor)

            predicted_class = int(logits.argmax(dim=1).item())

            if target_class is None:
                target_class = predicted_class

            score = logits[0, int(target_class)]
            score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture features."
                )

            # Average gradients over spatial dimensions
            weights = self.gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            # Weighted combination of feature maps
            cam = (
                weights * self.activations
            ).sum(dim=1, keepdim=True)

            cam = F.relu(cam)

            # Resize heatmap to original image dimensions
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0]

            # Normalize to [0, 1]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            heatmap = cam.detach().cpu().numpy()

        return heatmap, predicted_class

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# =====================================
# 2. REVERSE Z-SCORE NORMALIZATION
# =====================================

def denormalize_image(image_tensor, mean, std):
    """
    Convert a normalized tensor [3, H, W]
    back into an RGB image in [0, 1].
    """

    mean = torch.as_tensor(
        mean,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    std = torch.as_tensor(
        std,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    image = image_tensor * std + mean

    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).detach().cpu().numpy()


# =====================================
# 3. SAVE SIDE-BY-SIDE FIGURE
# =====================================

def save_gradcam_figure(
    image,
    heatmap,
    true_class,
    predicted_class,
    save_path,
    class_names,
    epoch=None,
):
    """
    Save:
        Original | Heatmap | Overlay
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(image)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    title = (
        f"True: {class_names[true_class]} | "
        f"Predicted: {class_names[predicted_class]}"
    )

    if epoch is not None:
        title += f" | Epoch {epoch}"

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================
# 4. GENERATE GRAD-CAM EXAMPLES
# =====================================

def generate_gradcam_examples(
    model,
    dataset,
    device,
    mean,
    std,
    class_names,
    save_dir,
    epoch=None,
    num_images=8,
    indices=None,
):
    """
    Generate Grad-CAM figures for selected dataset images.

    By default, uses the first num_images images.

    For comparisons across epochs/models,
    pass the same fixed indices every time.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = list(
            range(min(num_images, len(dataset)))
        )

    was_training = model.training
    model.eval()

    gradcam = GradCAM(model)

    try:
        for idx in indices:
            image_tensor, true_class = dataset[idx]

            input_tensor = (
                image_tensor.unsqueeze(0).to(device)
            )

            # Fixed target: ground-truth class
            heatmap, predicted_class = gradcam.generate(
                input_tensor,
                target_class=int(true_class),
            )

            original_image = denormalize_image(
                image_tensor,
                mean,
                std,
            )

            epoch_label = (
                f"epoch_{epoch:03d}"
                if epoch is not None
                else "final"
            )

            save_path = (
                save_dir
                / f"{epoch_label}_image_{idx:04d}.png"
            )

            save_gradcam_figure(
                image=original_image,
                heatmap=heatmap,
                true_class=int(true_class),
                predicted_class=predicted_class,
                save_path=save_path,
                class_names=class_names,
                epoch=epoch,
            )

    finally:
        gradcam.remove_hooks()
        model.train(was_training)

    print(
        f"Saved {len(indices)} Grad-CAM figures to {save_dir}"
    )


# =====================================
# 5. STANDALONE TEST
# =====================================

if __name__ == "__main__":

    from utils.model import BloodCellCNN

    model = BloodCellCNN(num_conv_layers=4)

    dummy_image = torch.randn(1, 3, 360, 363)

    gradcam = GradCAM(model)

    heatmap, predicted_class = gradcam.generate(
        dummy_image,
    )

    print("Grad-CAM heatmap shape:", heatmap.shape)
    print("Predicted class:", predicted_class)

    assert heatmap.shape == (360, 363)

    gradcam.remove_hooks()

    print("Grad-CAM test passed!")
