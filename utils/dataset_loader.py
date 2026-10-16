
from pathlib import Path
import csv
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class BloodCellDataset(Dataset):
    """
    Loads blood cell images using CSV manifests.

    Expected structure:
        bloodcells_processed/
            train/
            test/
            train.csv
            val.csv
            test.csv
            normalization.json

    Image resolution: 360 x 363 (no resizing).
    """

    def __init__(
        self,
        root_dir="bloodcells_processed",
        split="train",
        augment=False
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        if split not in ["train", "val", "test"]:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        # Read the CSV manifest
        csv_path = self.root_dir / f"{split}.csv"

        with open(csv_path, newline="", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

        if not self.records:
            raise ValueError(f"No images found in {csv_path}")

        # Load normalization statistics
        stats_path = self.root_dir / "normalization.json"

        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

        self.mean = stats["mean"]
        self.std = stats["std"]
        self.class_to_idx = stats["class_to_idx"]

        # Augmentation only for training
        transform_list = []

        if augment and split == "train":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),

                # 90-degree rotations preserve image content
                # but swap height and width for odd rotations.
                # Therefore, we use 180-degree rotations only.
                transforms.RandomApply([
                    transforms.RandomRotation(
                        degrees=(180, 180)
                    )
                ], p=0.5),
            ])

        transform_list.extend([
            # PIL dimensions: (width, height)
            # PyTorch dimensions: (channels, height, width)
            transforms.CenterCrop((363, 360)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        folder = record["folder"]
        filename = record["filename"]

        image_path = self.root_dir / folder / filename

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = int(record["class_idx"])

        return image, label


# =====================================
# CREATE DATALOADERS
# =====================================

def create_dataloaders(
    root_dir="bloodcells_processed",
    batch_size=32,
    num_workers=0
):
    train_dataset = BloodCellDataset(
        root_dir=root_dir,
        split="train",
        augment=True
    )

    val_dataset = BloodCellDataset(
        root_dir=root_dir,
        split="val",
        augment=False
    )

    test_dataset = BloodCellDataset(
        root_dir=root_dir,
        split="test",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# =====================================
# TEST DATASET LOADER
# =====================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir="bloodcells_processed",
        batch_size=4
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels:", labels.tolist())

    assert images.shape[1:] == (3, 360, 363), (
        f"Unexpected image shape: {images.shape}"
    )

    print("\nDataset loader test passed!")
