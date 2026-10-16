
from pathlib import Path
from collections import Counter
import csv
import json
import shutil

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# =====================================
# 1. CONFIGURATION
# =====================================

SOURCE_DIR = Path("bloodcells_dataset")
OUTPUT_DIR = Path("bloodcells_processed")

TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
RANDOM_SEED = 42

# Folder names and standardized filename prefixes
CLASS_CONFIG = {
    "basophil": {
        "prefix": "BA",
        "accepted": {"BA"},
    },
    "eosinophil": {
        "prefix": "EO",
        "accepted": {"EO"},
    },
    "erythroblast": {
        "prefix": "ERB",
        "accepted": {"ERB"},
    },
    "ig": {
        "prefix": "IG",
        "accepted": {"IG", "MMY", "MY", "PMY"},
    },
    "lymphocyte": {
        "prefix": "LY",
        "accepted": {"LY"},
    },
    "monocyte": {
        "prefix": "MO",
        "accepted": {"MO"},
    },
    "neutrophil": {
        "prefix": "BNE",
        "accepted": {"BNE", "NEURTOPHIL", "NEUTROPHIL", "SNE"},
    },
    "platelet": {
        "prefix": "PLATELET",
        "accepted": {"PLATELET"},
    },
}

CLASS_NAMES = list(CLASS_CONFIG.keys())
CLASS_TO_IDX = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# =====================================
# 2. SCAN AND VALIDATE IMAGES
# =====================================

def scan_images():
    records = []

    for class_name, config in CLASS_CONFIG.items():
        folder = SOURCE_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        count = 0

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            original_prefix = path.stem.rsplit("_", 1)[0].upper()

            if original_prefix not in config["accepted"]:
                raise ValueError(
                    f"Unexpected prefix '{original_prefix}' "
                    f"in {path}. Check the filename."
                )

            try:
                with Image.open(path) as img:
                    img.verify()

                with Image.open(path) as img:
                    img.convert("RGB").load()

            except (UnidentifiedImageError, OSError) as exc:
                print(f"Skipping unreadable image: {path}: {exc}")
                continue

            records.append({
                "source": path,
                "class_name": class_name,
                "class_idx": CLASS_TO_IDX[class_name],
                "original_prefix": original_prefix,
            })

            count += 1

        print(f"{class_name}: {count} valid images")

    if not records:
        raise RuntimeError("No valid images found.")

    return records


# =====================================
# 3. STRATIFIED TRAIN / VAL / TEST SPLIT
# =====================================

def split_records(records):
    labels = [r["class_idx"] for r in records]

    train_pool, test_records = train_test_split(
        records,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    train_labels = [r["class_idx"] for r in train_pool]

    train_records, val_records = train_test_split(
        train_pool,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        stratify=train_labels,
        random_state=RANDOM_SEED,
    )

    return train_records, val_records, test_records


# =====================================
# 4. COPY AND STANDARDIZE FILENAMES
# =====================================

def save_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def copy_images(train_records, val_records, test_records):
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename or remove it before running again."
        )

    (OUTPUT_DIR / "train").mkdir(parents=True)
    (OUTPUT_DIR / "test").mkdir(parents=True)

    mapping = []
    manifests = {"train": [], "val": [], "test": []}

    # A single counter per class ensures unique names
    counters = Counter()

    for split, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        physical_folder = "test" if split == "test" else "train"

        for record in records:
            class_name = record["class_name"]
            counters[class_name] += 1

            prefix = CLASS_CONFIG[class_name]["prefix"]

            new_name = (
                f"{prefix}_{counters[class_name]:06d}.jpg"
            )

            destination = (
                OUTPUT_DIR / physical_folder / new_name
            )

            # Copy original JPEG without re-encoding
            shutil.copy2(record["source"], destination)

            row = {
                "filename": new_name,
                "class_name": class_name,
                "class_idx": record["class_idx"],
                "folder": physical_folder,
            }

            manifests[split].append(row)

            mapping.append({
                "split": split,
                "original_path": str(record["source"]),
                "original_prefix": record["original_prefix"],
                "new_filename": new_name,
                "class_name": class_name,
            })

    columns = ["filename", "class_name", "class_idx", "folder"]

    for split, rows in manifests.items():
        save_csv(
            OUTPUT_DIR / f"{split}.csv",
            rows,
            columns,
        )

    save_csv(
        OUTPUT_DIR / "filename_mapping.csv",
        mapping,
        [
            "split",
            "original_path",
            "original_prefix",
            "new_filename",
            "class_name",
        ],
    )

    return manifests


# =====================================
# 5. TRAINING-ONLY RGB MEAN AND STD
# =====================================

def calculate_rgb_stats(train_manifest):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for row in train_manifest:
        path = OUTPUT_DIR / "train" / row["filename"]

        with Image.open(path) as img:
            rgb = np.asarray(
                img.convert("RGB"),
                dtype=np.float64,
            ) / 255.0

        pixels = rgb.reshape(-1, 3)

        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - mean ** 2
    std = np.sqrt(np.maximum(variance, 0))

    if np.any(std < 1e-8):
        raise ValueError("A channel has near-zero standard deviation.")

    return mean, std


# =====================================
# 6. MAIN
# =====================================

def main():
    records = scan_images()

    train_records, val_records, test_records = split_records(records)

    print("\nDataset split:")
    print("Train:", len(train_records))
    print("Validation:", len(val_records))
    print("Test:", len(test_records))

    manifests = copy_images(
        train_records,
        val_records,
        test_records,
    )

    mean, std = calculate_rgb_stats(manifests["train"])

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "class_to_idx": CLASS_TO_IDX,
        "seed": RANDOM_SEED,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
    }

    with open(
        OUTPUT_DIR / "normalization.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats, f, indent=2)

    print("\nRGB mean:", mean)
    print("RGB std:", std)
    print("\nDone! Output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
