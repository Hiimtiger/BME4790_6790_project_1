
from pathlib import Path
from collections import Counter
import csv

from PIL import Image
from tqdm.auto import tqdm


# =====================================
# SETTINGS
# =====================================

DATASET_DIR = Path("bloodcells_processed")

# PIL uses (width, height)
EXPECTED_SIZE = (363, 360)

OUTPUT_CSV = Path("results/image_size_report.csv")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# =====================================
# CHECK IMAGE DIMENSIONS
# =====================================

def check_image_sizes():

    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {DATASET_DIR}")
        return

    size_counts = Counter()
    mismatched_images = []
    unreadable_images = []

    print(f"\nScanning {len(image_paths)} images...\n")

    for image_path in tqdm(
        image_paths,
        desc="Checking image sizes",
        unit="image",
    ):
        try:
            with Image.open(image_path) as image:
                width, height = image.size

            size_counts[(width, height)] += 1

            if (width, height) != EXPECTED_SIZE:
                mismatched_images.append({
                    "filename": image_path.name,
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                    "expected_width": EXPECTED_SIZE[0],
                    "expected_height": EXPECTED_SIZE[1],
                })

        except Exception as error:
            unreadable_images.append({
                "path": str(image_path),
                "error": str(error),
            })

    # =====================================
    # SUMMARY
    # =====================================

    print("\n" + "=" * 55)
    print("IMAGE SIZE CHECK RESULTS")
    print("=" * 55)

    print(f"Total images: {len(image_paths)}")
    print(f"Expected size (W x H): {EXPECTED_SIZE}")

    print("\nDetected image sizes:")

    for size, count in size_counts.most_common():
        print(
            f"  {size[0]} x {size[1]}: "
            f"{count} images"
        )

    print(f"\nMismatched images: {len(mismatched_images)}")
    print(f"Unreadable images: {len(unreadable_images)}")

    # =====================================
    # PRINT MISMATCHES
    # =====================================

    if mismatched_images:
        print("\nFirst 20 mismatched images:")

        for item in mismatched_images[:20]:
            print(
                f"  {item['path']} "
                f"({item['width']} x {item['height']})"
            )
    else:
        print("\nAll readable images match the expected size!")

    # =====================================
    # SAVE CSV REPORT
    # =====================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "path",
                "width",
                "height",
                "expected_width",
                "expected_height",
            ],
        )

        writer.writeheader()
        writer.writerows(mismatched_images)

    print(f"\nMismatch report saved to: {OUTPUT_CSV}")

    if unreadable_images:
        print("\nUnreadable images:")
        for item in unreadable_images[:20]:
            print(f"  {item['path']}: {item['error']}")

    print("=" * 55)


if __name__ == "__main__":
    check_image_sizes()
