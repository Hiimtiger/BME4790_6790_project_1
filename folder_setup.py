
from pathlib import Path

# =====================================
# PROJECT FOLDER SETUP
# =====================================

ROOT = Path(__file__).resolve().parent

# Directories to create
FOLDERS = [
    "utils",
    "saved_model",
    "results",
    "results/cnn_2_layer",
    "results/cnn_2_layer/gradcam",
    "results/cnn_3_layer",
    "results/cnn_3_layer/gradcam",
    "results/cnn_4_layer",
    "results/cnn_4_layer/gradcam",
]

# Other project files
OTHER_FILES = [
    "requirements.txt",
    ".gitignore",
]

def main():
    print("Setting up blood cell classification project...\n")

    # Create directories
    for folder in FOLDERS:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {folder}")

    # Add common dependencies if requirements.txt is empty
    requirements = ROOT / "requirements.txt"

    if requirements.stat().st_size == 0:
        requirements.write_text(
            "torch\n"
            "torchvision\n"
            "numpy\n"
            "pandas\n"
            "matplotlib\n"
            "seaborn\n"
            "scikit-learn\n"
            "Pillow\n"
            "tqdm\n",
            encoding="utf-8",
        )

    # Add gitignore rules if file is empty
    gitignore = ROOT / ".gitignore"

    if gitignore.stat().st_size == 0:
        gitignore.write_text(
            "__pycache__/\n"
            "*.pyc\n"
            ".venv/\n"
            "bloodcells_dataset/\n"
            "bloodcells_processed/\n"
            "saved_model/*.pth\n",
            encoding="utf-8",
        )

    print("\nProject setup complete!")
    print("Existing files were not overwritten.")


if __name__ == "__main__":
    main()
