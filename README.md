# BME4790_6790_project_1
code for Cornell 2026 Fall BME 4790/6790 Project 1 Image classification /
please put the dataset: "bloodcells_processed" in the folder as well. relative paths are alreadt written in code, so no need to change anything
# Blood Cell Classification Using CNNs

This project compares three convolutional neural networks (CNNs) with **2, 3, and 4 convolutional layers** for classifying eight types of blood cells.

## 1. Install Required Libraries

Run the following command in your terminal:

```bash
python -m pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm
```

## 2. Train the Models

Run the following commands from the project's root directory.

### 2-Layer CNN

```bash
python main_2_layer.py
```

Trains and evaluates a CNN with **2 convolutional layers**.

### 3-Layer CNN

```bash
python main_3_layer.py
```

Trains and evaluates a CNN with **3 convolutional layers**.

### 4-Layer CNN

```bash
python main_4_layer.py
```

Trains and evaluates a CNN with **4 convolutional layers**.

Each script automatically uses a CUDA-compatible GPU if available, otherwise it uses the CPU.

## 3. Run Inference

The `inference.py` script uses a previously trained CNN to classify a single blood cell image.

Before running, open `inference.py` and configure:

- `NUM_CONV_LAYERS`: Choose `2`, `3`, or `4`.
- `IMAGE_PATH`: Path to the image you want to classify.
- `GENERATE_GRADCAM`: Set to `True` to generate a Grad-CAM visualization.

Run:

```bash
python inference.py
```

The script outputs the predicted blood cell class, confidence score, and class probabilities.

Inference results are saved in:

```text
results/inference/
```

## 4. Saved Models

The best-performing checkpoint for each CNN, selected by **lowest validation loss**, is saved automatically in the `saved_model/` folder.

```text
saved_model/
├── cnn_2_layer_best.pth
├── cnn_3_layer_best.pth
└── cnn_4_layer_best.pth
```

These `.pth` files contain the trained model weights and checkpoint information.

## 5. Training and Evaluation Results

Training curves, evaluation metrics, confusion matrices, and Grad-CAM visualizations are saved in:

```text
results/
├── cnn_2_layer/
├── cnn_3_layer/
├── cnn_4_layer/
└── inference/
```

Each model's results folder contains training history, accuracy and loss plots, classification metrics, and confusion matrices.
