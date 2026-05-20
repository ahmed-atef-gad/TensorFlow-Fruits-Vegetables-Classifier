# Fruits and Vegetables Freshness Classification

This project trains a TensorFlow/Keras Convolutional Neural Network (CNN) to classify images from the Kaggle **Fruits and Vegetables** dataset into 20 classes covering fresh and rotten fruits/vegetables.

The code is organized as a complete university Machine Learning assignment: data preparation, visualization, CNN training, evaluation, saved outputs, and single-image prediction.

## Dataset Structure

Place the dataset in the project root as:

```text
dataset/
├── Fruits/
│   ├── FreshApple
│   ├── FreshBanana
│   ├── FreshMango
│   ├── FreshOrange
│   ├── FreshStrawberry
│   ├── RottenApple
│   ├── RottenBanana
│   ├── RottenMango
│   ├── RottenOrange
│   └── RottenStrawberry
└── Vegetables/
    ├── FreshBellpepper
    ├── FreshCarrot
    ├── FreshCucumber
    ├── FreshPotato
    ├── FreshTomato
    ├── RottenBellpepper
    ├── RottenCarrot
    ├── RottenCucumber
    ├── RottenPotato
    └── RottenTomato
```

The loader treats each second-level folder, such as `FreshApple` or `RottenTomato`, as one class.

## Project Structure

```text
.
├── dataset/                    # Kaggle dataset
├── models/                     # Trained models, splits, reports
├── plots/                      # Data, training, and evaluation plots
├── notebooks/                  # Jupyter notebook version
├── src/
│   ├── config.py               # Paths and hyperparameters
│   ├── data.py                 # Data loading, splitting, normalization
│   ├── model.py                # CNN architecture
│   ├── visualize.py            # Matplotlib/seaborn plots
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Test evaluation
│   └── predict.py              # Single-image prediction
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- TensorFlow 2.x and Keras
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Jupyter Notebook

Use a Python version supported by your installed TensorFlow release. On native Windows, this project uses Python 3.10 with TensorFlow 2.10.1. For newer TensorFlow versions, use Linux, macOS, or WSL2.

## Model Architecture

The model is a CNN built from scratch:

- Input image size: `128 x 128 x 3`
- Data augmentation: horizontal flip, rotation, zoom, and shift
- Convolution blocks:
  - `Conv2D(32)` + `Conv2D(32)` + `MaxPooling2D` + `Dropout(0.25)`
  - `Conv2D(64)` + `Conv2D(64)` + `MaxPooling2D` + `Dropout(0.30)`
  - `Conv2D(128)` + `Conv2D(128)` + `MaxPooling2D` + `Dropout(0.35)`
- Classifier:
  - `Flatten`
  - `Dense(256, activation="relu")`
  - `Dropout(0.50)`
  - `Dense(20, activation="softmax")`

The training script prints the full Keras model summary and a layer-by-layer parameter report showing filters, neurons, weight shapes, and parameter counts.

## Training Configuration

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: categorical crossentropy
- Metrics: accuracy
- Epochs: `30` by default
- Split:
  - Training: 70%
  - Validation: 15%
  - Test: 15%
- Callbacks:
  - EarlyStopping on validation loss
  - ModelCheckpoint on validation accuracy
  - BackupAndRestore for interrupted training recovery
  - CSVLogger for epoch-by-epoch history

## How to Run

Create a new virtual environment first. On native Windows PowerShell, use Python 3.10 because native Windows TensorFlow wheels stopped after the 2.10 series.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you previously saw `No matching distribution found for tensorflow`, it means the environment was using a Python version without a compatible native Windows TensorFlow wheel. Recreate the environment with `py -3.10`, then verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Train the model:

```powershell
python -m src.train
```

Optional training arguments:

```powershell
python -m src.train --epochs 40 --batch-size 32 --learning-rate 0.0001 --image-size 128
```

If the terminal disconnects or training is interrupted, run the same command again. `BackupAndRestore` will try to continue from the last completed epoch:

```powershell
python -m src.train --epochs 50
```

You can also resume manually from a saved checkpoint:

```powershell
python -m src.train --epochs 50 --resume-from best
python -m src.train --epochs 50 --resume-from last
```

When you use `--resume-from last`, the value passed to `--epochs` is treated as the number of additional epochs to train after the last logged epoch.

For CPU training, image loading can run in background workers:

```powershell
python -m src.train --epochs 50 --workers 2
```

If the machine is stable and you want to try process-based loading:

```powershell
python -m src.train --epochs 50 --workers 2 --use-multiprocessing
```

On Windows, if multiprocessing causes instability, go back to:

```powershell
python -m src.train --epochs 50 --workers 1
```

Evaluate the best saved model:

```powershell
python -m src.evaluate
```

Predict a single image:

```powershell
python -m src.predict "dataset/Fruits/FreshApple/freshApple (1).jpg"
```

Run the local web interface:

```powershell
python app.py
```

Then open `http://127.0.0.1:5000`. The website lets you upload an image for prediction,
start training, and run evaluation. If you leave an argument field empty, the app uses
the same default values defined in `src/config.py` and the command-line parsers.

Run the notebook:

```powershell
jupyter notebook notebooks/fruits_vegetables_classification.ipynb
```

## Generated Outputs

Training creates:

- `models/best_fruits_vegetables_cnn.h5`
- `models/last_fruits_vegetables_cnn.h5`
- `models/final_fruits_vegetables_cnn.h5`
- `models/training_backup/`
- `models/class_indices.json`
- `models/train_split.csv`
- `models/val_split.csv`
- `models/test_split.csv`
- `plots/class_distribution.png`
- `plots/sample_images.png`
- `plots/training_curves.png`
- `plots/training_history.csv`
- `plots/training_log.csv`

Evaluation creates:

- `models/evaluation_report.txt`
- `models/classification_report.csv`
- `models/confusion_matrix.csv`
- `plots/confusion_matrix_heatmap.png`

## Example Outputs

After training, the terminal displays training and validation accuracy/loss for each epoch. Evaluation prints:

```text
Test loss: 0.1234
Test accuracy: 0.9650

Classification report:
...
```

Prediction prints:

```text
Predicted class: FreshApple
Prediction confidence: 98.42%
```

## Important ML Concepts Used

- **CNN:** Learns image features through convolution filters.
- **Neurons:** Units in dense layers that combine learned features for classification.
- **Weights:** Trainable numbers adjusted during backpropagation.
- **Activation functions:** ReLU introduces non-linearity; softmax produces class probabilities.
- **Optimizer:** Adam updates weights using gradients.
- **Learning rate:** Controls how large each update step is.
- **Dropout and augmentation:** Reduce overfitting by making the model less dependent on exact training images.
