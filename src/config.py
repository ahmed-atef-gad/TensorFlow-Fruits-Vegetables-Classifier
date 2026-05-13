"""Project configuration for the fruits and vegetables CNN classifier."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
RANDOM_SEED = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

EPOCHS = 30
LEARNING_RATE = 1e-4

BEST_MODEL_PATH = MODELS_DIR / "best_fruits_vegetables_cnn.h5"
FINAL_MODEL_PATH = MODELS_DIR / "final_fruits_vegetables_cnn.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
TRAIN_SPLIT_PATH = MODELS_DIR / "train_split.csv"
VAL_SPLIT_PATH = MODELS_DIR / "val_split.csv"
TEST_SPLIT_PATH = MODELS_DIR / "test_split.csv"
