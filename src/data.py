"""Data discovery, splitting, and image loading utilities.

The dataset is nested as dataset/Fruits/<class> and dataset/Vegetables/<class>.
Each second-level folder is a class label, for example FreshApple or RottenTomato.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def discover_images(dataset_dir: Path) -> pd.DataFrame:
    """Find all image files and assign labels from their class folder names."""
    dataset_dir = Path(dataset_dir)
    rows: list[dict[str, str]] = []

    for group_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "file_path": str(image_path.resolve()),
                            "label": class_dir.name,
                            "group": group_dir.name,
                        }
                    )

    if not rows:
        raise FileNotFoundError(
            f"No images were found under {dataset_dir}. Expected dataset/Fruits/<class> and "
            "dataset/Vegetables/<class> folders."
        )

    return pd.DataFrame(rows)


def make_class_indices(labels: Iterable[str]) -> dict[str, int]:
    """Create stable numeric ids for class names."""
    class_names = sorted(set(labels))
    return {class_name: index for index, class_name in enumerate(class_names)}


def add_label_indices(df: pd.DataFrame, class_indices: dict[str, int]) -> pd.DataFrame:
    """Attach integer class ids used by Keras categorical crossentropy."""
    df = df.copy()
    df["label_index"] = df["label"].map(class_indices)
    return df


def stratified_train_val_test_split(
    df: pd.DataFrame,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split images into train, validation, and test sets while preserving class ratios."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=validation_size + test_size,
        stratify=df["label"],
        random_state=random_seed,
        shuffle=True,
    )

    relative_test_size = test_size / (validation_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=random_seed,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class ImageClassificationSequence(tf.keras.utils.Sequence):
    """Keras data loader that resizes images and normalizes pixels to [0, 1].

    This class uses Pillow instead of TensorFlow's built-in directory loader so the
    project can read the nested Kaggle folders and WebP files reliably.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_indices: dict[str, int],
        image_size: tuple[int, int],
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_indices = class_indices
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = len(class_indices)
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch = self.df.iloc[batch_indices]

        images = np.zeros((len(batch), self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        labels = np.zeros((len(batch), self.num_classes), dtype=np.float32)

        for row_number, (_, row) in enumerate(batch.iterrows()):
            images[row_number] = load_image_as_array(row["file_path"], self.image_size)
            labels[row_number, int(row["label_index"])] = 1.0

        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_image_as_array(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load one RGB image, resize it, and scale pixel values to [0, 1]."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def save_class_indices(class_indices: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(class_indices, file, indent=2)


def load_class_indices(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, paths: tuple[Path, Path, Path]) -> None:
    train_path, val_path, test_path = paths
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


def load_splits(paths: tuple[Path, Path, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, val_path, test_path = paths
    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
