"""Plotting utilities for dataset inspection, training history, and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import load_image_as_array


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(14, 7))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sample_images(df: pd.DataFrame, image_size: tuple[int, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = df.sort_values("label").groupby("label", as_index=False).first()

    cols = 5
    rows = (len(sample_df) + cols - 1) // cols
    plt.figure(figsize=(16, rows * 3))

    for index, row in sample_df.iterrows():
        plt.subplot(rows, cols, index + 1)
        plt.imshow(load_image_as_array(row["file_path"], image_size))
        plt.title(row["label"], fontsize=10)
        plt.axis("off")

    plt.suptitle("One Sample Image From Each Class", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curves(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["accuracy"], label="Training accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["loss"], label="Training loss")
    plt.plot(history_df["val_loss"], label="Validation loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(confusion_matrix, class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
