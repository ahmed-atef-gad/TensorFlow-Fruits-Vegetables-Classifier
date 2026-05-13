"""Evaluate a trained CNN on the held-out test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_INDICES_PATH,
    IMAGE_SIZE,
    MODELS_DIR,
    PLOTS_DIR,
    TEST_SPLIT_PATH,
)
from .data import ImageClassificationSequence, load_class_indices
from .visualize import plot_confusion_matrix_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN on the test split.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to trained .h5 model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Square image size in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    image_size = (args.image_size, args.image_size)

    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError("Test split was not found. Run training first to create saved splits.")

    class_indices = load_class_indices(CLASS_INDICES_PATH)
    index_to_class = {index: class_name for class_name, index in class_indices.items()}
    class_names = [index_to_class[index] for index in range(len(index_to_class))]

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    test_sequence = ImageClassificationSequence(
        test_df, class_indices, image_size=image_size, batch_size=args.batch_size, shuffle=False
    )

    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_sequence, verbose=1)
    probabilities = model.predict(test_sequence, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_df["label_index"].to_numpy()

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_text = (
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report_text}\n"
    )
    (MODELS_DIR / "evaluation_report.txt").write_text(metrics_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(MODELS_DIR / "classification_report.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(MODELS_DIR / "confusion_matrix.csv")
    plot_confusion_matrix_heatmap(cm, class_names, PLOTS_DIR / "confusion_matrix_heatmap.png")

    print(metrics_text)
    print(f"Saved evaluation report to: {MODELS_DIR / 'evaluation_report.txt'}")
    print(f"Saved confusion matrix heatmap to: {PLOTS_DIR / 'confusion_matrix_heatmap.png'}")


if __name__ == "__main__":
    main()
