"""Predict the class of a single fruit or vegetable image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import BEST_MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE
from .data import load_class_indices, load_image_as_array


def predict_single_image(
    image_path: str | Path,
    model_path: str | Path = BEST_MODEL_PATH,
    class_indices_path: str | Path = CLASS_INDICES_PATH,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[str, float]:
    """Return the predicted class name and confidence for one image."""
    model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(Path(class_indices_path))
    index_to_class = {index: class_name for class_name, index in class_indices.items()}

    image = load_image_as_array(image_path, image_size)
    batch = np.expand_dims(image, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return index_to_class[predicted_index], confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one image with the trained CNN.")
    parser.add_argument("image_path", help="Path to an image file.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to trained model.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Square image size in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted_class, confidence = predict_single_image(
        image_path=args.image_path,
        model_path=args.model_path,
        image_size=(args.image_size, args.image_size),
    )
    print(f"Predicted class: {predicted_class}")
    print(f"Prediction confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
