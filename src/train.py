"""Train the fruits and vegetables CNN classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_INDICES_PATH,
    DATASET_DIR,
    EPOCHS,
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
    LEARNING_RATE,
    PLOTS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    VALIDATION_SIZE,
    VAL_SPLIT_PATH,
)
from .data import (
    ImageClassificationSequence,
    add_label_indices,
    discover_images,
    make_class_indices,
    save_class_indices,
    save_splits,
    stratified_train_val_test_split,
)
from .model import build_cnn_model, print_parameter_report
from .visualize import plot_class_distribution, plot_sample_images, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on the fruits and vegetables dataset.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Square image size in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    manifest = discover_images(Path(args.dataset_dir))
    class_indices = make_class_indices(manifest["label"])
    manifest = add_label_indices(manifest, class_indices)

    print(f"Discovered {len(manifest):,} images across {len(class_indices)} classes.")
    print("Class indices:", class_indices)

    train_df, val_df, test_df = stratified_train_val_test_split(
        manifest,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

    save_class_indices(class_indices, CLASS_INDICES_PATH)
    save_splits(train_df, val_df, test_df, (TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_SPLIT_PATH))
    plot_class_distribution(manifest, PLOTS_DIR / "class_distribution.png")
    plot_sample_images(manifest, image_size, PLOTS_DIR / "sample_images.png")

    train_sequence = ImageClassificationSequence(
        train_df, class_indices, image_size=image_size, batch_size=args.batch_size, shuffle=True
    )
    val_sequence = ImageClassificationSequence(
        val_df, class_indices, image_size=image_size, batch_size=args.batch_size, shuffle=False
    )

    model = build_cnn_model(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=len(class_indices),
        learning_rate=args.learning_rate,
    )

    print("\nModel summary")
    model.summary()
    print_parameter_report(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    pd.DataFrame(history.history).to_csv(PLOTS_DIR / "training_history.csv", index=False)
    plot_training_curves(history, PLOTS_DIR / "training_curves.png")

    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved final model to: {FINAL_MODEL_PATH}")
    print(f"Saved plots and training history to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
