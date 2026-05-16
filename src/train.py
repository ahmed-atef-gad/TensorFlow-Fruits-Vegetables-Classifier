"""Train the fruits and vegetables CNN classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import (
    BATCH_SIZE,
    BACKUP_DIR,
    BEST_MODEL_PATH,
    CLASS_INDICES_PATH,
    DATASET_DIR,
    EPOCHS,
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
    LAST_MODEL_PATH,
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
    parser.add_argument(
        "--resume-from",
        choices=["none", "best", "last"],
        default="none",
        help="Resume from a saved checkpoint. BackupAndRestore also resumes interrupted runs automatically.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of background workers for loading image batches. Use 1 if your machine is unstable.",
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Use process-based loading. On Windows, try the default threaded workers first.",
    )
    parser.add_argument("--max-queue-size", type=int, default=8, help="Maximum queued batches for Keras workers.")
    return parser.parse_args()


def history_dataframe(history, csv_log_path: Path) -> pd.DataFrame:
    """Return a clean metric table for plots and assignment output."""
    metric_columns = ["loss", "accuracy", "val_loss", "val_accuracy"]
    if csv_log_path.exists():
        csv_history = pd.read_csv(csv_log_path)
        if all(column in csv_history.columns for column in metric_columns):
            clean_history = csv_history.copy()
            if "epoch" in clean_history.columns:
                clean_history = clean_history.drop_duplicates(subset=["epoch"], keep="last")
            return clean_history.reset_index(drop=True)

    return pd.DataFrame(history.history)


def resume_epoch_from_log(csv_log_path: Path) -> int:
    """Return the next epoch index after the latest logged epoch."""
    if not csv_log_path.exists():
        return 0

    csv_history = pd.read_csv(csv_log_path)
    if "epoch" not in csv_history.columns or csv_history.empty:
        return 0

    return int(csv_history["epoch"].max()) + 1


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    history_log_path = PLOTS_DIR / "training_history.csv"
    resumable_log_path = PLOTS_DIR / "training_log.csv"
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

    resume_path = None
    if args.resume_from == "best":
        resume_path = BEST_MODEL_PATH
    elif args.resume_from == "last":
        resume_path = LAST_MODEL_PATH

    if resume_path is not None and resume_path.exists():
        resume_epoch = resume_epoch_from_log(resumable_log_path)
        print(f"\nResuming from checkpoint: {resume_path}")
        if resume_epoch > 0:
            print(f"Continuing from epoch {resume_epoch + 1}")
        model = tf.keras.models.load_model(resume_path)
    else:
        if resume_path is not None:
            print(f"\nCheckpoint not found at {resume_path}. Starting a new model.")
        resume_epoch = 0
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(LAST_MODEL_PATH),
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(BACKUP_DIR),
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(resumable_log_path),
            append=args.resume_from != "none" or BACKUP_DIR.exists() or resumable_log_path.exists(),
        ),
    ]

    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        initial_epoch=resume_epoch,
        epochs=resume_epoch + args.epochs,
        callbacks=callbacks,
        verbose=1,
        workers=args.workers,
        use_multiprocessing=args.use_multiprocessing,
        max_queue_size=args.max_queue_size,
    )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    clean_history = history_dataframe(history, resumable_log_path)
    clean_history.to_csv(history_log_path, index=False)
    plot_training_curves(clean_history, PLOTS_DIR / "training_curves.png")

    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved last epoch model to: {LAST_MODEL_PATH}")
    print(f"Saved final model to: {FINAL_MODEL_PATH}")
    print(f"Automatic interruption backup directory: {BACKUP_DIR}")
    print(f"Saved plots and training history to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
