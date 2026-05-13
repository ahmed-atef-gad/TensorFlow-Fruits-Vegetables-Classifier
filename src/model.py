"""CNN model definition for multi-class fruits and vegetables classification."""

from __future__ import annotations

import tensorflow as tf


def build_cnn_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
) -> tf.keras.Model:
    """Build and compile a CNN from scratch.

    CNN means Convolutional Neural Network. Convolution layers learn image filters,
    pooling layers reduce spatial size, dense layers combine learned features, and
    dropout helps prevent overfitting by randomly disabling neurons during training.
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", name="horizontal_flip"),
            tf.keras.layers.RandomRotation(0.12, name="random_rotation"),
            tf.keras.layers.RandomZoom(0.15, name="random_zoom"),
            tf.keras.layers.RandomTranslation(0.10, 0.10, name="random_shift"),
        ],
        name="training_data_augmentation",
    )

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)

    # Conv2D filters are learned weights that detect visual patterns such as edges,
    # colors, spots, bruises, and texture. ReLU keeps positive activations and adds
    # non-linearity so the network can model complex image patterns.
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_32_a")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_32_b")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_64_a")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_64_b")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_2")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_2")(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_128_a")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_128_b")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_3")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout_3")(x)

    # Flatten turns feature maps into a vector. Dense neurons learn combinations of
    # those features. Softmax converts the final logits into class probabilities.
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.50, name="dropout_4")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fruits_vegetables_cnn")

    # Adam adapts each weight's update size. The learning rate controls how large
    # every optimizer step is; 1e-4 is conservative for stable CNN training.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_parameter_report(model: tf.keras.Model) -> None:
    """Print layer-by-layer filters, neurons, weights, and parameter counts."""
    print("\nLayer parameter report")
    print("-" * 88)
    print(f"{'Layer':<30} {'Type':<18} {'Filters/Neurons':<18} {'Weight shapes':<20} {'Params':>8}")
    print("-" * 88)

    for layer in model.layers:
        if isinstance(layer, tf.keras.Sequential):
            print(f"{layer.name:<30} {'Sequential':<18} {'augmentation':<18} {'-':<20} {layer.count_params():>8}")
            continue

        filters_or_neurons = "-"
        if hasattr(layer, "filters"):
            filters_or_neurons = str(layer.filters)
        elif hasattr(layer, "units"):
            filters_or_neurons = str(layer.units)

        weight_shapes = ", ".join(str(tuple(weight.shape)) for weight in layer.weights) or "-"
        print(
            f"{layer.name:<30} {layer.__class__.__name__:<18} "
            f"{filters_or_neurons:<18} {weight_shapes:<20} {layer.count_params():>8}"
        )

    print("-" * 88)
    print(f"Total parameters: {model.count_params():,}")
