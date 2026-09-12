"""Week 6: MNIST digit classification with a convolutional neural network."""

from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ImportError as error:
    raise SystemExit(
        "TensorFlow is required for Week 6. Install it with: py -m pip install tensorflow"
    ) from error


DATASET_PATH = Path(__file__).parents[1] / "week4" / "mnist.npz"


def load_mnist(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, normalize pixels, and add the CNN channel dimension."""
    with np.load(dataset_path) as data:
        x_train = data["x_train"].astype(np.float32) / 255.0
        y_train = data["y_train"]
        x_test = data["x_test"].astype(np.float32) / 255.0
        y_test = data["y_test"]
    return x_train[..., np.newaxis], y_train, x_test[..., np.newaxis], y_test


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist(DATASET_PATH)
    model = build_model()
    model.summary()

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test[:10], verbose=0), axis=1)

    print("\n========== WEEK 6: MNIST CNN ==========")
    print("Test loss    :", round(float(loss), 4))
    print("Test accuracy:", round(float(accuracy) * 100, 2), "%")
    print("Actual labels    :", y_test[:10])
    print("Predicted labels:", predictions)


if __name__ == "__main__":
    main()
