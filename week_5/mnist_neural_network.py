"""Week 5: MNIST digit classification with a neural network."""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


DATASET_PATH = Path(__file__).parents[1] / "week4" / "mnist.npz"


def load_mnist(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST and flatten each 28x28 image into a feature vector."""
    with np.load(dataset_path) as data:
        x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
        y_train = data["y_train"]
        x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist(DATASET_PATH)

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=20,
        random_state=42,
        verbose=True,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("\n========== WEEK 5: MNIST NEURAL NETWORK ==========")
    print("Training images:", len(x_train))
    print("Testing images :", len(x_test))
    print("Accuracy       :", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("\nClassification Report")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nFirst 10 actual labels    :", y_test[:10])
    print("First 10 predicted labels:", predictions[:10])


if __name__ == "__main__":
    main()
