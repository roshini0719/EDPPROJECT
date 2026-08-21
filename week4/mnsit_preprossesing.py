from pathlib import Path

import numpy as np


def load_and_normalize_mnist(
	dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Load the local MNIST archive and scale image pixels to [0, 1]."""
	with np.load(dataset_path) as data:
		x_train = data["x_train"].astype(np.float32) / 255.0
		y_train = data["y_train"]
		x_test = data["x_test"].astype(np.float32) / 255.0
		y_test = data["y_test"]

	return x_train, y_train, x_test, y_test


if __name__ == "__main__":
	dataset_path = Path(__file__).with_name("mnist.npz")
	x_train, y_train, x_test, y_test = load_and_normalize_mnist(dataset_path)

	print("Training images:", x_train.shape)
	print("Training labels:", y_train.shape)
	print("Testing images:", x_test.shape)
	print("Testing labels:", y_test.shape)
	print("Image dtype:", x_train.dtype)
	print("Pixel range:", x_train.min(), "to", x_train.max())