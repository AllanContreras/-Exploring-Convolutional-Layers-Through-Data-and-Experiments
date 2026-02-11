import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(data_dir: Optional[str] = None):
    """Load data from SageMaker input channel (if provided) or fallback to Fashion-MNIST."""
    if data_dir and os.path.isdir(data_dir):
        # Expect npy files if provided. Otherwise fallback.
        x_train_path = os.path.join(data_dir, "x_train.npy")
        y_train_path = os.path.join(data_dir, "y_train.npy")
        x_test_path = os.path.join(data_dir, "x_test.npy")
        y_test_path = os.path.join(data_dir, "y_test.npy")
        if all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
            x_train = np.load(x_train_path)
            y_train = np.load(y_train_path)
            x_test = np.load(x_test_path)
            y_test = np.load(y_test_path)
        else:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def build_model(kernel_size: int = 3, dropout: float = 0.3, lr: float = 1e-3):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (kernel_size, kernel_size), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (kernel_size, kernel_size), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(dropout),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]) 
    return model


def main(args):
    set_seeds(args.seed)

    # SageMaker paths
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN")  # optional

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_data(train_data_dir)

    model = build_model(kernel_size=args.kernel_size, dropout=args.dropout, lr=args.lr)
    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_split=0.2,
                        verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    # Save model
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)

    # Save metrics
    metrics = {
        "history": history.history,
        "summary": {
            "test_acc": float(acc),
            "test_loss": float(loss)
        }
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
