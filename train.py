import argparse
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import DIGITS, DigitModel


def shift(images, columns):
    # Pad the 8x8 grid so pixels that leave one edge do not wrap to the other.
    padded = np.pad(images.reshape(-1, 8, 8), ((0, 0), (0, 0), (1, 1)))
    return padded[:, :, 1 - columns:9 - columns].reshape(-1, 64)


def train(path):
    images, digits = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        images, digits, test_size=0.25, stratify=digits, random_state=17
    )
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    baseline.fit(x_train, y_train)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    training_images = np.concatenate([x_train, shift(x_train, -1), shift(x_train, 1)])
    pipeline.fit(training_images, np.tile(y_train, 3))
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    if not np.array_equal(classifier.classes_, np.arange(DIGITS)):
        raise RuntimeError("The artifact requires all ten digit classes in order.")

    shifted_test = np.concatenate([shift(x_test, -1), shift(x_test, 1)])
    shifted_digits = np.tile(y_test, 2)
    evaluation = np.concatenate([x_test, shifted_test])
    expected = pipeline.predict_proba(evaluation)
    clean_probabilities = expected[:len(x_test)]
    labels = classifier.classes_[clean_probabilities.argmax(axis=1)]
    majority = np.bincount(y_train).argmax()
    metrics = {
        "train_rows": len(x_train),
        "fit_rows": len(training_images),
        "test_rows": len(x_test),
        "shifted_test_rows": len(shifted_test),
        "accuracy": accuracy_score(y_test, labels),
        "macro_f1": f1_score(y_test, labels, average="macro"),
        "log_loss": log_loss(y_test, clean_probabilities),
        "majority_accuracy": float((y_test == majority).mean()),
        "shifted_accuracy": accuracy_score(shifted_digits, expected[len(x_test):].argmax(axis=1)),
        "clean_only_model_accuracy": baseline.score(x_test, y_test),
        "clean_only_model_shifted_accuracy": baseline.score(shifted_test, shifted_digits),
    }
    artifact = {
        "schema_version": 1,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "split_seed": 17,
        "scikit_learn": version("scikit-learn"),
        "holdout": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        file.write(serialized)
    try:
        deployed = DigitModel.load(temporary).probabilities(evaluation)
        np.testing.assert_allclose(deployed, expected, atol=1e-12, rtol=1e-10)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and check the digit classifier artifact.")
    parser.add_argument("--out", type=Path, default=Path("model.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.out), indent=2))
