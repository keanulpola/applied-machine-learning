"""Small machine-learning algorithms implemented with NumPy.

This module is a post-course portfolio refactor of algorithms implemented
in graduate machine-learning coursework.  Dataset loading is kept outside
this module so the algorithms remain reusable and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using training-set statistics only.

    Works with either a 1-D feature vector or a 2-D ``(samples, features)``
    matrix. Zero-variance features receive scale 1.0.
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    axis = 0 if x_train.ndim == 2 else None
    mean = np.mean(x_train, axis=axis)
    std = np.std(x_train, axis=axis)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def radius_regression_1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Radius-based regression for a scalar feature.

    Training points are sorted once. Search boundaries are found with
    ``searchsorted`` and predictions are computed from prefix sums.
    If no point falls inside the radius, the training-target mean is used.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")

    x_train = np.asarray(x_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    x_test = np.asarray(x_test, dtype=float).ravel()
    if x_train.size != y_train.size or x_train.size == 0:
        raise ValueError("x_train and y_train must have equal non-zero length")

    order = np.argsort(x_train)
    xs = x_train[order]
    ys = y_train[order]
    prefix = np.concatenate(([0.0], np.cumsum(ys)))

    fallback = float(np.mean(ys))
    preds = np.empty_like(x_test, dtype=float)
    fallback_count = 0

    for i, x in enumerate(x_test):
        left = np.searchsorted(xs, x - radius, side="left")
        right = np.searchsorted(xs, x + radius, side="right")
        count = right - left
        if count:
            preds[i] = (prefix[right] - prefix[left]) / count
        else:
            preds[i] = fallback
            fallback_count += 1

    return preds, fallback_count / len(x_test) if len(x_test) else 0.0


def knn_regression_1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
) -> np.ndarray:
    """K-nearest-neighbor regression for a scalar feature.

    The training feature is sorted once.  Each query expands left/right from
    its insertion point, avoiding a full distance sort for every test sample.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    x_train = np.asarray(x_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    x_test = np.asarray(x_test, dtype=float).ravel()
    if x_train.size != y_train.size or x_train.size == 0:
        raise ValueError("x_train and y_train must have equal non-zero length")

    order = np.argsort(x_train)
    xs = x_train[order]
    ys = y_train[order]
    k_use = min(k, len(xs))
    preds = np.empty_like(x_test, dtype=float)

    for qi, x in enumerate(x_test):
        pos = int(np.searchsorted(xs, x))
        left, right = pos - 1, pos
        chosen: list[int] = []
        while len(chosen) < k_use:
            if left < 0:
                chosen.append(right)
                right += 1
            elif right >= len(xs):
                chosen.append(left)
                left -= 1
            elif abs(xs[left] - x) <= abs(xs[right] - x):
                chosen.append(left)
                left -= 1
            else:
                chosen.append(right)
                right += 1
        preds[qi] = np.mean(ys[chosen])

    return preds


def polynomial_features_1d(x: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, x, x^2, ..., x^degree] as sample rows."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x**power for power in range(degree + 1)])


def fit_least_squares_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Solve least squares with an explicit SVD pseudoinverse."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    U, s, vt = np.linalg.svd(X, full_matrices=False)
    cutoff = rcond * s[0] if s.size else 0.0
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
    return vt.T @ (s_inv * (U.T @ y))


def polynomial_exponents(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """All exponent tuples with total degree <= ``degree``."""
    if n_features <= 0 or degree < 0:
        raise ValueError("n_features must be positive and degree non-negative")
    exponents = [p for p in product(range(degree + 1), repeat=n_features) if sum(p) <= degree]
    exponents.sort(key=lambda p: (sum(p), p))
    return exponents


def multivariate_polynomial_features(
    X: np.ndarray, exponents: Iterable[tuple[int, ...]]
) -> np.ndarray:
    """Build a multivariate polynomial design matrix from sample rows."""
    X = np.asarray(X, dtype=float)
    exponents = list(exponents)
    H = np.ones((X.shape[0], len(exponents)), dtype=float)
    for col, powers in enumerate(exponents):
        for feature, power in enumerate(powers):
            if power:
                H[:, col] *= X[:, feature] ** power
    return H


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax across classes (last axis)."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).ravel()
    encoded = np.zeros((len(y), n_classes), dtype=float)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        matrix[truth, pred] += 1
    return matrix


@dataclass
class SoftmaxHistory:
    train_accuracy: np.ndarray
    test_accuracy: np.ndarray
    epoch_seconds: np.ndarray


def predict_softmax(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xb = np.column_stack((np.ones(len(X)), X))
    return np.argmax(softmax(Xb @ W), axis=1)


def train_softmax_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float = 0.1,
    epochs: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, SoftmaxHistory]:
    """Train multiclass linear softmax regression using mini-batch SGD."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    Xb = np.column_stack((np.ones(len(X_train)), X_train))
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    targets = one_hot(y_train, n_classes)
    W = np.zeros((Xb.shape[1], n_classes), dtype=float)
    rng = np.random.default_rng(seed)

    train_acc, test_acc, elapsed = [], [], []
    for _ in range(epochs):
        import time
        start = time.perf_counter()
        order = rng.permutation(len(X_train))
        for begin in range(0, len(X_train), batch_size):
            idx = order[begin : begin + batch_size]
            xb = Xb[idx]
            tb = targets[idx]
            probs = softmax(xb @ W)
            gradient = xb.T @ (probs - tb) / len(idx)
            W -= learning_rate * gradient

        train_acc.append(accuracy(y_train, predict_softmax(W, X_train)))
        test_acc.append(accuracy(y_test, predict_softmax(W, X_test)))
        elapsed.append(time.perf_counter() - start)

    return W, SoftmaxHistory(np.array(train_acc), np.array(test_acc), np.array(elapsed))


def kfold_indices(n_samples: int, n_splits: int, seed: int = 42) -> list[np.ndarray]:
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("n_splits must be between 2 and n_samples")
    rng = np.random.default_rng(seed)
    return [np.asarray(x) for x in np.array_split(rng.permutation(n_samples), n_splits)]
