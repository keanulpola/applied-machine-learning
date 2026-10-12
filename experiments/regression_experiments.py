"""Regression experiments using public California Housing and Auto MPG data."""

from pathlib import Path
import sys

# Use a non-GUI Matplotlib backend so plots can be saved without opening windows.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# Find the root folder of the project.
ROOT = Path(__file__).resolve().parents[1]

# Add the project root so Python can import files from src/.
sys.path.insert(0, str(ROOT))


# Import the machine-learning functions used in these experiments.
from src.ml_core import (  # noqa: E402
    fit_least_squares_svd,
    knn_regression_1d,
    mse,
    multivariate_polynomial_features,
    polynomial_exponents,
    radius_regression_1d,
    standardize_train_test,
)


# Create the results folder if it does not already exist.
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def california_housing_experiment() -> None:
    """Run radius-based and KNN regression on California Housing data."""

    # Load the California Housing dataset.
    data = fetch_california_housing()

    # Use median income as the input feature.
    x = data.data[:, 0]

    # Use median house value as the target.
    y = data.target

    # Create a reproducible random ordering of the data.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(x))

    # Use 5,000 samples for training.
    train_idx = order[:5000]

    # Use a separate set of 5,000 samples for testing.
    test_idx = order[10000:15000]

    # Split the feature data.
    x_train = x[train_idx]
    x_test = x[test_idx]

    # Split the target values.
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Standardize the input feature using training-set statistics.
    x_train_s, x_test_s, _, _ = standardize_train_test(
        x_train,
        x_test,
    )

    # Test several radius values for radius-based regression.
    radii = np.array([
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        1.0,
    ])

    radius_mse = []

    # Train and evaluate the radius model for each radius.
    for radius in radii:
        pred, _ = radius_regression_1d(
            x_train_s,
            y_train,
            x_test_s,
            radius,
        )

        radius_mse.append(
            mse(y_test, pred)
        )

    # Test several K values for K-nearest-neighbor regression.
    ks = np.array([
        1,
        3,
        5,
        10,
        20,
        50,
        100,
        500,
    ])

    knn_mse = []

    # Train and evaluate the KNN model for each K value.
    for k in ks:
        pred = knn_regression_1d(
            x_train_s,
            y_train,
            x_test_s,
            int(k),
        )

        knn_mse.append(
            mse(y_test, pred)
        )

    # Plot test MSE for the different radius values.
    plt.figure()

    plt.plot(
        radii,
        radius_mse,
        marker="o",
    )

    plt.xlabel("Radius")
    plt.ylabel("Test MSE")
    plt.title("California Housing: Radius Regression")
    plt.tight_layout()

    # Save the radius-regression results.
    plt.savefig(
        RESULTS / "california_radius_mse.png",
        dpi=200,
    )

    plt.close()

    # Plot test MSE for the different K values.
    plt.figure()

    plt.plot(
        ks,
        knn_mse,
        marker="o",
    )

    plt.xlabel("K")
    plt.ylabel("Test MSE")
    plt.title("California Housing: KNN Regression")
    plt.tight_layout()

    # Save the KNN results.
    plt.savefig(
        RESULTS / "california_knn_mse.png",
        dpi=200,
    )

    plt.close()

    # Print the best test MSE from each regression method.
    print(
        "California Housing best radius MSE:",
        min(radius_mse),
    )

    print(
        "California Housing best KNN MSE:",
        min(knn_mse),
    )


def auto_mpg_experiment() -> None:
    """Run multivariate polynomial regression on the Auto MPG dataset."""

    # URL for the public Auto MPG dataset.
    url = (
        "https://archive.ics.uci.edu/"
        "ml/machine-learning-databases/"
        "auto-mpg/auto-mpg.data"
    )

    # Assign names to the dataset columns.
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]

    # Load the Auto MPG dataset from UCI.
    frame = pd.read_csv(
        url,
        sep=r"\s+",
        names=columns,
        na_values="?",
        quotechar='"',
    )

    # Remove rows with missing horsepower values.
    frame = frame.dropna(
        subset=["horsepower"]
    )

    # Select the numeric vehicle features used for prediction.
    features = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]

    # Convert the selected features into a NumPy array.
    X = frame[features].to_numpy(
        dtype=float
    )

    # Use MPG as the target value.
    y = frame["mpg"].to_numpy(
        dtype=float
    )

    # Create a reproducible random ordering of the data.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(X))

    # Use 300 samples for training and 92 for testing.
    train_idx = order[:300]
    test_idx = order[300:392]

    # Split the feature data.
    X_train = X[train_idx]
    X_test = X[test_idx]

    # Split the target data.
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Standardize the features using training-set statistics.
    X_train_s, X_test_s, _, _ = standardize_train_test(
        X_train,
        X_test,
    )

    # Test polynomial models with degrees 1, 2, and 3.
    degrees = [1, 2, 3]

    train_scores = []
    test_scores = []

    # Train and evaluate one model for each polynomial degree.
    for degree in degrees:

        # Generate all polynomial exponent combinations.
        exponents = polynomial_exponents(
            X_train_s.shape[1],
            degree,
        )

        # Create polynomial features for the training data.
        H_train = multivariate_polynomial_features(
            X_train_s,
            exponents,
        )

        # Create matching polynomial features for the test data.
        H_test = multivariate_polynomial_features(
            X_test_s,
            exponents,
        )

        # Fit the regression weights using an SVD-based least-squares solution.
        w = fit_least_squares_svd(
            H_train,
            y_train,
        )

        # Calculate training MSE.
        train_prediction = H_train @ w

        train_scores.append(
            mse(
                y_train,
                train_prediction,
            )
        )

        # Calculate test MSE.
        test_prediction = H_test @ w

        test_scores.append(
            mse(
                y_test,
                test_prediction,
            )
        )

        # Print the result for this polynomial degree.
        print(
            f"Auto MPG degree {degree}: "
            f"train MSE={train_scores[-1]:.4f}, "
            f"test MSE={test_scores[-1]:.4f}, "
            f"terms={len(exponents)}"
        )

    # Plot training and testing MSE by polynomial degree.
    plt.figure()

    plt.plot(
        degrees,
        train_scores,
        marker="o",
        label="Train",
    )

    plt.plot(
        degrees,
        test_scores,
        marker="o",
        label="Test",
    )

    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Auto MPG: Multivariate Polynomial Regression")
    plt.legend()
    plt.tight_layout()

    # Save the Auto MPG regression plot.
    plt.savefig(
        RESULTS / "auto_mpg_multivariate_mse.png",
        dpi=200,
    )

    plt.close()


# Run both regression experiments when this file is executed directly.
if __name__ == "__main__":
    california_housing_experiment()
    auto_mpg_experiment()