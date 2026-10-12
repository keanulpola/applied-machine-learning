"""Wine multiclass classification with a NumPy softmax model and mini-batch SGD."""

from pathlib import Path
import sys

# Use a non-GUI Matplotlib backend so plots can be saved without opening windows.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine


# Find the root folder of the project.
ROOT = Path(__file__).resolve().parents[1]

# Add the project root so Python can import files from src/.
sys.path.insert(0, str(ROOT))


# Import the machine-learning functions used in this experiment.
from src.ml_core import (  # noqa: E402
    confusion_matrix,
    kfold_indices,
    predict_softmax,
    standardize_train_test,
    train_softmax_sgd,
)


# Create the results folder if it does not already exist.
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    """Run the Wine classification experiment."""

    # Load the Wine dataset.
    wine = load_wine()
    X_all, y_all = wine.data, wine.target

    # Keep the first 40 samples from each of the three classes.
    selected = []

    for cls in np.unique(y_all):
        selected.extend(np.where(y_all == cls)[0][:40])

    X = X_all[selected]
    y = y_all[selected]

    # Test three different mini-batch sizes.
    batch_sizes = [1, 36, 72]

    # Create 10 folds for cross-validation.
    folds = kfold_indices(len(y), 10, seed=42)

    # Train each model for 100 epochs.
    epochs = 100

    # Store average training accuracy, test accuracy, and confusion matrices.
    mean_train_curves = []
    mean_test_curves = []
    averaged_confusions = []

    # Run the experiment once for each batch size.
    for batch_size in batch_sizes:
        train_curves = []
        test_curves = []
        cms = []

        # Use each fold once as the test set.
        for fold_number, test_idx in enumerate(folds):

            # Combine the other nine folds into the training set.
            train_idx = np.concatenate(
                [
                    fold
                    for i, fold in enumerate(folds)
                    if i != fold_number
                ]
            )

            # Split the data into training and testing portions.
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Calculate standardization values using only the training data.
            X_train, X_test, _, _ = standardize_train_test(
                X_train_raw,
                X_test_raw,
            )

            # Train the softmax classifier using mini-batch SGD.
            W, history = train_softmax_sgd(
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=batch_size,
                learning_rate=0.1,
                epochs=epochs,
                seed=1000 + batch_size * 10 + fold_number,
            )

            # Save the accuracy history for this fold.
            train_curves.append(history.train_accuracy)
            test_curves.append(history.test_accuracy)

            # Make predictions on the test fold.
            predictions = predict_softmax(W, X_test)

            # Build and save the confusion matrix for this fold.
            cms.append(
                confusion_matrix(
                    y_test,
                    predictions,
                    3,
                )
            )

        # Average the results across all 10 folds.
        mean_train_curves.append(
            np.mean(train_curves, axis=0)
        )

        mean_test_curves.append(
            np.mean(test_curves, axis=0)
        )

        averaged_confusions.append(
            np.mean(cms, axis=0)
        )

        # Print the final average test accuracy for this batch size.
        print(
            f"Batch {batch_size}: "
            f"final CV accuracy={mean_test_curves[-1][-1]:.3f}"
        )

    # Create an x-axis from epoch 1 through epoch 100.
    epoch_axis = np.arange(1, epochs + 1)

    # Plot the average training and testing accuracy.
    plt.figure()

    for batch_size, train_curve, test_curve in zip(
        batch_sizes,
        mean_train_curves,
        mean_test_curves,
    ):
        plt.plot(
            epoch_axis,
            train_curve,
            label=f"train, batch={batch_size}",
        )

        plt.plot(
            epoch_axis,
            test_curve,
            linestyle="--",
            label=f"test, batch={batch_size}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Wine Classification: 10-Fold Accuracy")
    plt.legend()
    plt.tight_layout()

    # Save the accuracy plot.
    plt.savefig(
        RESULTS / "wine_accuracy.png",
        dpi=200,
    )

    plt.close()

    # Create one confusion-matrix plot for each batch size.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3),
    )

    for ax, batch_size, cm in zip(
        axes,
        batch_sizes,
        averaged_confusions,
    ):
        # Display the confusion matrix.
        ax.imshow(cm)

        ax.set_title(f"Batch {batch_size}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Write each confusion-matrix value inside its cell.
        for r in range(3):
            for c in range(3):
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]:.1f}",
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    # Save the confusion-matrix figure.
    fig.savefig(
        RESULTS / "wine_confusion_matrices.png",
        dpi=200,
    )

    plt.close(fig)


# Run the experiment only when this file is executed directly.
if __name__ == "__main__":
    main()