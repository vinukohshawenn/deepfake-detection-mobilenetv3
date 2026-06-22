"""
Evaluate the trained deepfake detector on the held-out test set.

Usage:
    python evaluate.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_pipeline

config.set_seeds()


def main():

    print("Loading test dataset...")
    _, _, test_ds = data_pipeline.build_datasets()
    test_ds = data_pipeline.prepare(test_ds, training=False)

    model_path = (
        config.BEST_MODEL_PATH
        if os.path.exists(config.BEST_MODEL_PATH)
        else config.MODEL_PATH
    )

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).flatten()

        y_prob.extend(probs)
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n========== TEST RESULTS ==========\n")

    print(classification_report(
        y_true,
        y_pred,
        target_names=["Real", "Fake"],
        digits=4,
    ))

    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC : {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix\n")
    print(cm)

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(im)
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            config.OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    ax2.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}",
    )

    ax2.plot(
        [0, 1],
        [0, 1],
        "--",
    )

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")

    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(
        os.path.join(
            config.OUTPUT_DIR,
            "roc_curve.png"
        )
    )

    print("\nSaved:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")


if __name__ == "__main__":
    main()