"""
Run inference on one image or an entire folder.

Examples
--------
python predict.py image.jpg

python predict.py data/Test/Fake

python predict.py data/Test/Real
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

import config

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


def load_model():

    model_path = (
        config.BEST_MODEL_PATH
        if os.path.exists(config.BEST_MODEL_PATH)
        else config.MODEL_PATH
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found.\n"
            "Run train.py first."
        )

    print(f"\nLoading model: {model_path}")

    return tf.keras.models.load_model(model_path)


def predict_image(model, image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=config.IMG_SIZE,
    )

    image = tf.keras.utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)

    start = time.perf_counter()

    prob_fake = float(
        model.predict(
            image,
            verbose=0,
        )[0][0]
    )

    elapsed = (time.perf_counter() - start) * 1000

    label = "FAKE" if prob_fake >= 0.5 else "REAL"

    confidence = (
        prob_fake
        if label == "FAKE"
        else 1.0 - prob_fake
    )

    return label, confidence, prob_fake, elapsed


def get_image_paths(target):

    if os.path.isfile(target):

        return [target]

    if os.path.isdir(target):

        return sorted(
            os.path.join(target, file)
            for file in os.listdir(target)
            if file.lower().endswith(
                VALID_EXTENSIONS
            )
        )

    raise FileNotFoundError(
        f"'{target}' does not exist."
    )


def main():

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python predict.py image.jpg\n"
            "python predict.py folder/\n"
        )

        sys.exit(1)

    model = load_model()

    paths = get_image_paths(sys.argv[1])

    print(f"\nFound {len(paths)} image(s).\n")

    fake_count = 0
    real_count = 0

    for path in paths:

        try:

            label, confidence, prob_fake, t = predict_image(
                model,
                path,
            )

            if label == "FAKE":
                fake_count += 1
            else:
                real_count += 1

            print(
                f"{os.path.basename(path):25}"
                f"{label:5}   "
                f"Confidence: {confidence:.2%}   "
                f"P(fake): {prob_fake:.3f}   "
                f"{t:.1f} ms"
            )

        except Exception as e:

            print(f"Skipping {path}: {e}")

    print("\n-----------------------------")
    print(f"REAL : {real_count}")
    print(f"FAKE : {fake_count}")
    print("-----------------------------")


if __name__ == "__main__":
    main()