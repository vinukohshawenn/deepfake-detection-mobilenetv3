"""
Data loading utilities.

Expected folder structure:

data/
├── Train/
│   ├── Real/
│   └── Fake/
│
└── Test/
    ├── Real/
    └── Fake/

Labels:
    Real = 0
    Fake = 1
"""

import tensorflow as tf
import config


def build_datasets():
    """
    Build train, validation and test datasets.

    - Training set comes from data/Train
    - Validation is created from 10% of data/Train
    - Test set comes from data/Test
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=config.VAL_SPLIT,
        subset="training",
        seed=config.SEED,
        labels="inferred",
        label_mode="binary",
        class_names=["Real", "Fake"],
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=config.VAL_SPLIT,
        subset="validation",
        seed=config.SEED,
        labels="inferred",
        label_mode="binary",
        class_names=["Real", "Fake"],
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["Real", "Fake"],
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def augment(image, label):
    """
    Training-time augmentation.
    """

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)

    # Apply JPEG compression only 30% of the time
    if tf.random.uniform([]) < 0.3:
        image = tf.map_fn(
            lambda img: tf.image.random_jpeg_quality(
                img,
                min_jpeg_quality=80,
                max_jpeg_quality=100,
            ),
            image,
        )

    return image, label


def prepare(ds, training=False):
    """
    Prepare datasets for training/evaluation.
    """

    if training:
        ds = ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds = ds.prefetch(tf.data.AUTOTUNE)

    else:
        ds = ds.cache().prefetch(tf.data.AUTOTUNE)

    return ds