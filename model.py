"""
Model architecture:
MobileNetV3-Small (ImageNet pretrained)
+
custom binary classification head.

Transfer learning pipeline:
1. Train classification head with frozen backbone.
2. Fine-tune the top MobileNetV3 layers using a low learning rate.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import config


def build_model():

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=config.IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        include_preprocessing=True,
    )

    base.trainable = False

    inputs = layers.Input(shape=config.IMG_SIZE + (3,))

    x = base(inputs, training=False)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="fake_probability",
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="deepfake_detector",
    )

    return model, base


def compile_for_phase1(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LR
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def unfreeze_for_finetune(model, base):

    base.trainable = True

    for layer in base.layers[:-config.UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    # Keep BatchNorm layers frozen
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.FINE_TUNE_LR
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model