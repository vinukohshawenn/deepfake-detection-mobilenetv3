"""
Train the Deepfake Detector.

Pipeline
--------
1. Train the custom classifier head.
2. Fine-tune the top MobileNetV3 layers.
3. Save the best checkpoint and final model.

Usage
-----
python train.py
"""

import json
import os
import time

import tensorflow as tf

import config
import data_pipeline
import model as model_lib

config.set_seeds()


def save_history(*histories):

    history = {}

    for h in histories:
        for key, values in h.history.items():
            history.setdefault(key, []).extend(values)

    with open(
        os.path.join(config.OUTPUT_DIR, "history.json"),
        "w",
    ) as f:

        json.dump(history, f, indent=4)


def main():

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    train_ds, val_ds, _ = data_pipeline.build_datasets()

    train_ds = data_pipeline.prepare(
        train_ds,
        training=True,
    )

    val_ds = data_pipeline.prepare(
        val_ds,
        training=False,
    )

    print("\nBuilding MobileNetV3 model...\n")

    model, base = model_lib.build_model()

    model = model_lib.compile_for_phase1(model)

    model.summary()

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            os.path.join(
                config.OUTPUT_DIR,
                "training_log.csv",
            )
        ),
    ]

    total_start = time.time()

    print("\n" + "=" * 60)
    print("PHASE 1 : Training Classification Head")
    print("=" * 60)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("PHASE 2 : Fine-Tuning MobileNetV3")
    print("=" * 60)

    model = model_lib.unfreeze_for_finetune(
        model,
        base,
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    model.save(config.MODEL_PATH)

    save_history(history1, history2)

    elapsed = (time.time() - total_start) / 60

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print(f"Final model : {config.MODEL_PATH}")
    print(f"Best model  : {config.BEST_MODEL_PATH}")
    print(f"History     : outputs/history.json")
    print(f"Training log: outputs/training_log.csv")
    print(f"Total time  : {elapsed:.2f} minutes")

    print("\nRun:")
    print("python evaluate.py")


if __name__ == "__main__":
    main()