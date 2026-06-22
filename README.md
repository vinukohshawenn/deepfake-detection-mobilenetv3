# Deepfake Image Detector using Transfer Learning

A lightweight deep learning pipeline for **binary deepfake image classification** using **TensorFlow** and **MobileNetV3-Small**. The model leverages transfer learning from ImageNet, followed by fine-tuning on a labeled deepfake dataset to classify images as **Real** or **Fake**.

The project is designed to be modular, reproducible, and CPU-friendly while remaining easy to extend with stronger backbones such as EfficientNet or Vision Transformers.

---

## Features

* Transfer learning using **MobileNetV3-Small (ImageNet pretrained)**
* Two-stage training (feature extraction + fine-tuning)
* Automatic train/validation split
* Dedicated held-out test set evaluation
* Data augmentation for improved generalization
* Early stopping and best-model checkpointing
* ROC-AUC, Precision, Recall, F1-score and Confusion Matrix
* Batch inference for single images or folders
* Clean and modular TensorFlow pipeline

---

## Project Structure

```
deepfake_detector/

├── config.py
├── data/
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   │
│   └── Test/
│       ├── Real/
│       └── Fake/
│
├── data_pipeline.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── outputs/
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input Image (224 × 224 × 3)
            │
            ▼
MobileNetV3Small
(ImageNet Pretrained)
            │
Global Average Pooling
            │
Batch Normalization
            │
Dense (256)
            │
Dropout
            │
Dense (64)
            │
Dropout
            │
Sigmoid
            │
Real / Fake
```

The model is trained using **transfer learning**:

**Phase 1**

* Freeze the MobileNetV3 backbone
* Train only the custom classification head

**Phase 2**

* Unfreeze the top MobileNetV3 layers
* Fine-tune using a lower learning rate

---

## Installation

Clone the repository and create a virtual environment.

```bash
git clone <repository-url>

cd deepfake_detector

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

The project expects the following directory structure:

```
data/

├── Train/
│   ├── Real/
│   └── Fake/
│
└── Test/
    ├── Real/
    └── Fake/
```

The validation dataset is automatically created from **10% of the training set** during training.

Using the dataset: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

---

## Training

Start training with

```bash
python train.py
```

Training consists of two stages:

1. Feature extraction using a frozen MobileNetV3 backbone.
2. Fine-tuning of the top backbone layers using a lower learning rate.

During training the following are automatically generated:

* Best model checkpoint
* Final trained model
* Training history
* CSV training log

All files are saved inside the `outputs/` directory.

---

## Evaluation

Evaluate the trained model on the held-out test dataset.

```bash
python evaluate.py
```

The evaluation reports

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

and saves

```
outputs/

confusion_matrix.png
roc_curve.png
```

---

## Prediction

Predict a single image

```bash
python predict.py image.jpg
```

Predict every image inside a folder

```bash
python predict.py data/Test/Fake
```

Example output

```
fake_102.jpg

Prediction : FAKE

Confidence : 98.4%

P(fake) : 0.984
```

---

## Configuration

All configurable parameters are stored in `config.py`.

Examples include

* Batch size
* Image resolution
* Learning rates
* Number of epochs
* Number of MobileNetV3 layers to fine-tune
* Validation split
* Random seed

---

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV3-Small
* NumPy
* Scikit-learn
* Matplotlib

---

## Performance

The current implementation was trained using transfer learning with MobileNetV3-Small.

Example evaluation on the held-out test dataset:

| Metric    |                Score |
| --------- | -------------------: |
| Accuracy  |            **68.9%** |
| Precision | **71.4% (weighted)** |
| Recall    |            **68.9%** |
| ROC-AUC   |            **0.772** |

Performance can be further improved through:

* Larger training datasets
* Stronger backbone networks (EfficientNetV2, ConvNeXt, ViT)
* Hyperparameter tuning
* Class weighting
* Threshold optimization

---

## Limitations

Like most CNN-based deepfake detectors, this model learns statistical patterns present in the training data. Performance may degrade on:

* Previously unseen deepfake generation methods
* Heavy image compression
* Adversarial perturbations
* Domain shifts between datasets

The classifier should therefore be treated as an assistive decision-support tool rather than definitive forensic evidence.

---

## Future Improvements

* EfficientNetV2 backbone
* Vision Transformer (ViT)
* Explainability using Grad-CAM
* Streamlit web interface
* ONNX / TensorFlow Lite export
* Docker deployment
* REST API using FastAPI
* Model comparison dashboard

---

## License

This project is intended for educational and research purposes.
