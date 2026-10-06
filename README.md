# Image Corruption Classification

This project investigates the ability of machine learning models to identify different types of image corruption. The project is part of an honors capstone focused on comparing traditional computer vision models with vision-language models (VLMs) for image corruption classification.

## Current Experiment

The current experiment performs multiclass image classification using images from the OPV2V dataset.

The six classes are:

- Clean
- Gaussian noise
- Shot noise
- Impulse noise
- Motion blur
- Defocus blur

A total of 4,800 images are used, with 800 images in each class.

The dataset is divided into:

- Training: 3,840 images
- Validation: 480 images
- Testing: 480 images

Each class contains:

- 640 training images
- 80 validation images
- 80 testing images

The images originate from complete four-camera frames. Each source frame is assigned to only one class and one dataset split. This prevents the same source frame from appearing in multiple classes or across training, validation, and testing.

## Corruption Generation

Image corruptions are generated using the `imagecorruptions` Python library.

Five corruption types are applied:

- Gaussian noise
- Shot noise
- Impulse noise
- Motion blur
- Defocus blur

Each corruption uses severity levels from 1 through 5.

For every corruption class, severity levels are evenly distributed:

- Training: 128 images per severity
- Validation: 16 images per severity
- Testing: 16 images per severity

A fixed random seed of 42 is used during dataset preparation and corruption assignment for reproducibility.

Information about each corrupted image, including its split, corruption type, and severity, is stored in `corruption_metadata.csv`.

## Machine Learning Models

Three convolutional neural network models are currently evaluated:

- EfficientNet-B0
- ResNet-18
- DenseNet-121

Each model uses pretrained ImageNet weights provided through TorchVision. The original classification layer is replaced with a new classification layer containing six outputs corresponding to the six image classes.

The models are fine-tuned using the same dataset and general training configuration.

### Training Configuration

- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Cross-Entropy Loss
- Batch size: 32
- Epochs: 5
- Training images: 3,840
- Validation images: 480
- Testing images: 480
- Device: Apple MPS when available, otherwise CUDA or CPU

The model checkpoint with the highest validation accuracy is saved for final testing.

## Results

All three models are evaluated on the same held-out test set containing 480 images.

| Model | Best Validation Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 98.54% | 99.17% | 99.19% | 99.17% | 99.17% |
| EfficientNet-B0 | 98.33% | 98.33% | 98.36% | 98.33% | 98.33% |
| ResNet-18 | 96.46% | 96.88% | 96.94% | 96.88% | 96.89% |

DenseNet-121 achieved the highest overall test accuracy, correctly classifying 476 of the 480 test images.

Detailed results and confusion matrices for each model are stored in the `results/` directory.

## Severity Analysis

Performance is also evaluated according to corruption severity.

DenseNet-121 achieved the following overall accuracy across the five corruption types at each severity level:

| Severity | Accuracy |
|---|---:|
| 1 | 97.50% |
| 2 | 98.75% |
| 3 | 100.00% |
| 4 | 100.00% |
| 5 | 100.00% |

These results show that lower-severity corruptions can be more difficult to distinguish, while DenseNet-121 correctly classified all corrupted test images at severity levels 3 through 5.

## Project Structure

```text
corruption-classifier/
├── scripts/
│   ├── prepare_dataset.py
│   ├── add_corruptions.py
│   ├── train_efficientnet.py
│   ├── evaluate_efficientnet.py
│   ├── train_resnet18.py
│   ├── evaluate_resnet18.py
│   ├── train_densenet121.py
│   └── evaluate_densenet121.py
├── results/
│   ├── efficientnet_results.txt
│   ├── efficientnet_confusion_matrix.png
│   ├── resnet18_results.txt
│   ├── resnet18_confusion_matrix.png
│   ├── densenet121_results.txt
│   └── densenet121_confusion_matrix.png
├── .gitignore
├── README.md
└── requirements.txt
```

The dataset and trained model files are excluded from GitHub because of their size.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Preparing the Dataset

Run the dataset preparation script:

```bash
python scripts/prepare_dataset.py
```

Then generate the corrupted images:

```bash
python scripts/add_corruptions.py
```

These scripts create the balanced multiclass dataset and generate the five corruption types.

## Training the Models

Train each model individually:

```bash
python scripts/train_efficientnet.py
python scripts/train_resnet18.py
python scripts/train_densenet121.py
```

Each script fine-tunes a pretrained model and saves the checkpoint with the highest validation accuracy.

## Evaluating the Models

Evaluate each trained model:

```bash
python scripts/evaluate_efficientnet.py
python scripts/evaluate_resnet18.py
python scripts/evaluate_densenet121.py
```

The evaluation scripts calculate overall accuracy, precision, recall, F1 score, per-class performance, confusion matrices, and performance by corruption severity.

Evaluation results and confusion matrix images are saved in the `results/` directory.

## Current Status

The current stage establishes machine learning baselines for image corruption classification. Future stages of the capstone will evaluate vision-language models on image corruption tasks and compare their performance with the traditional machine learning models.