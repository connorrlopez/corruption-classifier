# Image Corruption Classification

This project investigates the ability of machine learning models to detect corruption in images. The project is part of an honors capstone focused on comparing traditional computer vision models with vision-language models (VLMs) for image corruption detection.

## Current Experiment

The current experiment uses EfficientNet-B0 to perform binary image classification between:

- Clean images
- Images corrupted with Gaussian noise

The images are taken from the OPV2V dataset. A total of 1,000 clean images and 1,000 Gaussian-corrupted images are used.

The dataset is divided into:

- Training: 800 clean + 800 Gaussian
- Validation: 100 clean + 100 Gaussian
- Testing: 100 clean + 100 Gaussian

Source frames are kept within a single dataset split to prevent data leakage between training, validation, and testing.

## Gaussian Noise Generation

Gaussian noise is added to the clean images using the `imagecorruptions` Python library.

Each clean image is copied and corrupted using Gaussian noise with a randomly selected severity level from 1 to 5. A fixed random seed of 42 is used to make the corruption process reproducible.

The severity assigned to each corrupted image is recorded in `gaussian_severities.csv`.

This produces 1,000 Gaussian-corrupted images corresponding to the 1,000 original clean images.

## Model Training

EfficientNet-B0 is used as the traditional computer vision model for the initial classification experiment.

The model uses pretrained ImageNet weights through TorchVision. The original EfficientNet-B0 classification layer, which predicts 1,000 ImageNet classes, is replaced with a new classification layer containing two outputs:

- Clean
- Gaussian

The entire model is fine-tuned on the training dataset.

### Training Configuration

- Model: EfficientNet-B0
- Pretrained weights: ImageNet
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Cross-Entropy Loss
- Batch size: 32
- Epochs: 5
- Device: Apple MPS when available, otherwise CPU

After each epoch, the model is evaluated on the validation set. The model with the lowest validation loss is saved as `best_model.pth`.

## Results

The best trained model was evaluated on the held-out test set containing 200 images: 100 clean images and 100 Gaussian-corrupted images.

The model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |

### Confusion Matrix

| | Predicted Clean | Predicted Gaussian |
|---|---:|---:|
| Actual Clean | 100 | 0 |
| Actual Gaussian | 0 | 100 |

EfficientNet-B0 correctly classified all 200 test images, resulting in zero false positives and zero false negatives.

These results apply specifically to the current binary classification experiment using clean and Gaussian-corrupted OPV2V images. Additional corruption types and experiments will be evaluated in future stages of the project.

## Project Structure

```text
corruption-classifier/
├── scripts/
│   ├── prepare_dataset.py
│   ├── add_gaussian_noise.py
│   ├── train_classifier.py
│   └── evaluate_classifier.py
├── .gitignore
├── README.md
└── requirements.txt
'''

## Setup

Create and activate a Python virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Experiment

The scripts should be run from the root of the project in the following order:

```bash
python scripts/prepare_dataset.py
python scripts/add_gaussian_noise.py
python scripts/train_classifier.py
python scripts/evaluate_classifier.py
```

The first two scripts prepare the dataset and generate the corrupted images. The training script fine-tunes EfficientNet-B0 and saves the best model. The evaluation script loads the saved model and reports its performance on the held-out test set.