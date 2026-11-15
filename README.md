# Image Corruption Classification

This project investigates the ability of convolutional neural networks (CNNs) and vision-language models (VLMs) to identify different types of image corruption. The research is part of an honors capstone focused on comparing traditional machine learning approaches with multimodal language models for image corruption classification.

## Project Overview

The experiments use camera images from the OPV2V dataset and evaluate six image classes:

- Clean
- Gaussian noise
- Shot noise
- Impulse noise
- Motion blur
- Defocus blur

Three CNN architectures are evaluated: EfficientNet-B0, ResNet-18, and DenseNet-121. These models use pretrained ImageNet weights and are fine-tuned on labeled image corruption datasets.

The VLM experiments investigate whether multimodal models can classify image corruptions without task-specific fine-tuning. The selected models include Gemma 3 4B, Llama 3.2 Vision 11B, and Pixtral 12B. Few-shot evaluations provide labeled reference images to help the models distinguish between corruption types.

The project includes initial experiments using smaller datasets, expanded experiments using the full dataset, and additional evaluations at fixed corruption severity levels.

Performance is measured using classification accuracy, precision, recall, and F1 score.

## Dataset Preparation

The experiments use camera images from the OPV2V dataset. Five types of image corruption are generated using the `imagecorruptions` Python library. Unmodified images form the clean class.

### Initial Experiment

The initial experiment uses 4,800 images, with 800 images per class.

| Split | Images |
|---|---:|
| Training | 3,840 |
| Validation | 480 |
| Testing | 480 |

Each class contains 640 training images, 80 validation images, and 80 testing images.

Corruption severities range from 1 through 5, with severity levels evenly distributed within each corrupted class.

The dataset preparation process assigns complete source frames to individual classes and dataset splits to prevent overlap between training, validation, and testing.

A fixed random seed of 42 is used during dataset preparation and corruption assignment.

### Expanded Dataset

The expanded experiments use larger datasets generated from the available OPV2V images.

For the fixed-severity experiments, each dataset contains:

| Split | Images |
|---|---:|
| Training | 82,992 |
| Validation | 41,912 |
| Testing | 23,944 |

Separate datasets are generated for severity levels 1, 3, and 5. Each dataset includes the same six classification categories, with corrupted images generated at the selected severity level.

Dataset manifests are stored as CSV files containing image paths, source information, class labels, severity levels, and dataset split assignments.

The expanded experiments use separate training, validation, and testing splits to evaluate model performance on held-out images.

## CNN Experiments

Three convolutional neural network architectures are used:

- EfficientNet-B0
- ResNet-18
- DenseNet-121

Each model is initialized using pretrained ImageNet weights provided through TorchVision. The original classification layer is replaced with a six-output classification layer corresponding to the six image classes.

The models are fine-tuned using supervised learning.

### Initial Training Configuration

The initial experiments use the following configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross-Entropy Loss |
| Batch size | 32 |
| Epochs | 5 |
| Training images | 3,840 |
| Validation images | 480 |
| Testing images | 480 |

The checkpoint with the highest validation accuracy is saved for final evaluation.

The expanded experiments use separate training scripts and configurations appropriate for the larger datasets.

### Initial CNN Results

All three models were evaluated on the same held-out test set containing 480 images.

| Model | Validation Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 98.54% | 99.17% | 99.19% | 99.17% | 99.17% |
| EfficientNet-B0 | 98.33% | 98.33% | 98.36% | 98.33% | 98.33% |
| ResNet-18 | 96.46% | 96.88% | 96.94% | 96.88% | 96.89% |

DenseNet-121 achieved the highest test accuracy in the initial experiment, correctly classifying 476 of the 480 test images.

### Initial Severity Analysis

DenseNet-121 was also evaluated according to corruption severity.

| Severity | Accuracy |
|---|---:|
| 1 | 97.50% |
| 2 | 98.75% |
| 3 | 100.00% |
| 4 | 100.00% |
| 5 | 100.00% |

These results suggest that lower-severity corruptions can be more difficult to distinguish, while stronger corruptions may be easier for the trained model to identify.

## Expanded CNN Severity Experiments

Additional experiments evaluate each CNN architecture using datasets generated at fixed corruption severity levels.

Each model is trained and evaluated separately at severity levels 1, 3, and 5, producing nine CNN experiments in total.

Each evaluation uses 23,944 test images.

### Test Accuracy

| Model | Severity 1 | Severity 3 | Severity 5 |
|---|---:|---:|---:|
| EfficientNet-B0 | 100.00% | 99.99% | 100.00% |
| ResNet-18 | 99.97% | 99.99% | 99.97% |
| DenseNet-121 | 99.99% | 100.00% | 100.00% |

*Results are rounded to two decimal places. A displayed value of 100.00% does not necessarily indicate that every image was classified correctly.*

The CNN models achieved consistently high accuracy across all three severity levels.

These results are being reviewed to better understand the effects of dataset construction, model pretraining, and the classification task. Additional validation may be necessary to determine how well the models generalize beyond the generated datasets.

## Vision-Language Model Experiments

The project also evaluates multimodal language models on the same six-class image corruption classification task.

The selected models are:

- Gemma 3 4B
- Llama 3.2 Vision 11B
- Pixtral 12B

Unlike the CNN architectures, the VLMs are evaluated without task-specific fine-tuning.

### Few-Shot Evaluation

The few-shot evaluation provides the model with six labeled reference images, one representing each image class.

The model is then given a test image and asked to identify its corruption category.

The available categories are:

- `clean`
- `gaussian`
- `shot`
- `impulse`
- `motion_blur`
- `defocus_blur`

The evaluation uses reference images from the training split and test images from the held-out testing split.

Predictions are compared against the ground-truth labels to calculate accuracy, precision, recall, and F1 score.

This methodology differs from CNN fine-tuning because the VLMs are not trained on the classification dataset. Instead, they use their existing multimodal capabilities and the reference examples provided in the prompt.

### Gemma 3 4B Results

Gemma 3 4B has completed few-shot evaluations at severity levels 1, 3, and 5.

Each experiment evaluates 23,944 test images.

| Severity | Test Accuracy | Macro F1 |
|---|---:|---:|
| 1 | 38.64% | 33.31% |
| 3 | 44.45% | 37.56% |
| 5 | 46.57% | 37.20% |

The model achieved higher overall accuracy as corruption severity increased, although macro F1 did not improve consistently.

These results are substantially lower than the supervised CNN results. However, the two approaches use different evaluation conditions because the CNNs are fine-tuned on labeled training data, while Gemma is evaluated using few-shot prompting without task-specific training.

### Llama and Pixtral

Llama 3.2 Vision 11B and Pixtral 12B are included in the VLM evaluation pipeline.

Preliminary testing has been conducted with Llama to investigate prompt formatting and classification behavior. Full comparative results are not yet reported.

Pixtral evaluation scripts are included in the repository, but final results are not yet reported.

## Project Structure

```text
corruption-classifier/
├── scripts/
│   ├── cnn/
│   │   ├── train_efficientnet.py
│   │   ├── train_resnet18.py
│   │   ├── train_densenet121.py
│   │   ├── train_efficientnet_full.py
│   │   ├── train_resnet18_full.py
│   │   ├── train_densenet121_full.py
│   │   ├── train_efficientnet_severity.py
│   │   ├── train_resnet18_severity.py
│   │   ├── train_densenet121_severity.py
│   │   ├── evaluate_severity.py
│   │   └── [other CNN evaluation scripts]
│   ├── data/
│   │   ├── prepare_dataset.py
│   │   ├── add_corruptions.py
│   │   ├── prepare_full_dataset.py
│   │   ├── full_dataset_loader.py
│   │   ├── prepare_severity_datasets.py
│   │   └── severity_dataset_loader.py
│   └── vlm/
│       ├── evaluate_gemma.py
│       ├── evaluate_gemma_fewshot.py
│       ├── evaluate_gemma_fewshot_full.py
│       ├── evaluate_llama.py
│       ├── evaluate_llama_fewshot_full.py
│       ├── evaluate_pixtral.py
│       ├── prepare_vlm_dataset.py
│       └── [additional VLM test scripts]
├── tests/
│   ├── test_alternative_corruptions.py
│   └── test_impulse_density.py
├── .gitignore
├── README.md
└── requirements.txt
```

Datasets, model checkpoints, and large generated experiment outputs are excluded from GitHub because of their size.

## Environment and Setup

The project uses Python, PyTorch, TorchVision, and additional libraries for image corruption generation and evaluation.

CNN experiments have been conducted locally using Apple Silicon and on a Linux workstation with NVIDIA GPUs.

The larger experiments use an Ubuntu workstation with NVIDIA RTX A6000 GPUs.

### Create a Python Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The VLM experiments use a separate environment with additional Hugging Face Transformers dependencies and model-specific requirements.

Access to some pretrained VLM checkpoints may require accepting the model's license terms through Hugging Face.

## Running the Experiments

The following commands illustrate the main experiment entry points. Dataset paths, checkpoint locations, and script arguments should be checked against the relevant source files before execution.

### Initial Dataset Preparation

```bash
python scripts/data/prepare_dataset.py
python scripts/data/add_corruptions.py
```

### Initial CNN Training

```bash
python scripts/cnn/train_efficientnet.py
python scripts/cnn/train_resnet18.py
python scripts/cnn/train_densenet121.py
```

### Initial CNN Evaluation

```bash
python scripts/cnn/evaluate_efficientnet.py
python scripts/cnn/evaluate_resnet18.py
python scripts/cnn/evaluate_densenet121.py
```

### Fixed-Severity CNN Experiments

The fixed-severity scripts use shared dataset loaders located in `scripts/data/`.

To support package-style imports, these scripts should be launched from the repository root using Python's module execution option.

For example:

```bash
python -m scripts.cnn.train_efficientnet_severity --help
python -m scripts.cnn.train_resnet18_severity --help
python -m scripts.cnn.train_densenet121_severity --help
python -m scripts.cnn.evaluate_severity --help
```

The supported command-line arguments and dataset locations are defined in the individual scripts.

### VLM Evaluation

VLM evaluation scripts are located in `scripts/vlm/`.

The full few-shot evaluation entry points include:

```text
scripts/vlm/evaluate_gemma_fewshot_full.py
scripts/vlm/evaluate_llama_fewshot_full.py
```

These scripts use pretrained vision-language models, labeled reference images, and held-out test images to perform corruption classification.

Model access, GPU memory, dependencies, and command-line arguments must be configured for the selected model.

## Reproducibility and Limitations

The experiments use fixed random seeds and dataset manifests to support reproducibility.

The fixed-severity experiments are designed to maintain separate training, validation, and testing splits, with source-frame separation intended to reduce overlap.

The high CNN accuracy requires careful interpretation. Pretrained CNNs may benefit from transferable visual features, and synthetic corruptions may contain patterns that are easier to distinguish than real-world image degradation.

The VLM comparison also has an important limitation: the CNNs receive supervised task-specific training, while the VLMs are evaluated through prompting without fine-tuning. Their results therefore represent different learning and evaluation conditions.

Additional experiments may be needed to evaluate generalization to alternative corruption implementations, unseen data sources, and different prompting strategies.

## Current Status

The following work has been completed:

- Initial CNN experiments on a balanced 4,800-image dataset.
- Expanded CNN evaluations using the larger OPV2V dataset.
- Nine fixed-severity CNN experiments across three architectures and three severity levels.
- Gemma 3 4B few-shot evaluations at severity levels 1, 3, and 5.
- Implementation and preliminary testing of additional VLM evaluation scripts.

Ongoing work includes further evaluation of Llama and Pixtral, investigation of the high CNN accuracy, and comparison of CNN and VLM classification behavior.

The final goal is to evaluate the strengths and limitations of supervised CNNs and pretrained VLMs for identifying synthetic image corruption.