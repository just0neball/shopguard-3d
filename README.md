# ShopGuard-3D

### Shoplifting Video Classification Using a Custom 3D CNN

## 1. Project Overview

ShopGuard-3D is a deep learning project that classifies video clips into two categories:

- Non-shoplifting
- Shoplifting

The project uses a custom 3D Convolutional Neural Network (3D CNN) built and trained from scratch using PyTorch.

## 2. Dataset

**Dataset:** [Shoplifting Detection Dataset](https://www.kaggle.com/datasets/nebalelshobary/shoplifting-detection-dataset)

The dataset contains 855 videos:

- Non-shoplifting: 531 videos
- Shoplifting: 324 videos

## 3. Data Preprocessing

The following preprocessing steps were implemented:

- Sample 16 frames from each video.
- Resize frames to 160 × 160 pixels while preserving aspect ratio using padding.
- Convert video frames into tensors for model input.
- Apply video-level augmentation to training samples only.

## 4. Dataset Splitting

The dataset was divided into training, validation, and test sets using group-aware splitting.

| Split | Total Videos | Non-shoplifting | Shoplifting |
|---|---:|---:|---:|
| Training | 611 | 379 | 232 |
| Validation | 122 | 76 | 46 |
| Test | 122 | 76 | 46 |
| **Total** | **855** | **531** | **324** |

Grouping was based on video filenames to reduce the risk of related videos appearing in multiple splits.

## 5. Model Architecture

A custom 3D CNN was implemented without pretrained weights.

The architecture includes:

- 3D convolutional layers: 3 → 16 → 32 → 64 → 128 channels.
- Batch Normalization and ReLU activation.
- Max Pooling layers.
- Adaptive Average Pooling.
- Fully connected classification layers.
- Dropout for regularization.
- Two output classes.

**Total parameters:** 300,466

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Maximum epochs | 25 |
| Batch size | 4 |
| Initial learning rate | 0.001 |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Early-stopping patience | 5 epochs |
| Training device | CUDA GPU |

The best model checkpoint was selected based on validation loss.

**Best epoch:** 24  
**Best validation loss:** 0.3644

## 7. Test Evaluation

The saved best checkpoint was evaluated on the held-out test set containing 122 videos.

| Metric | Result |
|---|---:|
| Accuracy | 87.70% |
| Shoplifting Precision | 80.39% |
| Shoplifting Recall | 89.13% |
| Shoplifting F1-score | 84.54% |

### Confusion Matrix

Rows represent actual classes; columns represent predicted classes.

| | Predicted Non-shoplifting | Predicted Shoplifting |
|---|---:|---:|
| Actual Non-shoplifting | 66 | 10 |
| Actual Shoplifting | 5 | 41 |

## 8. Current Project Files

- `shoplifting-detection-3d-cnn.ipynb` — Notebook containing the implementation, training, and evaluation.
- `best_shoplifting_3dcnn.pth` — Saved model checkpoint.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Git ignore configuration.
- `README.md` — Project documentation.

## 9. Current Limitations

- The model classifies an entire video into one of two classes.
- It does not localize the exact moment of a suspected event.
- It does not identify or track individuals.
- The reported metrics reflect evaluation on this dataset and do not establish real-world performance.

## Disclaimer

This project is an educational deep learning experiment. Predictions should not be treated as proof of theft or used as the sole basis for accusing or taking action against any person.
