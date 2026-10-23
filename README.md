# ShopGuard-3D

### Shoplifting Video Classification Using a Custom 3D CNN

ShopGuard-3D is a deep learning project that classifies videos into two categories: **Shoplifting** and **Non-shoplifting**. It uses a custom 3D Convolutional Neural Network (3D CNN) trained from scratch to learn spatial and temporal features from video frames.

## Project Overview

The goal of this project is to explore video-based shoplifting classification using deep learning.

Unlike a conventional 2D CNN that processes individual images, a 3D CNN applies convolutions across spatial and temporal dimensions, allowing the model to learn information from multiple frames.

**This project performs video-level classification. It does not localize people, identify individuals, or pinpoint the exact moment of a suspected event.**

## Dataset

The project uses the Shoplifting Detection Dataset available through Kaggle.

- Total videos: 855
- Non-shoplifting videos: 531
- Shoplifting videos: 324

Dataset link: [Shoplifting Detection Dataset on Kaggle](https://www.kaggle.com/datasets/nebalelshobary/shoplifting-detection-dataset)

The dataset is split into training, validation, and test sets using group-aware splitting to reduce the risk of related videos appearing across different subsets.

| Split | Videos | Non-shoplifting | Shoplifting |
|---|---:|---:|---:|
| Training | 611 | 379 | 232 |
| Validation | 122 | 76 | 46 |
| Test | 122 | 76 | 46 |
| **Total** | **855** | **531** | **324** |

The grouping is based on video filenames. This does not guarantee that visually similar videos with unrelated filenames are separated.

## Data Preprocessing

- Sample 16 frames from each video.
- Resize frames to 160 × 160 pixels while preserving aspect ratio using padding.
- Convert video frames into tensors suitable for 3D CNN input.
- Apply video-level augmentation to training samples only.
- Use separate training, validation, and test datasets.

## Model Architecture

A custom 3D CNN is implemented and trained from scratch without pretrained weights.

The model includes:

- 3D convolutional layers with channel progression: 3 → 16 → 32 → 64 → 128.
- Batch Normalization and ReLU activations.
- 3D max pooling.
- Adaptive average pooling.
- Fully connected classification layers with dropout.
- Two output classes.

Total model parameters: **300,466**.

## Training Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Model | Custom 3D CNN |
| Maximum epochs | 25 |
| Batch size | 4 |
| Initial learning rate | 0.001 |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Early-stopping patience | 5 epochs |
| Input frames | 16 |
| Frame size | 160 × 160 |

The best checkpoint is selected using validation loss.

## Test Results

The best checkpoint was evaluated on the held-out test set of 122 videos.

| Metric | Score |
|---|---:|
| Accuracy | 87.70% |
| Shoplifting Precision | 80.39% |
| Shoplifting Recall | 89.13% |
| Shoplifting F1-score | 84.54% |

### Confusion Matrix

Rows represent actual labels; columns represent predicted labels.

| | Predicted: Non-shoplifting | Predicted: Shoplifting |
|---|---:|---:|
| Actual: Non-shoplifting | 66 | 10 |
| Actual: Shoplifting | 5 | 41 |

These results are an initial experimental evaluation on this dataset. They do not establish real-world reliability or guarantee performance on footage from different stores, cameras, or environments.

## Requirements

The project uses Python and the following main libraries:

- PyTorch
- NumPy
- Pandas
- OpenCV
- Pillow
- scikit-learn
- Matplotlib

See `requirements.txt` for the project dependencies.

## How to Run

The notebook was developed in Kaggle and uses a Kaggle dataset input path.

1. Open the notebook in Kaggle.
2. Add the dataset to the notebook using Kaggle's dataset interface.
3. Verify the dataset path in the notebook.
4. Run the notebook cells in order.
5. Train the model and evaluate the saved best checkpoint on the test set.

Running locally may require changes to the dataset path and installation of compatible Python and PyTorch versions.

## Limitations

- The model predicts one class for an entire video.
- It does not identify or track individuals.
- It does not provide temporal localization of suspected shoplifting.
- Performance on unseen environments has not been established.
- Filename-based grouping cannot guarantee the absence of visual duplicates across splits.

## Future Improvements

- Analyze training and validation curves to assess overfitting.
- Evaluate performance on additional independent datasets.
- Investigate class-specific errors and false alarms.
- Explore temporal localization of suspicious events.
- Develop a simple interface for video upload and inference.

## Author

**[Your Name]**

GitHub: [Your GitHub Profile URL]

## Disclaimer

This project is an educational machine learning experiment. Its predictions should not be treated as proof of theft or used as the sole basis for accusing or taking action against any person. Any practical deployment would require extensive independent evaluation, human review, and privacy safeguards.
