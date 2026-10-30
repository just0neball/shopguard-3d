# ShopGuard: Shoplifting Detection with Deep Learning

**End-to-end binary video classification** · **Week 5: 3D CNN from scratch** · **Week 6: pretrained VideoMAE fine-tuning**

This repository documents a two-stage research project that classifies shopping video clips as **Normal** or **Shoplifting**. It starts with a lightweight 3D CNN trained from scratch and then evaluates a pretrained [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) fine-tuned on the same task.

> **Responsible use:** Model predictions are imperfect screening signals, **not evidence that someone has committed theft**. False negatives, false positives, privacy, and human review matter in real-world deployments.

## Project at a glance

| Stage | Model | What it covers | Notebook |
|---|---|---|---|
| **Week 5** | Custom 3D CNN | Frame sampling, clip-level augmentation, dataset class, training from scratch, metrics | [Week 5 notebook](shoplifting-detection-3d-cnn.ipynb) |
| **Week 6** | VideoMAE (transfer learning) | Pretrained model, binary classifier, fine-tuning, held-out evaluation, inference benchmarking | [Week 6 notebook](week6/shoplifting_videomae.ipynb) |

## Results: matched Week 6 test videos

Both saved model checkpoints were evaluated on the **same 122 Week 6 test clips** (76 normal, 46 shoplifting). These are observations from **one recorded Kaggle run**, not statistical-significance claims.

| Metric | Week 5 · 3D CNN | Week 6 · VideoMAE |
|:--|--:|--:|
| Test accuracy | 87.70% | **96.72%** |
| Shoplifting precision | 80.39% | **100.00%** |
| Shoplifting recall | 89.13% | **91.30%** |
| Shoplifting F1-score | 84.54% | **95.45%** |
| Model parameters | 300,466 | 86,237,954 |
| Checkpoint size | 3.46 MB | 329.05 MB |
| FP32 latency, batch 1 | **3.539 ms** | 121.453 ms |
| AMP/FP16 latency, batch 1 | **1.826 ms** | 32.771 ms |

**F1 difference:** +10.92 percentage points for the fine-tuned VideoMAE in this evaluation. However, VideoMAE requires considerably more compute and storage. The VideoMAE confusion matrix is `[[76, 0], [4, 42]]` (rows = true classes; columns = predicted classes, ordered Normal/Shoplifting).

**Benchmark conditions:** Tesla T4 GPU, single GPU, **batch size 1**, synthetic model inputs, warmed-up model-only forward-pass timings. FP32 and AMP/FP16 are **separate** experiments. Video decode/preprocessing is excluded. VideoMAE consumes 16 frames at 224×224; the CNN uses 16 frames at 160×160. A separate batched dual-GPU result in the notebook is not directly comparable to these batch-1 timings.

### Training and test visualizations

![VideoMAE train and validation curves](week6/assets/training_curves.png)

![VideoMAE confusion matrix](week6/assets/confusion_matrix.png)

## Repository structure

```text
shopguard-3d/
├── README.md                         # Overall project and results
├── shoplifting-detection-3d-cnn.ipynb # Week 5: from-scratch 3D CNN
├── best_shoplifting_3dcnn.pth        # Existing Week 5 weights
├── requirements.txt                  # Existing Week 5 dependencies
├── .gitignore                        # Existing root ignore rules
└── week6/
    ├── README.md                     # VideoMAE implementation details
    ├── shoplifting_videomae.ipynb    # Week 6 completed notebook + outputs
    ├── requirements.txt              # Week 6 dependencies
    ├── .gitignore                    # Ignore large weights/video files
    ├── assets/
    │   ├── training_curves.png
    │   └── confusion_matrix.png
    └── results/
        └── same_test_set_comparison.csv
```

## Reproduce the experiments

1. Open the Week 5 or [Week 6 notebook](week6/shoplifting_videomae.ipynb) in **Kaggle Notebooks**.
2. Attach the authorized labeled shoplifting video dataset and Week 5 checkpoint. The Week 6 notebook expects the dataset under `/kaggle/input/datasets/nebalelshobary/shoplifting-detection-dataset` (change `DATASET_PATH` if your mount differs); it searches `/kaggle/input/` for `best_shoplifting_3dcnn*.pth`.
3. For Week 6, select GPU acceleration (recorded experiment used **2 × Tesla T4**) and enable internet for the first Hugging Face download. The runtime used `transformers==5.16.1` and `torch==2.11.0+cu128`; requirements are in [`week6/requirements.txt`](week6/requirements.txt).
4. Run the notebook top to bottom. Week 6 performs **10 training epochs**, selects its checkpoint by validation F1, tests both models on the same Week 6 test clips, and exports plots/CSV files under `/kaggle/working/`.

The Week 6 notebook includes a compatibility mapping for VideoMAE attention-bias keys (`q_bias`/`v_bias`) when using the recorded Transformers version. Its binary classification head is newly initialized before fine-tuning.

## Evaluation limitations

- **Important:** We could re-evaluate the Week 5 checkpoint on the exact Week 6 test clips, but **the original Week 5 training manifest was not saved**. Some of those clips might have been present during Week 5 training; the CNN's independence from this test set is **unverified**. A rigorous paired study should reconstruct or independently audit the original Week 5 training membership.
- The 855-clip dataset uses group-aware Week 6 splitting (611 train, 122 validation, 122 test). This guards against overlap in the constructed `Group_ID`s; it does not prove the absence of all near-duplicates.
- Training augmentation is not completely deterministic. A new run may differ even with the same seed. The **saved notebook outputs** are from a completed run, and the test estimates are based on only 122 clips.
- VideoMAE missed 4 of the 46 positive examples. A security-oriented deployment would require further validation and a human-in-the-loop review process.

For training details, error analysis, checkpoint storage, and hyperparameters, see [Week 6 documentation](week6/README.md).

## Weights and dataset

The existing 3D CNN checkpoint is part of Week 5. The fine-tuned VideoMAE checkpoint is about **329 MB**, above GitHub's **100 MB ordinary-file limit**, so it is intentionally **not committed**. Store large weights outside Git or with a configured large-file mechanism, subject to permission and upstream licenses. Raw video data is not distributed here.

## Technology

Python · PyTorch · Hugging Face Transformers · OpenCV · scikit-learn · NumPy/Pandas · Kaggle GPUs

## Source model

[MCG-NJU/videomae-base-finetuned-kinetics on Hugging Face](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics). This repository reports task-specific experiments and does not claim ownership of the upstream model, dataset, or videos.
