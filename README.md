# Brain Tumor Detection using Deep Learning (VGG16 + Transfer Learning)

This project aims to classify MRI scans into four brain tumor categories using **deep learning**, specifically leveraging **transfer learning** with the **VGG16** architecture. Early and accurate classification of brain tumors can assist doctors and radiologists in making informed medical decisions.

## Project Overview

- **Goal:** Classify MRI images into 4 tumor types
- **Model Used:** Pre-trained **VGG16** (without top layers)
- **Approach:** Transfer Learning using Keras + TensorFlow
- **Dataset:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Dataset Details

The dataset contains **1,311 MRI images** organized into 4 classes:

| Class        | Description                   | Image Count |
|--------------|-------------------------------|-------------|
| `glioma`     | Malignant tumors from glial cells | 300         |
| `meningioma` | Tumors in the meninges (brain lining) | 306         |
| `notumor`    | Normal brain MRIs without any tumor | 405         |
| `pituitary`  | Tumors in the pituitary gland | 300         |

## Results
- **Training Accuracy:** 98.36%
- **Validation Accuracy:** 96%
