# Duster: Deep Embedded Clustering for Image Matching

## Context: 3D Reconstruction & The "Doppelgangers" Problem
This project focuses on advancing 3D reconstruction techniques from unorganized and "messy" image collections. A key challenge in this domain is the **"Doppelgangers"** scenario, where real-world image collections contain visually similar but distinct locations or irrelevant outlier images.

To achieve high-quality 3D models, it is critical to:
1.  **Cluster** images into their respective scenes.
2.  **Identify and discard** irrelevant "outlier" images before proceeding with Structure from Motion (SfM).

## Project Overview: Duster
**Duster** implements an unsupervised clustering solution to address this problem. It leverages **Deep Embedded Clustering (DEC)** with robust **DINOv2** features to automatically group images by scene and identify outliers, enhancing the reliability of downstream 3D reconstruction pipelines.

### Methodology
1.  **Robust Feature Extraction**:
    -   We use the **DINOv2 (ViT-S/16)** model to extract semantic image embeddings. DINOv2 provides robust features that generalize well across different viewpoints and lighting conditions, suitable for distinguishing between similar-looking scenes.

2.  **Unsupervised Clustering**:
    -   The core approach uses **Deep Embedded Clustering (DEC)**.
    -   **Initialization**: Cluster centroids are initialized using **K-Means** on the extracted features.
    -   **Refinement**: The model iteratively refines both embeddings (optional) and centroids by minimizing the **KL divergence** between:
        -   $Q$: The soft assignment of samples to clusters (Student's t-distribution kernel).
        -   $P$: A target auxiliary distribution derived from $Q$ that encourages high-confidence assignments (sharpening predictions).

## Setup
Please use the provided conda environment configuration. You can create and activate the `imc2025` environment using:

```bash
conda env create -f environment.yml
conda activate imc2025
```

## Repository Structure
- `duster.py`: Defines the `Duster` model architecture (DEC wrapper) and the `Images` dataset loader for handling preprocessed image data.
- `training.py`: Contains the training logic, including K-Means initialization and the iterative KL divergence optimization loop.
