# Brain Tumor Detection and Classification using MOAOA-FDL

Implementation of the paper:
> **"Multi-objective Archimedes Optimization Algorithm with Fusion-based Deep Learning
> model for brain tumor diagnosis and classification"**
> B. Devanathan & M. Kamarasan — Multimedia Tools and Applications (2023)

---

## Project Structure

```
Opti-Project/
├── data/
│   ├── augmented/          # Augmented MRI images (yes/no)
│   ├── segmented_multi/    # AOA + Shannon segmented images (PNG)
│   ├── tumor_only/         # Extracted tumor region masks
│   └── features/
│       ├── features.npy    # Fused feature matrix
│       ├── labels.npy      # Class labels (0=tumor, 1=normal)
│       ├── indices.npy     # Selected feature indices
│       └── lstm_model.pth  # Trained LSTM model
│   └── results/
│       ├── accuracy_curve.png
│       ├── loss_curve.png
│       ├── confusion_matrix.png
│       ├── sensitivity_specificity.png
│       ├── accuracy_comparison.png
│       └── metrics.csv
├── src/
│   ├── classification/
│   │   └── models/
│   │       ├── classifier.py         # AOA + LSTM classification
│   │       └── lstm.py               # LSTM model architecture
│   ├── feature_extraction/
│   │   └── extractors/
│   │       └── feature_extraction.py # MobileNet + EfficientNet feature fusion
│   ├── performance/
│   │   └── evaluation/
│   │       └── results.py            # Plots and metrics
│   ├── preprocessing/
│   │   ├── pipeline/
│   │   │   └── process_all.py        # Pipeline entry for preprocessing
│   │   └── techniques/
│   │       ├── augmentation.py       # Rotating images
│   │       ├── clahe.py              # Contrast enhancement
│   │       ├── preprocess.py         # Resizing
│   │       └── skull_scraping.py     # Removing the skull
│   └── segmentation/
│       ├── algorithms/
│       │   └── entropy.py            # AOA + Shannon multi-threshold segmentation
│       └── extraction/
│           └── tumor_extraction.py   # Tumor region extraction
├── notebooks/
│   └── performance/
│       └── results.ipynb   # Visualization notebooks
├── main.py                 # Feature loading and verification script
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Pipeline Overview

```
Input MRI Images
      ↓
Image Preprocessing (src/preprocessing/pipeline/process_all.py)
(CLAHE + Skull Stripping + Augmentation)
      ↓
Image Segmentation (src/segmentation/algorithms/entropy.py)
(AOA + Shannon Entropy Multi-level Thresholding)
      ↓
Tumor Extraction (src/segmentation/extraction/tumor_extraction.py)
(Morphological Operations + Largest Contour)
      ↓
Feature Extraction (src/feature_extraction/extractors/feature_extraction.py)
(MobileNet-V2 + EfficientNet-B0 → Fusion → Entropy Selection)
      ↓
Classification (src/classification/models/classifier.py)
(AOA Hyperparameter Optimization + LSTM)
      ↓
Results & Evaluation (src/performance/evaluation/results.py)
```

---

## Dataset

- **Kaggle Brain MRI Dataset**
  - Source: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
  - Normal: 98 images → augmented to 490
  - Tumor:  155 images → augmented to 770
  - Total:  1260 images
- **Split:** 70% training / 30% testing

---

## Setup & Installation

### 1. Clone / Download the project
Navigate to the project root directory in your terminal.

### 2. Create and activate virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

Run each script in order from the project root:

### Step 1 — Preprocessing
```bash
python -m src.preprocessing.pipeline.process_all
```

### Step 2 — Segmentation
```bash
python -m src.segmentation.algorithms.entropy
```

### Step 3 — Tumor Extraction
```bash
python -m src.segmentation.extraction.tumor_extraction
```

### Step 4 — Feature Extraction
```bash
python -m src.feature_extraction.extractors.feature_extraction
```

### Step 5 — Classification
```bash
python -m src.classification.models.classifier
```

### Step 6 — Results & Plots
```bash
python -m src.performance.evaluation.results
```

---

## Results

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 95.50% |
| Sensitivity | 97.28% |
| Specificity | 94.37% |
| F-Score     | 95.52% |
| MCC         | 0.9075 |
| Kappa       | 0.9064 |

---

## Reference

B. Devanathan, M. Kamarasan,
*"Multi-objective Archimedes Optimization Algorithm with Fusion-based Deep Learning
model for brain tumor diagnosis and classification"*,
Multimedia Tools and Applications, 82, 16985-17007 (2023).
https://doi.org/10.1007/s11042-022-14164-5
