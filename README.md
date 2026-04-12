# Brain Tumor Detection and Classification using MOAOA-FDL

Implementation of the paper:
> **"Multi-objective Archimedes Optimization Algorithm with Fusion-based Deep Learning
> model for brain tumor diagnosis and classification"**
> B. Devanathan & M. Kamarasan — Multimedia Tools and Applications (2023)

---

## Project Structure

```
optimisation/
├── data/
│   ├── augmented/          # Augmented MRI images (yes/no)
│   ├── segmented_multi/    # AOA + Shannon segmented images (PNG)
│   ├── tumor_only/         # Extracted tumor region masks
│   └── features/
│       ├── features.npy    # Fused feature matrix (1260, 1186)
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
│   └── models/
│       ├── entropy.py            # AOA + Shannon multi-threshold segmentation
│       ├── tumor_extraction.py   # Tumor region extraction
│       ├── feature_extraction.py # MobileNet + EfficientNet feature fusion
│       ├── classifier.py         # AOA + LSTM classification
│       └── results.py            # Plots and metrics
|   └── preprocessing/
|       ├── augmentation.py       # Rotating images
|       ├── clahe.py              #Contrast enhancement
|       ├── preprocess.py         # Resizing
|       ├── process_all.py        # Entire preprocessing can be done 
|       └── skull_scraping.py     # Removing the skull
|   └── segmentation/
|       ├── entropy.py            # Shannon Entropy
|       └── tumor_extraction.py   # Extracting Tumour for future visualisations
|
|   └── utils/
|       └──visualize.py           # visualisation
|    
├── venv/                   # Python virtual environment
└── README.md
```

---

## Pipeline Overview

```
Input MRI Images
      ↓
Image Preprocessing
(CLAHE Contrast Enhancement + Skull Stripping + Data Augmentation)
      ↓
Image Segmentation
(AOA + Shannon Entropy Multi-level Thresholding)
      ↓
Tumor Extraction
(Morphological Operations + Largest Contour)
      ↓
Feature Extraction
(MobileNet-V2 + EfficientNet-B0 → Fusion → Entropy Feature Selection)
      ↓
Classification
(AOA Hyperparameter Optimization + LSTM)
      ↓
Results & Evaluation
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

## Methods

### 1. Segmentation — `entropy.py`
- **AOA** (Archimedes Optimization Algorithm) selects optimal thresholds
- **Shannon Entropy** used as fitness function
- 3 thresholds → 4 intensity regions
- Output saved as **lossless PNG** to preserve exact pixel values

### 2. Tumor Extraction — `tumor_extraction.py`
- Takes highest intensity region from segmented image
- Morphological opening + closing for noise removal
- Keeps largest contour only (min area = 100px)

### 3. Feature Extraction — `feature_extraction.py`
- **MobileNet-V2** — ImageNet pretrained, GlobalAvgPool → 1280-d
- **EfficientNet-B0** — ImageNet pretrained, GlobalAvgPool → 1280-d
- Concatenated → 2560-d fused vector
- **Entropy-based selection** → top 1186 features

### 4. Classification — `classifier.py`
- **AOA** optimizes: learning rate, batch size, hidden size
- **LSTM** — 2 layers, dropout 0.5, softmax output
- Best params found: `lr=0.00594, batch=46, hidden=256`
- Final training: 100 epochs

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

### Comparison with Existing Methods (Kaggle Dataset)

| Method               | Sensitivity | Specificity | Accuracy   |
|----------------------|-------------|-------------|------------|
| **MOAOA-FDL (Ours)** | **97.28%**  | **94.37%**  | **95.50%** |
| SVM-KP               | 94.73%      | 97.59%      | 96.18%     |
| SVM-RBF              | 95.62%      | 83.71%      | 89.88%     |
| Decision Tree        | 97.88%      | 91.71%      | 94.95%     |
| CART                 | 88.00%      | 80.00%      | 84.00%     |
| Random Forest        | 96.00%      | 80.00%      | 88.00%     |
| k-NN                 | 80.00%      | 80.00%      | 80.00%     |
| Linear SVM           | 96.00%      | 80.00%      | 88.00%     |

---

## Setup & Installation

### 1. Clone / Download the project
```bash
cd C:\Users\chara\Desktop\optimisation
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install torch torchvision timm
pip install opencv-python numpy scikit-learn
pip install matplotlib seaborn
```

---

## How to Run

Run each script in order from the project root:

### Step 1 — Segmentation
```bash
python src/segmentation/entropy.py
```

### Step 2 — Tumor Extraction
```bash
python src/segmentation/tumor_extraction.py
```

### Step 3 — Feature Extraction
```bash
python src/models/feature_extraction.py
```

### Step 4 — Classification
```bash
python src/models/classifier.py
```

### Step 5 — Results & Plots
```bash
python src/models/results.py
```

---

## Dependencies

| Package        | Version |
|----------------|---------|
| Python         | 3.10+   |
| PyTorch        | 2.0+    |
| torchvision    | 0.15+   |
| timm           | 0.9+    |
| opencv-python  | 4.7+    |
| numpy          | 1.24+   |
| scikit-learn   | 1.2+    |
| matplotlib     | 3.7+    |
| seaborn        | 0.12+   |

---

## Reference

B. Devanathan, M. Kamarasan,
*"Multi-objective Archimedes Optimization Algorithm with Fusion-based Deep Learning
model for brain tumor diagnosis and classification"*,
Multimedia Tools and Applications, 82, 16985-17007 (2023).
https://doi.org/10.1007/s11042-022-14164-5
