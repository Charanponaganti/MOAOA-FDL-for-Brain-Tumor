import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import timm

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------
# Dense Multi-layer MobileNet-V2 Feature Extractor
# Paper: "MobileNet" — using MobileNet-V2 (citation [4])
# Extracts AvgPool from ALL 19 layers + MaxPool from deep
# stages to match the paper's fused dimension of ~7835.
# Output: 4716-d feature vector per image
# ---------------------------------------------------------
class MobileNetV2MultiLayer(nn.Module):
    """Extract features from ALL layers of MobileNet-V2.

    AvgPool from all 19 convolutional layers:
        Layer  0: 32    Layer  7: 64    Layer 14: 160
        Layer  1: 16    Layer  8: 64    Layer 15: 160
        Layer  2: 24    Layer  9: 64    Layer 16: 160
        Layer  3: 24    Layer 10: 64    Layer 17: 320
        Layer  4: 32    Layer 11: 96    Layer 18: 1280
        Layer  5: 32    Layer 12: 96
        Layer  6: 32    Layer 13: 96
        AvgPool subtotal: 2836

    MaxPool from deep stages [3, 13, 16, 17, 18]:
        24 + 96 + 160 + 320 + 1280 = 1880
        MaxPool subtotal: 1880

    Total: 4716
    """
    MAXPOOL_LAYERS = {3, 13, 16, 17, 18}

    def __init__(self):
        super().__init__()
        mob = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mob.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            outputs.append(self.avg_pool(x).flatten(1))
            if i in self.MAXPOOL_LAYERS:
                outputs.append(self.max_pool(x).flatten(1))
        return torch.cat(outputs, dim=1)  # (batch, 4716)


# ---------------------------------------------------------
# Dense Multi-layer EfficientNet-B0 Feature Extractor
# Paper: "EfficientNet-B0" as described in Section 3.4
# Extracts AvgPool from conv_stem, ALL 16 individual
# sub-blocks within the 7 stages, and conv_head.
# Output: 3120-d feature vector per image
# ---------------------------------------------------------
class EfficientNetB0MultiLayer(nn.Module):
    """Extract features from ALL sub-blocks of EfficientNet-B0.

    Instead of only extracting from 7 stage endpoints, this
    extracts from each individual MBConv block within stages:

        conv_stem:      32
        blocks[0][0]:   16
        blocks[1][0-1]: 24, 24
        blocks[2][0-1]: 40, 40
        blocks[3][0-2]: 80, 80, 80
        blocks[4][0-2]: 112, 112, 112
        blocks[5][0-3]: 192, 192, 192, 192
        blocks[6][0]:   320
        conv_head:      1280
        Total: 3120
    """
    def __init__(self):
        super().__init__()
        model = timm.create_model("efficientnet_b0", pretrained=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.blocks = model.blocks
        self.conv_head = model.conv_head
        self.bn2 = model.bn2
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        outputs = []
        # Stem: conv + BN/activation -> 32 channels
        x = self.bn1(self.conv_stem(x))
        outputs.append(self.pool(x).flatten(1))
        # ALL individual sub-blocks (16 total across 7 stages)
        for stage in self.blocks:
            for block in stage:
                x = block(x)
                outputs.append(self.pool(x).flatten(1))
        # Head: conv + BN/activation -> 1280 channels
        x = self.bn2(self.conv_head(x))
        outputs.append(self.pool(x).flatten(1))
        return torch.cat(outputs, dim=1)  # (batch, 3120)


# ---------------------------------------------------------
# Build feature extractors
# ---------------------------------------------------------
def build_models():
    mob = MobileNetV2MultiLayer().to(device).eval()
    eff = EfficientNetB0MultiLayer().to(device).eval()

    # Verify output dimensions
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        mob_out = mob(dummy).shape[1]
        eff_out = eff(dummy).shape[1]
    print(f"MobileNet-V2 multi-layer features  : {mob_out}-d")
    print(f"EfficientNet-B0 multi-layer features: {eff_out}-d")
    print(f"Fused feature dimension             : {mob_out + eff_out}-d")

    return mob, eff


# ---------------------------------------------------------
# Image transform (ImageNet normalization)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Grayscale -> RGB (3-channel for pretrained models)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply transform -> tensor (1, 3, 224, 224)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------
# Entropy-based feature selection
# Paper: "Entropy is exploited for choosing 1186 score-based
# features from the fused feature vector"
# ---------------------------------------------------------
def entropy_feature_selection(features, n_select=1186):
    """
    Select top n_select features based on Shannon entropy score.
    features: (n_samples, n_features)
    """
    n_bins = 50
    scores = []

    for i in range(features.shape[1]):
        col = features[:, i]
        hist, _ = np.histogram(col, bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        scores.append(entropy)

    scores = np.array(scores)
    top_indices = np.argsort(scores)[::-1][:n_select]
    top_indices = np.sort(top_indices)

    print(f"  Top entropy score : {scores[top_indices[0]]:.4f}")
    print(f"  Min entropy score : {scores[top_indices[-1]]:.4f}")

    return features[:, top_indices], top_indices


# ---------------------------------------------------------
# Extract features from dataset
# ---------------------------------------------------------
def extract_features(input_dir, output_dir, labels=("yes", "no"), n_select=1186):
    os.makedirs(output_dir, exist_ok=True)

    mob_model, eff_model = build_models()

    all_features = []
    all_labels   = []

    for label_idx, label in enumerate(labels):
        in_path = os.path.join(input_dir, label)

        files = [f for f in os.listdir(in_path)
                 if f.lower().endswith(".png")]

        print(f"\n[FEATURE] Processing '{label}' ({len(files)} images)")

        for i, fname in enumerate(files, 1):
            img_path = os.path.join(in_path, fname)
            tensor = preprocess_image(img_path)

            if tensor is None:
                print(f"  Skipping {fname}")
                continue

            with torch.no_grad():
                mob_feat = mob_model(tensor).cpu().numpy().flatten()  # 4716-d
                eff_feat = eff_model(tensor).cpu().numpy().flatten()  # 3120-d

            # Fuse by concatenation (paper Section 3.4)
            fused = np.concatenate([mob_feat, eff_feat])  # 7836-d (~paper's 7835)
            all_features.append(fused)
            all_labels.append(label_idx)

            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {fname} | fused shape: {fused.shape}")

    all_features = np.array(all_features)
    all_labels   = np.array(all_labels)

    print(f"\nFull fused feature matrix : {all_features.shape}")

    # Entropy-based feature selection (paper: 1186 from total)
    print(f"Selecting top {n_select} features via entropy...")
    selected_features, selected_indices = entropy_feature_selection(all_features, n_select)
    print(f"Selected feature matrix   : {selected_features.shape}")

    # Save
    np.save(os.path.join(output_dir, "features.npy"), selected_features)
    np.save(os.path.join(output_dir, "labels.npy"),   all_labels)
    np.save(os.path.join(output_dir, "indices.npy"),  selected_indices)

    print(f"\nSaved to '{output_dir}' [OK]")
    return selected_features, all_labels


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get the project root directory (2 levels up from src/models/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    extract_features(
        input_dir  = os.path.join(base_dir, "data/segmented_multi"),
        output_dir = os.path.join(base_dir, "data/features"),
        labels     = ("yes", "no"),
        n_select   = 1186
    )