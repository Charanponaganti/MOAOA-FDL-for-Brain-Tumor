import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import timm

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------
# Build MobileNet + EfficientNet feature extractors
# ---------------------------------------------------------
def build_models():
    # MobileNet V2 - remove classifier head
    mob = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    mob.classifier = torch.nn.Identity()  # remove final FC
    mob = mob.to(device).eval()

    # EfficientNet-B0 via timm
    eff = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # num_classes=0 = no head
    eff = eff.to(device).eval()

    print(f"MobileNet output features  : {mob(torch.zeros(1,3,224,224).to(device)).shape}")
    print(f"EfficientNet output features: {eff(torch.zeros(1,3,224,224).to(device)).shape}")

    return mob, eff


# ---------------------------------------------------------
# Image transform
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

    # Grayscale -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply transform -> tensor (1, 3, 224, 224)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------
# Entropy-based feature selection
# ---------------------------------------------------------
def entropy_feature_selection(features, n_select=1186):
    """
    Select top n_select features based on Shannon entropy score
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
                mob_feat = mob_model(tensor).cpu().numpy().flatten()  # 1280-d
                eff_feat = eff_model(tensor).cpu().numpy().flatten()  # 1280-d

            # Fuse by concatenation
            fused = np.concatenate([mob_feat, eff_feat])
            all_features.append(fused)
            all_labels.append(label_idx)

            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {fname} | fused shape: {fused.shape}")

    all_features = np.array(all_features)
    all_labels   = np.array(all_labels)

    print(f"\nFull fused feature matrix : {all_features.shape}")

    # Entropy-based feature selection
    print(f"Selecting top {n_select} features via entropy...")
    selected_features, selected_indices = entropy_feature_selection(all_features, n_select)
    print(f"Selected feature matrix   : {selected_features.shape}")

    # Save
    np.save(os.path.join(output_dir, "features.npy"), selected_features)
    np.save(os.path.join(output_dir, "labels.npy"),   all_labels)
    np.save(os.path.join(output_dir, "indices.npy"),  selected_indices)

    print(f"\nSaved to '{output_dir}' ✅")
    return selected_features, all_labels


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    extract_features(
        input_dir  = "data/segmented_multi",
        output_dir = "data/features",
        labels     = ("yes", "no"),
        n_select   = 1186
    )