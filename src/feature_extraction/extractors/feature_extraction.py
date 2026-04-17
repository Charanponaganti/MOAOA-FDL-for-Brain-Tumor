import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import timm

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract features from mobilenet 1st model
class MobFeats(nn.Module):
   
    layers_to_pool = {3, 13, 16, 17, 18}

    def __init__(self):
        super().__init__()
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.f = m.features
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        res = []
        for i, layer in enumerate(self.f):
            x = layer(x)
            res.append(self.avg(x).flatten(1))
            if i in self.layers_to_pool:
                res.append(self.mx(x).flatten(1))
        return torch.cat(res, dim=1)


# Extract features from efficientnet 2nd model
class EffFeats(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("efficientnet_b0", pretrained=True)
        self.stem = model.conv_stem
        self.bn1 = model.bn1
        self.blocks = model.blocks
        self.head = model.conv_head
        self.bn2 = model.bn2
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        res = []
        # stem part
        x = self.bn1(self.stem(x))
        res.append(self.pool(x).flatten(1))
        # all blocks
        for s in self.blocks:
            for b in s:
                x = b(x)
                res.append(self.pool(x).flatten(1))
        # head
        x = self.bn2(self.head(x))
        res.append(self.pool(x).flatten(1))
        return torch.cat(res, dim=1)

# get the total features
def get_models():
    m1 = MobFeats().to(device).eval()
    m2 = EffFeats().to(device).eval()

    # check sizes
    d = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out1 = m1(d).shape[1]
        out2 = m2(d).shape[1]
    
    print(f"MobNet features: {out1}")
    print(f"EffNet features: {out2}")
    print(f"Total: {out1+out2}")
    return m1, m2


# transformations
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    # grayscale to rgb
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    t = img_transform(img).unsqueeze(0).to(device)
    return t


def select_features(feats, n=1186):
    # entropy based selection
    bins = 50
    scores = []
    for i in range(feats.shape[1]):
        c = feats[:, i]
        h, _ = np.histogram(c, bins=bins, density=True)
        h = h[h > 0]
        e = -np.sum(h * np.log2(h + 1e-10))
        scores.append(e)

    scores = np.array(scores)
    idx = np.argsort(scores)[::-1][:n]
    idx = np.sort(idx)
    return feats[:, idx], idx


def run_extraction(in_dir, out_dir, classes=("yes", "no"), n=1186):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    m1, m2 = get_models()
    all_f = []
    all_l = []

    for li, name in enumerate(classes):
        p = os.path.join(in_dir, name)
        files = [f for f in os.listdir(p) if f.lower().endswith(".png")]
        
        print(f"\nProcessing {name} - {len(files)} files")
        for i, f in enumerate(files, 1):
            img = load_img(os.path.join(p, f))
            if img is None: continue

            with torch.no_grad():
                f1 = m1(img).cpu().numpy().flatten()
                f2 = m2(img).cpu().numpy().flatten()

            fused = np.concatenate([f1, f2])
            all_f.append(fused)
            all_l.append(li)

            if i % 50 == 0:
                print(f"Done {i}/{len(files)}")

    all_f = np.array(all_f)
    all_l = np.array(all_l)

    print("Selecting top features...")
    final_f, final_idx = select_features(all_f, n)
    
    np.save(os.path.join(out_dir, "features.npy"), final_f)
    np.save(os.path.join(out_dir, "labels.npy"), all_l)
    np.save(os.path.join(out_dir, "indices.npy"), final_idx)
    print("Saved features successfully.")


if __name__ == "__main__":
    # 4 levels up: extractors -> feature_extraction -> src -> project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    run_extraction(
        in_dir = os.path.join(base, "data/outputs/segmented"),
        out_dir = os.path.join(base, "data/outputs/features"),
        classes = ("yes", "no"),
        n = 1186
    )
