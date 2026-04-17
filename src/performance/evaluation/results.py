import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef, cohen_kappa_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
import sys

# Ensure project root is in path for imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.classification.models.lstm import MyLSTM, N_STEPS, H_SIZE

# Get the project root directory
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
out_dir = os.path.join(path, "data/outputs/results")
if not os.path.exists(out_dir): os.makedirs(out_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params


# load data
x_data = np.load(os.path.join(path, "data/outputs/features/features.npy"))
y_data = np.load(os.path.join(path, "data/outputs/features/labels.npy"))

sc = StandardScaler()
x_data = sc.fit_transform(x_data)

x_tr, x_te, y_tr, y_te = train_test_split(x_data, y_data, test_size=0.3, random_state=42, stratify=y_data)

# prep tensors
dim = x_tr.shape[1] // N_STEPS
def to_t(X, y):
    xt = torch.tensor(X[:, :N_STEPS*dim], dtype=torch.float32).reshape(-1, N_STEPS, dim).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    return xt, yt

xtr, ytr = to_t(x_tr, y_tr)
xte, yte = to_t(x_te, y_te)

# best params from previous run
lr = 0.00601250713516592
bs = 34
eps = 390

print("Training to get curves...")
model = MyLSTM(dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()

loader = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True)

t_loss, t_acc = [], []
v_loss, v_acc = [], []

for e in range(1, eps + 1):
    model.train()
    L, ok, tot = 0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        optimizer.step()
        L += loss.item()
        ok += (torch.argmax(out, 1) == yb).sum().item()
        tot += yb.size(0)
    
    t_loss.append(L / len(loader))
    t_acc.append(ok / tot)

    model.eval()
    with torch.no_grad():
        out_v = model(xte)
        v_l = crit(out_v, yte).item()
        preds = torch.argmax(out_v, 1).cpu().numpy()
        v_a = (preds == y_te).mean()
    
    v_loss.append(v_l)
    v_acc.append(v_a)

    if e % 20 == 0:
        print(f"E {e} | loss {t_loss[-1]:.4f} | acc {v_a:.4f}")


# 1. Acc Graph
plt.figure()
plt.plot(t_acc, label="train")
plt.plot(v_acc, label="test")
plt.title("Accuracy Plot")
plt.legend()
plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
plt.close()

# 2. Loss Graph
plt.figure()
plt.plot(t_loss, label="train")
plt.plot(v_loss, label="test")
plt.title("Loss Plot")
plt.legend()
plt.savefig(os.path.join(out_dir, "loss_curve.png"))
plt.close()

# 3. CM
cm = confusion_matrix(y_te, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Results")
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
plt.close()

# get metrics
tn, fp, fn, tp = cm.ravel()
s1 = tp / (tp + fn + 1e-8)
s2 = tn / (tn + fp + 1e-8)
f1 = f1_score(y_te, preds, average="weighted")
mcc = matthews_corrcoef(y_te, preds)
kap = cohen_kappa_score(y_te, preds)
acc = (preds == y_te).mean()

# plots for comparison
names = ["Mine", "SVM-KP", "SVM-RBF", "Tree", "CART", "RF", "kNN", "LSVM"]
sens_vals = [round(s1*100, 2), 94.73, 95.62, 97.88, 88.0, 96.0, 80.0, 96.0]
spec_vals = [round(s2*100, 2), 97.59, 83.71, 91.71, 80.0, 80.0, 80.0, 80.0]

x = np.arange(len(names))
w = 0.3
plt.figure(figsize=(10,6))
plt.bar(x - w/2, sens_vals, w, label="Sens")
plt.bar(x + w/2, spec_vals, w, label="Spec")
plt.xticks(x, names, rotation=45)
plt.legend()
plt.title("Comparison Chart")
plt.savefig(os.path.join(out_dir, "sensitivity_specificity.png"))
plt.close()

accs = [round(acc*100, 2), 96.18, 89.88, 94.95, 84.0, 88.0, 80.0, 88.0]
plt.figure(figsize=(10,6))
plt.bar(names, accs, color=["blue"] + ["gray"]*7)
plt.title("Final Accuracy Comparison")
plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"))
plt.close()

# csv save
with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Val"])
    w.writerow(["Acc", f"{acc*100:.2f}%"])
    w.writerow(["Sens", f"{s1*100:.2f}%"])
    w.writerow(["Spec", f"{s2*100:.2f}%"])
    w.writerow(["F1", f"{f1*100:.2f}%"])
    w.writerow(["MCC", f"{mcc:.4f}"])
    w.writerow(["Kappa", f"{kap:.4f}"])

print("All done. Files saved to:", out_dir)
