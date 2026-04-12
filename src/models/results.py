import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "data/results")
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Constants (must match classifier.py)
# ---------------------------------------------------------
SEQ_LEN     = 2
HIDDEN_SIZE = 128

# ---------------------------------------------------------
# LSTM Model (same as classifier.py)
# ---------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ---------------------------------------------------------
# Load data and model
# ---------------------------------------------------------
X = np.load(os.path.join(base_dir, "data/features/features.npy"))
y = np.load(os.path.join(base_dir, "data/features/labels.npy"))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# LSTM reshaping: split features into SEQ_LEN timesteps
n_features = X_train.shape[1]
input_size = n_features // SEQ_LEN

def to_tensor(X, y):
    Xt = torch.tensor(X[:, :SEQ_LEN * input_size], dtype=torch.float32)
    Xt = Xt.reshape(-1, SEQ_LEN, input_size).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    return Xt, yt

X_tr_t,  y_tr_t  = to_tensor(X_tr,  y_tr)
X_test_t, y_test_t = to_tensor(X_test, y_test)

# Best hyperparameters — update these after re-running classifier.py
best_lr = 0.00601250713516592
best_batch = 34
best_epochs = 390

# ---------------------------------------------------------
# Retrain and record loss/accuracy per epoch
# ---------------------------------------------------------
print("Retraining to record curves...")

model     = LSTMClassifier(input_size=input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
criterion = nn.CrossEntropyLoss()

dataset = TensorDataset(X_tr_t, y_tr_t)
loader  = DataLoader(dataset, batch_size=best_batch, shuffle=True)

train_losses = []
train_accs   = []
val_losses   = []
val_accs     = []

for epoch in range(1, best_epochs + 1):
    # --- Train ---
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (torch.argmax(out, dim=1) == yb).sum().item()
        total      += yb.size(0)

    train_losses.append(total_loss / len(loader))
    train_accs.append(correct / total)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_out  = model(X_test_t)
        val_loss = criterion(val_out, y_test_t).item()
        val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
        val_acc  = (val_pred == y_test).mean()

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:03d}/{best_epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")


# ---------------------------------------------------------
# 1. Accuracy Curve
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Training Accuracy",   color="blue")
plt.plot(val_accs,   label="Validation Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy Per Epoch")
plt.title("Accuracy Graph")
plt.legend()
plt.tight_layout()
plt.savefig("data/results/accuracy_curve.png", dpi=150)
plt.close()
print("Saved: accuracy_curve.png")


# ---------------------------------------------------------
# 2. Loss Curve
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss",   color="blue")
plt.plot(val_losses,   label="Validation Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss Per Epoch")
plt.title("Loss Graph")
plt.legend()
plt.tight_layout()
plt.savefig("data/results/loss_curve.png", dpi=150)
plt.close()
print("Saved: loss_curve.png")


# ---------------------------------------------------------
# 3. Confusion Matrix + Compute actual metrics
# ---------------------------------------------------------
cm = confusion_matrix(y_test, val_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Compute our actual metrics from the model
import csv
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score

tn, fp, fn, tp = cm.ravel()
sensitivity_val = tp / (tp + fn + 1e-8)
specificity_val = tn / (tn + fp + 1e-8)
f1    = f1_score(y_test, val_pred, average="weighted")
mcc   = matthews_corrcoef(y_test, val_pred)
kappa = cohen_kappa_score(y_test, val_pred)
acc   = (val_pred == y_test).mean()


# ---------------------------------------------------------
# 4. Sensitivity & Specificity Comparison Bar Chart
# MOAOA-FDL = our actual results; others from paper Table 2
# ---------------------------------------------------------
methods     = ["MOAOA-FDL", "SVM-KP", "SVM-RBF", "Decision Tree",
               "CART", "Random Forest", "k-NN", "Linear SVM"]
sensitivity = [round(sensitivity_val * 100, 2),
                            94.73,    95.62,     97.88,
               88.00,       96.00,    80.00,     96.00]
specificity = [round(specificity_val * 100, 2),
                            97.59,    83.71,     91.71,
               80.00,       80.00,    80.00,     80.00]

x     = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, sensitivity, width, label="Sensitivity", color="steelblue")
bars2 = ax.bar(x + width/2, specificity, width, label="Specificity", color="orange")

ax.set_ylabel("Values (%)")
ax.set_title("Sensitivity and Specificity Comparison")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha="right")
ax.set_ylim(70, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "sensitivity_specificity.png"), dpi=150)
plt.close()
print("Saved: sensitivity_specificity.png")


# ---------------------------------------------------------
# 5. Accuracy Comparison Bar Chart
# MOAOA-FDL = our actual accuracy; others from paper Table 2
# ---------------------------------------------------------
acc_values = [round(acc * 100, 2),
              96.18, 89.88, 94.95, 84.00, 88.00, 80.00, 88.00]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["steelblue"] + ["orange"] * (len(methods) - 1)
ax.bar(methods, acc_values, color=colors)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Comparison")
ax.set_ylim(70, 105)
ax.set_xticklabels(methods, rotation=20, ha="right")

for i, v in enumerate(acc_values):
    ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=150)
plt.close()
print("Saved: accuracy_comparison.png")


# ---------------------------------------------------------
# 6. Save metrics to CSV
# ---------------------------------------------------------
metrics = {
    "Accuracy"   : f"{acc*100:.2f}%",
    "Sensitivity": f"{sensitivity_val*100:.2f}%",
    "Specificity": f"{specificity_val*100:.2f}%",
    "F-Score"    : f"{f1*100:.2f}%",
    "MCC"        : f"{mcc:.4f}",
    "Kappa"      : f"{kappa:.4f}",
}

with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for k, v in metrics.items():
        writer.writerow([k, v])

print("Saved: metrics.csv")
print("\nFinal Metrics:")
for k, v in metrics.items():
    print(f"  {k:12}: {v}")

print("\nAll results saved to data/results/ - Done!")