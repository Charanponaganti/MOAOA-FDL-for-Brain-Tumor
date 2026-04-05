import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, cohen_kappa_score,
                             classification_report)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last timestep
        return self.fc(out)


# ---------------------------------------------------------
# AOA Hyperparameter Optimization
# ---------------------------------------------------------
class AOA_LSTM:
    def __init__(self, n_particles=10, max_iter=20):
        self.n = n_particles
        self.T = max_iter

        # Search space: [learning_rate, batch_size, hidden_size]
        self.lb = np.array([1e-5, 8,  64])
        self.ub = np.array([1e-2, 64, 256])

    def _decode(self, pos):
        lr          = float(np.clip(pos[0], self.lb[0], self.ub[0]))
        batch_size  = int(np.clip(round(pos[1]), self.lb[1], self.ub[1]))
        hidden_size = int(np.clip(round(pos[2]), self.lb[2], self.ub[2]))
        return lr, batch_size, hidden_size

    def _fitness(self, pos, X_train, y_train, X_val, y_val, input_size):
        lr, batch_size, hidden_size = self._decode(pos)

        model = LSTMClassifier(input_size=input_size,
                               hidden_size=hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Quick training - 5 epochs for fitness evaluation
        dataset = TensorDataset(X_train, y_train)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(5):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=1).cpu().numpy()

        acc = accuracy_score(y_val.cpu().numpy(), preds)
        return acc

    def optimise(self, X_train, y_train, X_val, y_val, input_size):
        rng = np.random.default_rng(42)

        pos = rng.uniform(self.lb, self.ub, (self.n, 3))
        den = rng.random((self.n, 3))
        vol = rng.random((self.n, 3))
        acc = rng.uniform(self.lb, self.ub, (self.n, 3))

        fitness = np.array([self._fitness(pos[i], X_train, y_train,
                                          X_val, y_val, input_size)
                            for i in range(self.n)])

        best_idx = np.argmax(fitness)
        x_best   = pos[best_idx].copy()
        print(f"  AOA Init | best acc: {fitness[best_idx]:.4f} | "
              f"params: {self._decode(x_best)}")

        for t in range(1, self.T + 1):
            TF = np.exp((t - self.T) / self.T)
            d  = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)

            den = den + rng.random((self.n, 3)) * (den[best_idx] - den)
            vol = vol + rng.random((self.n, 3)) * (vol[best_idx] - vol)

            if TF <= 0.5:
                mr  = rng.integers(0, self.n, self.n)
                acc = (den[mr] * vol[mr] * acc[mr]) / (den * vol + 1e-8)
            else:
                acc = (den[best_idx] * vol[best_idx] * acc[best_idx]) / (den * vol + 1e-8)

            a_min, a_max = acc.min(), acc.max()
            acc_norm = (0.1 + 0.8 * (acc - a_min) / (a_max - a_min + 1e-8))

            if TF <= 0.5:
                x_rand = rng.uniform(self.lb, self.ub, (self.n, 3))
                pos = pos + 2 * rng.random((self.n, 3)) * acc_norm * d * (x_rand - pos)
            else:
                F   = np.where(rng.random((self.n, 3)) > 0.5, 1, -1)
                pos = x_best + F * 6 * rng.random((self.n, 3)) * acc_norm * d * (x_best - pos)

            pos = np.clip(pos, self.lb, self.ub)

            fitness = np.array([self._fitness(pos[i], X_train, y_train,
                                              X_val, y_val, input_size)
                                for i in range(self.n)])

            new_best = np.argmax(fitness)
            if fitness[new_best] > fitness[best_idx]:
                best_idx = new_best
                x_best   = pos[new_best].copy()

            print(f"  AOA Iter {t:02d}/{self.T} | best acc: {fitness[best_idx]:.4f} | "
                  f"params: {self._decode(x_best)}")

        return self._decode(x_best)


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------
def evaluate(y_true, y_pred):
    acc     = accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, average="weighted")
    mcc     = matthews_corrcoef(y_true, y_pred)
    kappa   = cohen_kappa_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print("\n--- Results ---")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  F-Score    : {f1*100:.2f}%")
    print(f"  MCC        : {mcc:.4f}")
    print(f"  Kappa      : {kappa:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Tumor", "Normal"]))

    return acc, sensitivity, specificity, f1, mcc, kappa


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":

    # Load features
    X = np.load("data/features/features.npy")
    y = np.load("data/features/labels.npy")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split (70/30 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Validation split from train (for AOA fitness)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # Convert to tensors
    # LSTM expects (batch, seq_len, input_size) -> seq_len=1 here
    def to_tensor(X, y):
        return (torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(y, dtype=torch.long).to(device))

    X_tr_t,  y_tr_t  = to_tensor(X_tr,   y_tr)
    X_val_t, y_val_t = to_tensor(X_val,  y_val)
    X_test_t,y_test_t= to_tensor(X_test, y_test)

    input_size = X_train.shape[1]  # 1186

    # AOA hyperparameter optimization
    print("\nRunning AOA for hyperparameter optimization...")
    best_lr, best_batch, best_hidden = AOA_LSTM(
        n_particles=10, max_iter=20
    ).optimise(X_tr_t, y_tr_t, X_val_t, y_val_t, input_size)

    print(f"\nBest params -> lr: {best_lr} | batch: {best_batch} | hidden: {best_hidden}")

    # Final training with best params
    print("\nFinal training with best hyperparameters...")
    model = LSTMClassifier(input_size=input_size,
                           hidden_size=best_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=best_batch, shuffle=True)

    # Full training - 100 epochs
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/100 | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()

    evaluate(y_test, preds)

    # Save model
    torch.save(model.state_dict(), "data/features/lstm_model.pth")
    print("\nModel saved to data/features/lstm_model.pth")