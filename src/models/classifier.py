import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, cohen_kappa_score,
                             classification_report)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
SEQ_LEN     = 2     # Number of LSTM timesteps (feature vector split)
HIDDEN_SIZE = 128   # Fixed LSTM hidden dimension


# ---------------------------------------------------------
# LSTM Model
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
        # x: (batch, SEQ_LEN, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last timestep output
        return self.fc(out)


# ---------------------------------------------------------
# Multi-Objective AOA (MOAOA) Hyperparameter Optimization
# Paper: "Multi-objective Archimedes Optimization Algorithm"
# Optimizes 3 objectives simultaneously:
#   1. Classification error  (minimize)
#   2. Training cost / epochs (minimize)
#   3. Loss instability       (minimize)
# Uses Pareto dominance, non-dominated sorting, crowding
# distance, and an archive of trade-off solutions.
# ---------------------------------------------------------
class MOAOA_LSTM:
    def __init__(self, n_particles=15, max_iter=30):
        self.n = n_particles
        self.T = max_iter
        self.n_obj = 3

        # Search space: [learning_rate, batch_size, epoch_count]
        self.lb = np.array([1e-5, 8,    50])
        self.ub = np.array([1e-2, 64, 1200])

        # Archive of non-dominated (Pareto-optimal) solutions
        self.archive = []
        self.archive_max = 50

    def _decode(self, pos):
        lr         = float(np.clip(pos[0], self.lb[0], self.ub[0]))
        batch_size = int(np.clip(round(pos[1]), self.lb[1], self.ub[1]))
        epochs     = int(np.clip(round(pos[2]), self.lb[2], self.ub[2]))
        return lr, batch_size, epochs

    # ---------- multi-objective evaluation ----------
    def _evaluate(self, pos, X_train, y_train, X_val, y_val, input_size):
        lr, batch_size, epochs = self._decode(pos)

        model     = LSTMClassifier(input_size=input_size).to(device)
        optim     = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        eval_epochs = max(5, epochs // 20)
        dataset = TensorDataset(X_train, y_train)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        model.train()
        for _ in range(eval_epochs):
            ep_loss = 0
            for xb, yb in loader:
                optim.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optim.step()
                ep_loss += loss.item()
            losses.append(ep_loss / len(loader))

        # Obj 1 — classification error (lower = better)
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=1).cpu().numpy()
        error = 1.0 - accuracy_score(y_val.cpu().numpy(), preds)

        # Obj 2 — normalized training cost (lower = cheaper)
        cost = (epochs - self.lb[2]) / (self.ub[2] - self.lb[2])

        # Obj 3 — loss instability: coefficient of variation (lower = stabler)
        stability = np.std(losses) / (np.mean(losses) + 1e-8) if len(losses) > 1 else 1.0

        return np.array([error, cost, stability])

    # ---------- Pareto helpers ----------
    @staticmethod
    def _dominates(a, b):
        """A dominates B if A <= B everywhere and A < B somewhere."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _non_dominated_indices(self, objs):
        n = len(objs)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i != j and not dominated[j] and self._dominates(objs[j], objs[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _crowding_distance(self, objs):
        n = len(objs)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(objs.shape[1]):
            idx = np.argsort(objs[:, m])
            dist[idx[0]]  = np.inf
            dist[idx[-1]] = np.inf
            span = objs[idx[-1], m] - objs[idx[0], m]
            if span < 1e-10:
                continue
            for k in range(1, n - 1):
                dist[idx[k]] += (objs[idx[k+1], m] - objs[idx[k-1], m]) / span
        return dist

    # ---------- archive management ----------
    def _update_archive(self, positions, objectives):
        all_pos = list(positions)
        all_obj = list(objectives)
        for p, o in self.archive:
            all_pos.append(p)
            all_obj.append(o)
        all_pos = np.array(all_pos)
        all_obj = np.array(all_obj)

        nd = self._non_dominated_indices(all_obj)
        if len(nd) > self.archive_max:
            cd   = self._crowding_distance(all_obj[nd])
            keep = np.argsort(cd)[::-1][:self.archive_max]
            nd   = nd[keep]
        self.archive = [(all_pos[i].copy(), all_obj[i].copy()) for i in nd]

    def _select_leader(self, rng):
        if len(self.archive) <= 1:
            return self.archive[0][0] if self.archive else None
        objs = np.array([o for _, o in self.archive])
        cd   = self._crowding_distance(objs)
        cd   = np.where(np.isinf(cd), np.nanmax(cd[np.isfinite(cd)]) * 2
                        if np.any(np.isfinite(cd)) else 1.0, cd)
        probs = cd / (cd.sum() + 1e-10)
        return self.archive[rng.choice(len(self.archive), p=probs)][0]

    # ---------- main optimisation loop ----------
    def optimise(self, X_train, y_train, X_val, y_val, input_size):
        rng = np.random.default_rng(42)

        pos = rng.uniform(self.lb, self.ub, (self.n, 3))
        den = rng.random((self.n, 3))
        vol = rng.random((self.n, 3))
        accel = rng.uniform(self.lb, self.ub, (self.n, 3))

        objectives = np.array([self._evaluate(pos[i], X_train, y_train,
                                              X_val, y_val, input_size)
                                for i in range(self.n)])
        self._update_archive(pos, objectives)

        best_sol = min(self.archive, key=lambda x: x[1][0])
        x_best     = best_sol[0].copy()
        best_error = best_sol[1][0]

        print(f"  MOAOA Init | best acc: {(1-best_error)*100:.2f}% | "
              f"archive: {len(self.archive)} | params: {self._decode(x_best)}")

        for t in range(1, self.T + 1):
            TF = np.exp((t - self.T) / self.T)
            d  = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)

            leader   = self._select_leader(rng)
            best_idx = np.argmin(objectives[:, 0])

            den = den + rng.random((self.n, 3)) * (den[best_idx] - den)
            vol = vol + rng.random((self.n, 3)) * (vol[best_idx] - vol)

            if TF <= 0.5:
                mr    = rng.integers(0, self.n, self.n)
                accel = (den[mr] * vol[mr] * accel[mr]) / (den * vol + 1e-8)
            else:
                accel = (den[best_idx] * vol[best_idx] * accel[best_idx]) / (den * vol + 1e-8)

            a_min, a_max = accel.min(), accel.max()
            acc_norm = 0.1 + 0.8 * (accel - a_min) / (a_max - a_min + 1e-8)

            if TF <= 0.5:
                x_rand = rng.uniform(self.lb, self.ub, (self.n, 3))
                pos = pos + 2 * rng.random((self.n, 3)) * acc_norm * d * (x_rand - pos)
            else:
                F   = np.where(rng.random((self.n, 3)) > 0.5, 1, -1)
                pos = leader + F * 6 * rng.random((self.n, 3)) * acc_norm * d * (leader - pos)

            pos = np.clip(pos, self.lb, self.ub)

            objectives = np.array([self._evaluate(pos[i], X_train, y_train,
                                                  X_val, y_val, input_size)
                                    for i in range(self.n)])
            self._update_archive(pos, objectives)

            best_sol = min(self.archive, key=lambda x: x[1][0])
            if best_sol[1][0] < best_error:
                x_best     = best_sol[0].copy()
                best_error = best_sol[1][0]

            print(f"  MOAOA Iter {t:02d}/{self.T} | best acc: {(1-best_error)*100:.2f}% | "
                  f"archive: {len(self.archive)} | params: {self._decode(x_best)}")

        final = min(self.archive, key=lambda x: x[1][0])
        return self._decode(final[0])


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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    X = np.load(os.path.join(base_dir, "data/features/features.npy"))
    y = np.load(os.path.join(base_dir, "data/features/labels.npy"))

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

    # LSTM expects (batch, seq_len, input_size)
    # Split features into SEQ_LEN=2 timesteps
    n_features = X_train.shape[1]           # 1186
    input_size = n_features // SEQ_LEN      # 593

    def to_tensor(X, y):
        Xt = torch.tensor(X[:, :SEQ_LEN * input_size], dtype=torch.float32)
        Xt = Xt.reshape(-1, SEQ_LEN, input_size).to(device)
        yt = torch.tensor(y, dtype=torch.long).to(device)
        return Xt, yt

    X_tr_t,  y_tr_t  = to_tensor(X_tr,   y_tr)
    X_val_t, y_val_t = to_tensor(X_val,  y_val)
    X_test_t,y_test_t= to_tensor(X_test, y_test)

    # MOAOA hyperparameter optimization (3 objectives)
    print("\nRunning MOAOA for hyperparameter optimization...")
    print(f"  Objectives: error, training cost, loss stability")
    print(f"  Optimizing: learning_rate, batch_size, epoch_count")
    best_lr, best_batch, best_epochs = MOAOA_LSTM(
        n_particles=15, max_iter=30
    ).optimise(X_tr_t, y_tr_t, X_val_t, y_val_t, input_size)

    print(f"\nBest params -> lr: {best_lr} | batch: {best_batch} | epochs: {best_epochs}")

    # Final training with best params
    print(f"\nFinal training with best hyperparameters ({best_epochs} epochs)...")
    model = LSTMClassifier(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=best_batch, shuffle=True)

    # Full training — using AOA-optimized epoch count
    log_every = max(1, best_epochs // 10)
    for epoch in range(1, best_epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % log_every == 0 or epoch == best_epochs:
            print(f"  Epoch {epoch:03d}/{best_epochs} | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()

    evaluate(y_test, preds)

    # Save model
    torch.save(model.state_dict(), "data/features/lstm_model.pth")
    print("\nModel saved to data/features/lstm_model.pth")