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

# --- settings ---
n_steps = 2
h_size = 128


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=h_size, num_layers=2, num_classes=2, dropout=0.5):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, n_steps, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)


# Multi-Objective AOA
class MOAOA_Optimizer:
    def __init__(self, pop_size=15, iters=30):
        self.n = pop_size
        self.T = iters
        
        # [lr, batch_size, epochs]
        self.lower_b = np.array([1e-5, 8, 50])
        self.upper_b = np.array([1e-2, 64, 1200])

        self.archive = []
        self.archive_limit = 50

    def get_params(self, pos):
        lr = float(np.clip(pos[0], self.lower_b[0], self.upper_b[0]))
        bs = int(np.clip(round(pos[1]), self.lower_b[1], self.upper_b[1]))
        eps = int(np.clip(round(pos[2]), self.lower_b[2], self.upper_b[2]))
        return lr, bs, eps

    def fitness_func(self, pos, X_train, y_train, X_val, y_val, input_size):
        lr, bs, eps = self.get_params(pos)

        model = MyLSTM(input_size=input_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        # small training for eval
        eval_eps = max(5, eps // 20)
        ds = TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=bs, shuffle=True)

        loss_list = []
        model.train()
        for _ in range(eval_eps):
            L = 0
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                L += loss.item()
            loss_list.append(L / len(loader))

        # 1. Error
        model.eval()
        with torch.no_grad():
            p = torch.argmax(model(X_val), dim=1).cpu().numpy()
        err = 1.0 - accuracy_score(y_val.cpu().numpy(), p)

        # 2. Cost
        cost = (eps - self.lower_b[2]) / (self.upper_b[2] - self.lower_b[2])

        # 3. Stability
        if len(loss_list) > 1:
            stb = np.std(loss_list) / (np.mean(loss_list) + 1e-8)
        else:
            stb = 1.0

        return np.array([err, cost, stb])

    def check_dominance(self, a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def get_non_dominated(self, objs):
        n = len(objs)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]: continue
            for j in range(n):
                if i != j and not dominated[j] and self.check_dominance(objs[j], objs[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def calc_crowding(self, objs):
        n = len(objs)
        if n <= 2: return np.full(n, np.inf)
        dists = np.zeros(n)
        for m in range(objs.shape[1]):
            idx = np.argsort(objs[:, m])
            dists[idx[0]] = np.inf
            dists[idx[-1]] = np.inf
            span = objs[idx[-1], m] - objs[idx[0], m]
            if span < 1e-10: continue
            for k in range(1, n - 1):
                dists[idx[k]] += (objs[idx[idx[k+1]], m] - objs[idx[idx[k-1]], m]) / span
        return dists

    def update_archive(self, pos_list, obj_list):
        all_p = list(pos_list)
        all_o = list(obj_list)
        for p, o in self.archive:
            all_p.append(p)
            all_o.append(o)
        
        all_p = np.array(all_p)
        all_o = np.array(all_o)

        nd = self.get_non_dominated(all_o)
        if len(nd) > self.archive_limit:
            # fix this part later maybe
            dists = np.zeros(len(nd))
            # simplified crowding
            idx_sorted = np.argsort(all_o[nd, 0])
            dists[idx_sorted[0]] = np.inf
            dists[idx_sorted[-1]] = np.inf
            for i in range(1, len(nd)-1):
                dists[idx_sorted[i]] = all_o[nd[idx_sorted[i+1]], 0] - all_o[nd[idx_sorted[i-1]], 0]
            
            keep = np.argsort(dists)[::-1][:self.archive_limit]
            nd = nd[keep]
            
        self.archive = [(all_p[i].copy(), all_o[i].copy()) for i in nd]

    def run_optimization(self, X_train, y_train, X_val, y_val, input_size):
        rand = np.random.default_rng(42)

        pos = rand.uniform(self.lower_b, self.upper_b, (self.n, 3))
        den = rand.random((self.n, 3))
        vol = rand.random((self.n, 3))
        acc = rand.uniform(self.lower_b, self.upper_b, (self.n, 3))

        objs = np.array([self.fitness_func(pos[i], X_train, y_train, X_val, y_val, input_size) for i in range(self.n)])
        self.update_archive(pos, objs)

        # start loop
        for t in range(1, self.T + 1):
            tf = np.exp((t - self.T) / self.T)
            d = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)

            best_idx = np.argmin(objs[:, 0])
            x_best = pos[best_idx].copy()

            den = den + rand.random((self.n, 3)) * (den[best_idx] - den)
            vol = vol + rand.random((self.n, 3)) * (vol[best_idx] - vol)

            if tf <= 0.5:
                r_idx = rand.integers(0, self.n, self.n)
                acc = (den[r_idx] * vol[r_idx] * acc[r_idx]) / (den * vol + 1e-8)
            else:
                acc = (den[best_idx] * vol[best_idx] * acc[best_idx]) / (den * vol + 1e-8)

            a_low, a_high = acc.min(), acc.max()
            acc_norm = 0.1 + 0.8 * (acc - a_low) / (a_high - a_low + 1e-8)

            if tf <= 0.5:
                x_rand = rand.uniform(self.lower_b, self.upper_b, (self.n, 3))
                pos = pos + 2 * rand.random((self.n, 3)) * acc_norm * d * (x_rand - pos)
            else:
                F = np.where(rand.random((self.n, 3)) > 0.5, 1, -1)
                # pick a random one from archive as leader
                lead = self.archive[rand.choice(len(self.archive))][0]
                pos = lead + F * 6 * rand.random((self.n, 3)) * acc_norm * d * (lead - pos)

            pos = np.clip(pos, self.lower_b, self.upper_b)
            objs = np.array([self.fitness_func(pos[i], X_train, y_train, X_val, y_val, input_size) for i in range(self.n)])
            self.update_archive(pos, objs)

            tmp_best = min(self.archive, key=lambda x: x[1][0])
            print(f"Iter {t}/{self.T} | Best Acc: {(1-tmp_best[1][0])*100:.2f}%")

        final_sol = min(self.archive, key=lambda x: x[1][0])
        return self.get_params(final_sol[0])


def do_evaluation(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}%")
    print(f"F1: {f1*100:.2f}% | MCC: {mcc:.4f} | Kappa: {kap:.4f}")
    return acc, sens, spec, f1, mcc, kap


# MAIN SCRIPT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # path stuff
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features = np.load(os.path.join(path, "data/features/features.npy"))
    labels = np.load(os.path.join(path, "data/features/labels.npy"))

    import sklearn.preprocessing
    sc = sklearn.preprocessing.StandardScaler()
    X_scaled = sc.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, random_state=42, stratify=labels)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # prepare tensors
    in_dim = X_train.shape[1] // n_steps
    
    def prep_data(X, y):
        xt = torch.tensor(X[:, :n_steps * in_dim], dtype=torch.float32).reshape(-1, n_steps, in_dim).to(device)
        yt = torch.tensor(y, dtype=torch.long).to(device)
        return xt, yt

    xtr, ytr = prep_data(X_tr, y_tr)
    xval, yval = prep_data(X_val, y_val)
    xtest, ytest = prep_data(X_test, y_test)

    # optimize
    print("\nStarting optimization...")
    opt_lr, opt_bs, opt_eps = MOAOA_Optimizer(15, 30).run_optimization(xtr, ytr, xval, yval, in_dim)
    print(f"Best: lr={opt_lr}, bs={opt_bs}, eps={opt_eps}")

    # train final model
    m = MyLSTM(in_dim).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=opt_lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=opt_bs, shuffle=True)
    for e in range(1, opt_eps + 1):
        m.train()
        L = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            l = loss_fn(m(xb), yb)
            l.backward()
            optimizer.step()
            L += l.item()
        if e % 50 == 0 or e == opt_eps:
            print(f"Epoch {e}/{opt_eps} - loss: {L/len(loader):.4f}")

    # final eval
    m.eval()
    with torch.no_grad():
        preds = torch.argmax(m(xtest), dim=1).cpu().numpy()
    do_evaluation(y_test, preds)

    torch.save(m.state_dict(), "data/features/lstm_model.pth")
    print("Done.")