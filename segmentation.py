def get_tumor_mask(img_path, mask_path, N=20, iters=50):
    import cv2
    import numpy as np

    # --- load images ---
    image = cv2.imread(img_path, 0)
    maskimg = cv2.imread(mask_path, 0)

    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if maskimg is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # --- histogram ---
    hist = cv2.calcHist([image], [0], maskimg, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)

    # --- entropy ---
    def entropy(region):
        region = region[region > 0]
        if region.size == 0:
            return 0.0
        region = region / region.sum()
        return -np.sum(region * np.log(region))

    # --- fitness ---
    def shannonforT(T):
        t1, t2 = sorted(map(int, np.round(T)))
        t1 = np.clip(t1, 1, 254)
        t2 = np.clip(t2, 1, 254)
        if t1 >= t2:
            t2 = min(t1 + 1, 255)

        r1 = hist[:t1]
        r2 = hist[t1:t2]
        r3 = hist[t2:]

        return entropy(r1) + entropy(r2) + entropy(r3)

    # --- AOA components ---
    def initialize(N, dim, lb, ub):
        X = np.random.uniform(lb, ub, (N, dim))
        den = np.random.rand(N, dim)
        vol = np.random.rand(N, dim)
        acc = lb + np.random.rand(N, dim) * (ub - lb)
        return X, den, vol, acc

    def update_den_vol(den, vol, den_best, vol_best):
        rand = np.random.rand(*den.shape)
        return den + rand * (den_best - den), vol + rand * (vol_best - vol)

    def compute_TF(t, tmax):
        return np.exp((t - tmax) / tmax)

    def compute_d(t, tmax):
        return np.exp((tmax - t) / tmax) - (t / tmax)

    def update_acc(acc, den, vol, den_best, vol_best, acc_best, TF):
        new_acc = np.zeros_like(acc)
        for i in range(len(acc)):
            if TF <= 0.5:
                mr = np.random.randint(len(acc))
                new_acc[i] = (den[mr] + vol[mr]*acc[mr]) / (den[i]*vol[i] + 1e-9)
            else:
                new_acc[i] = (den_best + vol_best*acc_best) / (den[i]*vol[i] + 1e-9)
        return new_acc

    def normalize_acc(acc):
        return 0.9 * (acc - acc.min()) / (acc.max() - acc.min() + 1e-9) + 0.1

    def update_position(X, acc_norm, best, TF, d, lb, ub):
        C1, C2 = 2, 6
        new_X = np.copy(X)

        for i in range(len(X)):
            rand = np.random.rand()
            if TF <= 0.5:
                rand_idx = np.random.randint(len(X))
                new_X[i] = X[i] + C1*rand*acc_norm[i]*d*(X[rand_idx] - X[i])
            else:
                F = 1 if np.random.rand() < 0.5 else -1
                new_X[i] = best + F*C2*rand*acc_norm[i]*d*(best - X[i])

            new_X[i] = np.clip(new_X[i], lb, ub)

        return new_X

    # --- AOA main ---
    def AOA():
        lb, ub = 1, 254
        X, den, vol, acc = initialize(N, 2, lb, ub)

        best = None
        best_score = -np.inf

        for t in range(iters):
            scores = np.array([shannonforT(x) for x in X])

            idx = np.argmax(scores)
            if scores[idx] > best_score:
                best_score = scores[idx]
                best = X[idx].copy()
                den_best = den[idx].copy()
                vol_best = vol[idx].copy()
                acc_best = acc[idx].copy()

            TF = compute_TF(t, iters)
            d = compute_d(t, iters)

            den, vol = update_den_vol(den, vol, den_best, vol_best)
            acc = update_acc(acc, den, vol, den_best, vol_best, acc_best, TF)
            acc_norm = normalize_acc(acc)
            X = update_position(X, acc_norm, best, TF, d, lb, ub)

        return sorted(best.astype(int))

    # --- get thresholds ---
    t1, t2 = AOA()

   # --- segmentation ---
    # --- segmentation ---
    r3 = image.copy()
    r3[r3 <= t2] = 0

    tumor_mask = (r3 > 0).astype(np.uint8) * 255

    # --- clean mask ---
    kernel = np.ones((5,5), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)

    # --- largest component ---
    num_labels, labels = cv2.connectedComponents(tumor_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax([
            np.sum(labels == i) for i in range(1, num_labels)
        ])
        tumor_mask = (labels == largest_label).astype(np.uint8) * 255

    # =========================
    # 🔥 FILL INSIDE (important)
    # =========================

    filled = tumor_mask.copy()
    h, w = filled.shape
    mask_flood = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(filled, mask_flood, (0,0), 255)
    filled_inv = cv2.bitwise_not(filled)

    # combine to fill holes
    filled = tumor_mask | filled_inv

    # return FILLED tumor region
    return filled

    