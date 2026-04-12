import cv2
import numpy as np
import os

# ---------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------
def shannon_entropy(prob):
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def multi_threshold_entropy(hist_norm, thresholds):
    """
    Compute entropy for multiple thresholds
    thresholds = sorted list like [t1, t2, t3]
    """
    thresholds = sorted([int(t) for t in thresholds])
    regions = []
    
    prev = 0
    for t in thresholds:
        regions.append(hist_norm[prev:t])
        prev = t
    regions.append(hist_norm[prev:])  # last region

    total_entropy = 0

    for region in regions:
        p = region.sum()
        if p > 0:
            total_entropy += shannon_entropy(region / p)

    return total_entropy


# ---------------------------------------------------------
# MULTI-THRESHOLD AOA
# ---------------------------------------------------------
class AOA_Multi:
    def __init__(self, n_particles=20, max_iter=50, n_thresholds=3):
        self.n = n_particles
        self.T = max_iter
        self.k = n_thresholds
        self.lb = 1
        self.ub = 254

    def _fitness(self, positions, hist_norm):
        fitness = []
        for pos in positions:
            thresholds = np.clip(pos, self.lb, self.ub)
            fitness.append(multi_threshold_entropy(hist_norm, thresholds))
        return np.array(fitness)

    def optimise(self, img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()

        rng = np.random.default_rng()

        # Initialize particles (each particle = vector of thresholds)
        pos = rng.uniform(self.lb, self.ub, (self.n, self.k))
        den = rng.random((self.n, self.k))
        vol = rng.random((self.n, self.k))
        acc = rng.uniform(self.lb, self.ub, (self.n, self.k))

        fitness = self._fitness(pos, hist_norm)
        best_idx = np.argmax(fitness)
        x_best = pos[best_idx].copy()

        for t in range(1, self.T + 1):
            TF = np.exp((t - self.T) / self.T)
            d = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)

            # Update density & volume
            den = den + rng.random((self.n, self.k)) * (den[best_idx] - den)
            vol = vol + rng.random((self.n, self.k)) * (vol[best_idx] - vol)

            # Acceleration update
            if TF <= 0.5:
                mr = rng.integers(0, self.n, self.n)
                acc = (den[mr] * vol[mr] * acc[mr]) / (den * vol + 1e-8)
            else:
                acc = (den[best_idx] * vol[best_idx] * acc[best_idx]) / (den * vol + 1e-8)

            # Normalize acceleration
            a_min, a_max = acc.min(), acc.max()
            acc_norm = (0.1 + 0.8 * (acc - a_min) / (a_max - a_min)
                        if a_max > a_min else np.full_like(acc, 0.5))

            # Position update
            if TF <= 0.5:
                x_rand = rng.uniform(self.lb, self.ub, (self.n, self.k))
                pos = pos + 2 * rng.random((self.n, self.k)) * acc_norm * d * (x_rand - pos)
            else:
                F = np.where(rng.random((self.n, self.k)) > 0.5, 1, -1)
                pos = x_best + F * 6 * rng.random((self.n, self.k)) * acc_norm * d * (x_best - pos)

            pos = np.clip(pos, self.lb, self.ub)

            # Evaluate
            fitness = self._fitness(pos, hist_norm)
            new_best = np.argmax(fitness)

            if fitness[new_best] > fitness[best_idx]:
                best_idx = new_best
                x_best = pos[new_best].copy()

        return sorted([int(round(t)) for t in x_best])


# ---------------------------------------------------------
# SEGMENT USING MULTIPLE THRESHOLDS
# ---------------------------------------------------------
def segment_multi(img, n_thresholds=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresholds = AOA_Multi(n_thresholds=n_thresholds).optimise(img)

    levels = [0] + thresholds + [256]
    
    # ✅ Explicit intensity values for each region
    n_regions = len(levels) - 1
    intensity_values = [int(i * 255 / (n_regions - 1)) for i in range(n_regions)]
    # → gives exactly [0, 85, 170, 255] for 4 regions
    
    segmented = np.zeros_like(img)

    for i in range(n_regions):
        lower = levels[i]
        upper = levels[i + 1]

        if i == n_regions - 1:
            mask = (img >= lower) & (img <= upper)
        else:
            mask = (img >= lower) & (img < upper)

        segmented[mask] = intensity_values[i]

    return segmented, thresholds

def segment_dataset_multi(input_dir, output_dir, labels=("yes", "no"), n_thresholds=3):
    for label in labels:
        in_path = os.path.join(input_dir, label)
        out_path = os.path.join(output_dir, label)
        os.makedirs(out_path, exist_ok=True)

        files = [f for f in os.listdir(in_path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        print(f"[MULTI] Processing {label} ({len(files)} images)")

        for i, fname in enumerate(files, 1):
            img = cv2.imread(os.path.join(in_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            segmented, thresholds = segment_multi(img, n_thresholds)

            # ✅ Force PNG extension to avoid JPEG compression artifacts
            out_fname = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(out_path, out_fname), segmented)

            print(f"[{i}/{len(files)}] {fname} -> {thresholds}")
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    segment_dataset_multi(
        input_dir=os.path.join(base_dir, "data/augmented"),
        output_dir=os.path.join(base_dir, "data/segmented_multi"),
        labels=("yes", "no"),
        n_thresholds=3
    )

    print("Done [OK]")