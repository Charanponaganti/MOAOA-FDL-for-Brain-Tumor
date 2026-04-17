import cv2
import numpy as np
import os

# shannon entropy logic
def calc_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def get_multi_entropy(h_norm, t_list):
    t_list = sorted([int(t) for t in t_list])
    regs = []
    
    last = 0
    for t in t_list:
        regs.append(h_norm[last:t])
        last = t
    regs.append(h_norm[last:])

    ent = 0
    for r in regs:
        p_sum = r.sum()
        if p_sum > 0:
            ent += calc_entropy(r / p_sum)
    return ent


class ThresholdFinder:
    def __init__(self, n=20, iters=50, k=3):
        self.n = n
        self.T = iters
        self.k = k
        self.min_v = 1
        self.max_v = 254

    def fitness(self, positions, h_norm):
        f = []
        for p in positions:
            t = np.clip(p, self.min_v, self.max_v)
            f.append(get_multi_entropy(h_norm, t))
        return np.array(f)

    def find(self, img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hn = h / h.sum()

        r = np.random.default_rng()

        # init
        p = r.uniform(self.min_v, self.max_v, (self.n, self.k))
        d = r.random((self.n, self.k))
        v = r.random((self.n, self.k))
        a = r.uniform(self.min_v, self.max_v, (self.n, self.k))

        fit = self.fitness(p, hn)
        best_i = np.argmax(fit)
        x_best = p[best_i].copy()

        for t in range(1, self.T + 1):
            tf = np.exp((t - self.T) / self.T)
            dist_val = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)

            d = d + r.random((self.n, self.k)) * (d[best_i] - d)
            v = v + r.random((self.n, self.k)) * (v[best_i] - v)

            if tf <= 0.5:
                # exploration
                m_idx = r.integers(0, self.n, self.n)
                a = (d[m_idx] * v[m_idx] * a[m_idx]) / (d * v + 1e-8)
            else:
                # exploitation
                a = (d[best_i] * v[best_i] * a[best_i]) / (d * v + 1e-8)

            a_min, a_max = a.min(), a.max()
            an = (0.1 + 0.8 * (a - a_min) / (a_max - a_min) if a_max > a_min else np.full_like(a, 0.5))

            if tf <= 0.5:
                xr = r.uniform(self.min_v, self.max_v, (self.n, self.k))
                p = p + 2 * r.random((self.n, self.k)) * an * dist_val * (xr - p)
            else:
                F = np.where(r.random((self.n, self.k)) > 0.5, 1, -1)
                p = x_best + F * 6 * r.random((self.n, self.k)) * an * dist_val * (x_best - p)

            p = np.clip(p, self.min_v, self.max_v)
            fit = self.fitness(p, hn)
            new_b = np.argmax(fit)

            if fit[new_b] > fit[best_i]:
                best_i = new_b
                x_best = p[new_b].copy()

        return sorted([int(round(val)) for val in x_best])


def do_segment(img, k=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t_vals = ThresholdFinder(k=k).find(img)
    lvls = [0] + t_vals + [256]
    
    n_r = len(lvls) - 1
    ints = [int(i * 255 / (n_r - 1)) for i in range(n_r)]
    
    res = np.zeros_like(img)
    for i in range(n_r):
        low, high = lvls[i], lvls[i+1]
        if i == n_r - 1:
            m = (img >= low) & (img <= high)
        else:
            m = (img >= low) & (img < high)
        res[m] = ints[i]

    return res, t_vals

def run_dataset_seg(in_dir, out_dir, labels=("yes", "no"), k=3):
    for l in labels:
        p_in = os.path.join(in_dir, l)
        p_out = os.path.join(out_dir, l)
        if not os.path.exists(p_out): os.makedirs(p_out)

        files = [f for f in os.listdir(p_in) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"Seg processing {l} - {len(files)} files")

        for i, f in enumerate(files, 1):
            img = cv2.imread(os.path.join(p_in, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            seg, t = do_segment(img, k)
            out_f = os.path.splitext(f)[0] + ".png"
            cv2.imwrite(os.path.join(p_out, out_f), seg)
            
            if i % 20 == 0:
                print(f"{i}/{len(files)} - {t}")


if __name__ == "__main__":
    # 4 levels up: algorithms -> segmentation -> src -> project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    run_dataset_seg(
        in_dir=os.path.join(base, "data/outputs/augmented"),
        out_dir=os.path.join(base, "data/outputs/segmented"),
        labels=("yes", "no"),
        k=3
    )
    print("finished.")
