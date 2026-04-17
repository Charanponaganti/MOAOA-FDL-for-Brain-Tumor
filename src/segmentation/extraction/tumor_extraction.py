import cv2
import numpy as np
import os

# extract only the brightest part (tumor)
def get_tumor(img):
    vals = np.unique(img)
    if len(vals) < 3: return np.zeros_like(img)

    # brightest region
    best = vals[-1]
    m = (img == best).astype(np.uint8) * 255

    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros_like(m)

    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 100:
        return np.zeros_like(m)

    res = np.zeros_like(m)
    cv2.drawContours(res, [big], -1, 255, -1)
    return res


def run_batch_tumor(in_d, out_d, labels=("yes", "no")):
    for l in labels:
        p1 = os.path.join(in_d, l)
        p2 = os.path.join(out_d, l)
        if not os.path.exists(p2): os.makedirs(p2)

        files = [f for f in os.listdir(p1) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"Batch tumor extraction for {l}")

        for i, f in enumerate(files, 1):
            p = os.path.join(p1, f)
            seg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if seg is None: continue

            tumor = get_tumor(seg)
            cv2.imwrite(os.path.join(p2, f), tumor)
            
            if i % 20 == 0:
                print(f"[{i}/{len(files)}] - tumor done")


if __name__ == "__main__":
    # 4 levels up: extraction -> segmentation -> src -> project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    run_batch_tumor(
        in_d=os.path.join(base, "data/outputs/segmented"),
        out_d=os.path.join(base, "data/outputs/tumor_only"),
        labels=("yes", "no")
    )
    print("Tumor extraction finished.")
