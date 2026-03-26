import cv2
import numpy as np
import os
from scipy import ndimage

input_folder   = "archive/no"
mask_folder    = "archive/mask_no"
stripped_folder = "archive/strip_no"

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(stripped_folder, exist_ok=True)


def skull_strip(img):
    # 1. Head mask via Otsu
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    head_mask = np.zeros_like(img)
    cv2.drawContours(head_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    # Fill holes (sinuses, eye sockets)
    flood = head_mask.copy()
    cv2.floodFill(flood, np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8), (0, 0), 255)
    head_mask = cv2.bitwise_or(head_mask, cv2.bitwise_not(flood))

    # 2. Distance transform — every pixel's distance from head boundary
    dist = cv2.distanceTransform(head_mask, cv2.DIST_L2, 5)

    # 3. Auto-detect skull thickness from intensity-distance profile
    #    Skull = bright ring just inside boundary → find its peak, then where it drops
    means = {}
    for d in range(1, 35):
        band = (dist >= d) & (dist < d+1) & (head_mask > 0)
        if band.sum() > 50:
            means[d] = img[band].mean()

    # Find the intensity peak within first 20px (= skull peak)
    skull_region = {d: m for d, m in means.items() if d <= 20}
    peak_d = max(skull_region, key=skull_region.get)
    peak_val = skull_region[peak_d]

    # Find where intensity drops ≥10 units after the peak = inner skull edge
    strip_d = peak_d + 2  # safe default
    for d in sorted(means.keys()):
        if d > peak_d and means[d] < peak_val - 10:
            strip_d = d
            break

    # Pull back 2px so we never clip the outer cortex
    strip_d = max(strip_d - 2, peak_d)

    # 4. Threshold distance map at skull depth
    _, brain_mask = cv2.threshold(dist, strip_d, 255, cv2.THRESH_BINARY)
    brain_mask = brain_mask.astype(np.uint8)

    # 5. Close gaps (ventricles & sulci are dark — don't exclude them)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, k_close, iterations=3)

    # 6. Fill all enclosed holes robustly
    brain_mask = ndimage.binary_fill_holes(brain_mask > 0).astype(np.uint8) * 255

    # 7. Keep only the largest connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain_mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(brain_mask)
        clean[labels == largest] = 255
        brain_mask = clean

    # 8. Clip to original head boundary
    brain_mask = cv2.bitwise_and(brain_mask, head_mask)

    # 9. Smooth edges
    brain_mask = cv2.GaussianBlur(brain_mask, (5, 5), 0)
    _, brain_mask = cv2.threshold(brain_mask, 127, 255, cv2.THRESH_BINARY)

    return brain_mask


for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(input_folder, filename), 0)
    if img is None:
        print(f"❌ Failed: {filename}")
        continue

    try:
        mask = skull_strip(img)
        if mask is None:
            print(f"⚠️  No contour: {filename}")
            continue

        result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join(mask_folder, filename), mask)
        cv2.imwrite(os.path.join(stripped_folder, filename), result)
        print(f"✅ {filename}")

    except Exception as e:
        print(f"❌ {filename}: {e}")

print("🎉 Done!")