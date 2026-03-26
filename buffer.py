import cv2
import numpy as np
import os
from scipy import ndimage

input_folder   = "archive/yes"
mask_folder    = "archive/mask_yes"
stripped_folder = "archive/strip_yes"

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(stripped_folder, exist_ok=True)


def skull_strip(img):
    # ---------- 1. HEAD MASK ----------
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    head_mask = np.zeros_like(img)
    cv2.drawContours(
        head_mask,
        [max(contours, key=cv2.contourArea)],
        -1,
        255,
        -1
    )

    # Fill holes
    head_mask = ndimage.binary_fill_holes(head_mask > 0).astype(np.uint8) * 255

    # ---------- 2. DISTANCE TRANSFORM ----------
    dist = cv2.distanceTransform(head_mask, cv2.DIST_L2, 5)

    # Normalize distance
    dist_norm = dist / (dist.max() + 1e-5)

    # ---------- 3. SAFE BRAIN CORE ----------
    # Keep only deep region (remove skull zone)
    core = (dist_norm > 0.25).astype(np.uint8) * 255

    # ---------- 4. EXPAND BACK (CONTROLLED GROWTH) ----------
    # Grow region but avoid skull
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.dilate(core, kernel, iterations=3)

    # ---------- 5. INTERSECT WITH HEAD ----------
    brain_mask = cv2.bitwise_and(brain_mask, head_mask)

    # ---------- 6. REMOVE THIN SKULL REMNANTS ----------
    # Erode slightly to remove bright boundary skull
    brain_mask = cv2.erode(brain_mask, kernel, iterations=1)

    # ---------- 7. FILL HOLES (VERY IMPORTANT) ----------
    brain_mask = ndimage.binary_fill_holes(brain_mask > 0).astype(np.uint8) * 255

    # ---------- 8. KEEP LARGEST COMPONENT ----------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain_mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(brain_mask)
        clean[labels == largest] = 255
        brain_mask = clean

    # ---------- 9. FINAL SMOOTH ----------
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
            print(f"⚠️ No contour: {filename}")
            continue

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite(os.path.join(mask_folder, filename), mask)
        cv2.imwrite(os.path.join(stripped_folder, filename), result)

        print(f"✅ {filename}")

    except Exception as e:
        print(f"❌ {filename}: {e}")

print("🎉 Done!")