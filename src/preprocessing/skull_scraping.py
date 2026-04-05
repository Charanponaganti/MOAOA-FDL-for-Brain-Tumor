import cv2
import numpy as np
from scipy import ndimage

def skull_strip(img):
    import cv2
    import numpy as np
    from scipy import ndimage

    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        return img  # fallback

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
    dist_norm = dist / (dist.max() + 1e-5)

    # ---------- 3. SAFE CORE ----------
    core = (dist_norm > 0.25).astype(np.uint8) * 255

    # ---------- 4. CONTROLLED EXPANSION ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.dilate(core, kernel, iterations=3)

    # ---------- 5. INTERSECT ----------
    brain_mask = cv2.bitwise_and(brain_mask, head_mask)

    # ---------- 6. REMOVE SKULL EDGE ----------
    brain_mask = cv2.erode(brain_mask, kernel, iterations=1)

    # ---------- 7. FILL HOLES ----------
    brain_mask = ndimage.binary_fill_holes(brain_mask > 0).astype(np.uint8) * 255

    # ---------- 8. LARGEST COMPONENT ----------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain_mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(brain_mask)
        clean[labels == largest] = 255
        brain_mask = clean

    # ---------- 9. SMOOTH ----------
    brain_mask = cv2.GaussianBlur(brain_mask, (5, 5), 0)
    _, brain_mask = cv2.threshold(brain_mask, 127, 255, cv2.THRESH_BINARY)

    # 🔥 FINAL OUTPUT (IMPORTANT)
    result = cv2.bitwise_and(img, img, mask=brain_mask)

    return result