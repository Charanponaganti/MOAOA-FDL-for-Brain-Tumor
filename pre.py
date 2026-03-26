import cv2
import numpy as np
import os

# 📁 Folders
input_folder = "archive/no"
mask_folder = "archive/mask_no"
stripped_folder = "archive/strip_no"

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(stripped_folder, exist_ok=True)

# 🔁 Loop through images
for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, 0)

    if img is None:
        print(f"❌ Failed to load: {filename}")
        continue

    try:
        # 1️⃣ CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)

        # 2️⃣ Threshold (Otsu)
        _, thresh = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3️⃣ Morphological cleaning
        kernel_small = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_small)

        # 4️⃣ Largest component (initial brain region)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing)

        if len(stats) <= 1:
            print(f"⚠️ Skipping (no components): {filename}")
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        brain_mask = np.zeros_like(closing)
        brain_mask[labels == largest_label] = 255

        # 5️⃣ Invert mask
        brain_mask = 255 - brain_mask

        # 🔥 6️⃣ Distance Transform (adaptive)
        dist = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)

        thresh_val = 0.3 * dist.max()   # adaptive threshold
        _, brain_mask = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
        brain_mask = brain_mask.astype("uint8")

        # 🛑 Fallback if mask too small
        if np.sum(brain_mask) < 5000:
            print(f"⚠️ Weak mask, fallback used: {filename}")
            brain_mask = closing

        # 7️⃣ Smooth edges
        brain_mask = cv2.GaussianBlur(brain_mask, (5,5), 0)
        _, brain_mask = cv2.threshold(brain_mask, 127, 255, cv2.THRESH_BINARY)

        # 8️⃣ Largest component again (final cleanup)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain_mask)

        if len(stats) <= 1:
            print(f"⚠️ Skipping after refinement: {filename}")
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        clean_mask = np.zeros_like(brain_mask)
        clean_mask[labels == largest_label] = 255

        # 9️⃣ Apply mask
        result = cv2.bitwise_and(img, img, mask=clean_mask)

        # 🔟 Save outputs
        mask_path = os.path.join(mask_folder, filename)
        stripped_path = os.path.join(stripped_folder, filename)

        cv2.imwrite(mask_path, clean_mask)
        cv2.imwrite(stripped_path, result)

        print(f"✅ Processed: {filename}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("🎉 All images processed successfully!")