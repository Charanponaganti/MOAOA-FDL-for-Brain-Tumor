import cv2
import numpy as np
import os

# ---------------------------------------------------------
# Tumor extraction from segmented image
# ---------------------------------------------------------
def extract_tumor(segmented):
    levels = np.unique(segmented)
    print("Unique levels:", levels)  # should now show exactly 4 values

    if len(levels) < 3:
        return np.zeros_like(segmented)

    # Highest intensity = brightest region = likely tumor in MRI
    target = levels[-1]
    mask = (segmented == target).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    print(f"  Largest contour area: {cv2.contourArea(largest)}")

    if cv2.contourArea(largest) < 100:
        return np.zeros_like(mask)

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean

# ---------------------------------------------------------
# Batch processing
# ---------------------------------------------------------
def extract_dataset(input_dir, output_dir, labels=("yes", "no")):
    """
    input_dir  = segmented images folder
    output_dir = tumor-only images folder
    """

    for label in labels:
        in_path = os.path.join(input_dir, label)
        out_path = os.path.join(output_dir, label)

        os.makedirs(out_path, exist_ok=True)

        files = [f for f in os.listdir(in_path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        print(f"[TUMOR] Processing {label} ({len(files)} images)")

        for i, fname in enumerate(files, 1):
            img_path = os.path.join(in_path, fname)
            segmented = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if segmented is None:
                print(f"Skipping {fname}")
                continue
            print("Unique levels:", np.unique(segmented))

            tumor = extract_tumor(segmented)

            cv2.imwrite(os.path.join(out_path, fname), tumor)

            print(f"[{i}/{len(files)}] {fname}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    test = cv2.imread("data/segmented_multi/yes/some_image.jpg", cv2.IMREAD_GRAYSCALE)
    print("Unique pixel values:", np.unique(test))
    extract_dataset(
        input_dir="data/segmented_multi",
        output_dir="data/tumor_only",
        labels=("yes", "no")
    )
    # Remove the stray print(np.unique(segmented)) line — it's undefined here
    print("Tumor extraction complete ✅")