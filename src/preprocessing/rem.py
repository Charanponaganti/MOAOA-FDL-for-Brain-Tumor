import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return img


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def skull_stripping(img):
    # Step 1: Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 2: Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Moderate erosion
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(thresh, kernel_erode, iterations=3)

    # Step 4: Close holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close, iterations=4)

    # Step 5: Remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Step 6: Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opening)
    if num_labels < 2:
        return img, np.ones_like(img) * 255

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = np.zeros_like(img)
    brain_mask[labels == largest_label] = 255

    # Step 7: Flood fill holes
    h, w = brain_mask.shape
    floodfill = brain_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfill, flood_mask, (0, 0), 255)
    inv_floodfill = cv2.bitwise_not(floodfill)
    brain_mask = brain_mask | inv_floodfill

    # Step 8: ---- NEW: Fit ellipse instead of convex hull ----
    # Ellipse perfectly matches oval brain shape
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:  # fitEllipse needs at least 5 points
            ellipse = cv2.fitEllipse(largest_contour)

            # Shrink ellipse axes to cut skull ring
            (cx, cy), (major, minor), angle = ellipse
            scale = 0.88  # shrink to 88% — adjust this if needed
            shrunk_ellipse = ((cx, cy), (major * scale, minor * scale), angle)

            brain_mask = np.zeros_like(brain_mask)
            cv2.ellipse(brain_mask, shrunk_ellipse, 255, thickness=cv2.FILLED)

    # Step 9: Smooth edges
    brain_mask = cv2.medianBlur(brain_mask, 11)

    # Step 10: Apply mask
    stripped = cv2.bitwise_and(img, img, mask=brain_mask)

    return stripped, brain_mask


def resize_normalize(img, size=(224, 224)):
    resized = cv2.resize(img, size)
    normalized = resized / 255.0
    return normalized


def preprocess_image(path):
    # Step 1: Load original grayscale image
    img = load_image(path)

    # Step 2: Skull strip on original (before CLAHE)
    stripped_img, mask = skull_stripping(img)

    # Step 3: Apply CLAHE on skull-stripped image
    clahe_img = apply_clahe(stripped_img)

    # Step 4: Resize and normalize for model input
    final_img = resize_normalize(clahe_img)

    return img, clahe_img, mask, stripped_img, final_img


def show_results(original, clahe, mask, stripped):
    titles = ["Original", "CLAHE Enhanced", "Brain Mask", "Skull Stripped"]
    images = [original, clahe, mask, stripped]

    plt.figure(figsize=(16, 4))
    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(1, 4, i + 1)
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("preprocessing_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    path = "data/Y9.jpg"

    # --- Preprocessing ---
    original, clahe, mask, stripped, final = preprocess_image(path)
    show_results(original, clahe, mask, stripped)

    print(f"Final image shape : {final.shape}")
    print(f"Value range       : [{final.min():.2f}, {final.max():.2f}]")