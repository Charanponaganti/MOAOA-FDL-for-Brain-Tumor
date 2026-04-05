import cv2
import numpy as np
import os


def flip_horizontal(img):
    """Flip image left-right."""
    return cv2.flip(img, 1)


def flip_vertical(img):
    """Flip image top-bottom."""
    return cv2.flip(img, 0)


def rotate_90(img):
    """Rotate image 90 degrees clockwise."""
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate_180(img):
    """Rotate image 180 degrees."""
    return cv2.rotate(img, cv2.ROTATE_180)


# Maps suffix label -> transform function
AUGMENTATIONS = {
    "flip_h":  flip_horizontal,
    "flip_v":  flip_vertical,
    "rot90":   rotate_90,
    "rot180":  rotate_180,
}


def augment_image(img):
    """
    Apply all four augmentations to a single image.

    Parameters
    ----------
    img : np.ndarray
        Preprocessed image (H x W, uint8 or float32).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of augmentation name -> augmented image.
        Does NOT include the original; caller decides whether to keep it.
    """
    return {name: fn(img) for name, fn in AUGMENTATIONS.items()}


def augment_dataset(input_dir, output_dir, labels=("yes", "no")):
    """
    Walk input_dir/<label>/ for every label, apply all four augmentations
    to each image, and write results to output_dir/<label>/.

    Original images are copied across unchanged so the output folder is
    self-contained.

    Parameters
    ----------
    input_dir  : str   Root of the processed dataset, e.g. "data/processed"
    output_dir : str   Destination root,              e.g. "data/augmented"
    labels     : tuple Sub-folder names to process.
    """
    for label in labels:
        in_path  = os.path.join(input_dir,  label)
        out_path = os.path.join(output_dir, label)
        os.makedirs(out_path, exist_ok=True)

        files = [f for f in os.listdir(in_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        if not files:
            print(f"[augment] No images found in {in_path}, skipping.")
            continue

        for fname in files:
            img_path = os.path.join(in_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"[augment] Could not read {img_path}, skipping.")
                continue

            stem, ext = os.path.splitext(fname)

            # --- original ------------------------------------------------
            cv2.imwrite(os.path.join(out_path, fname), img)

            # --- augmented variants --------------------------------------
            for aug_name, aug_img in augment_image(img).items():
                out_name = f"{stem}_{aug_name}{ext}"
                cv2.imwrite(os.path.join(out_path, out_name), aug_img)

        total = len(files) * (1 + len(AUGMENTATIONS))
        print(f"[augment] {label}: {len(files)} originals -> {total} total images")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    INPUT_DIR  = "data/processed"
    OUTPUT_DIR = "data/augmented"

    augment_dataset(INPUT_DIR, OUTPUT_DIR, labels=("yes", "no"))
    print("Augmentation complete.")