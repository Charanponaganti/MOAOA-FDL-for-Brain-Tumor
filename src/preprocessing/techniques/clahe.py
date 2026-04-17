import cv2
import numpy as np

def apply_clahe(img):
    # convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)