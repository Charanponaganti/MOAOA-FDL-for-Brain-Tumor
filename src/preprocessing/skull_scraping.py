import cv2
import numpy as np
from scipy import ndimage

def skull_strip(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    head_mask = np.zeros_like(img)
    cv2.drawContours(head_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    dist = cv2.distanceTransform(head_mask, cv2.DIST_L2, 5)

    _, brain_mask = cv2.threshold(dist, 10, 255, cv2.THRESH_BINARY)
    brain_mask = brain_mask.astype(np.uint8)

    brain_mask = ndimage.binary_fill_holes(brain_mask > 0).astype(np.uint8) * 255

    return cv2.bitwise_and(img, img, mask=brain_mask)