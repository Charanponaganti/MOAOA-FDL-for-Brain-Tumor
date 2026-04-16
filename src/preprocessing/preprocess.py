import cv2
from .clahe import apply_clahe
from .skull_scraping import skull_strip
import numpy as np

import numpy as np

def crop_to_brain(img):
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # add padding safely
    pad = 5
    y_min = max(y_min - pad, 0)
    x_min = max(x_min - pad, 0)
    y_max = min(y_max + pad, img.shape[0])
    x_max = min(x_max + pad, img.shape[1])

    return img[y_min:y_max, x_min:x_max]
def preprocess(img, size=(224,224)):
    img = cv2.resize(img, size)

    img = apply_clahe(img)
    img = skull_strip(img)

    if img is None:
        return None

    img = crop_to_brain(img)    

    img = cv2.resize(img, size)  # resize again
    img = img / 255.0

    return img