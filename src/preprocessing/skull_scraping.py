import cv2
import numpy as np
from scipy import ndimage


#We used a differnt logic for skull scraping 

def skull_strip(img):
    # make sure its gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur and threshold
    b = cv2.GaussianBlur(img, (5, 5), 0)
    _, t = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morph logic
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=3)

    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img 

    # biggest contour is usually the head
    m1 = np.zeros_like(img)
    cv2.drawContours(m1, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    m1 = ndimage.binary_fill_holes(m1 > 0).astype(np.uint8) * 255

    # distance transform to find center
    d = cv2.distanceTransform(m1, cv2.DIST_L2, 5)
    dn = d / (d.max() + 1e-5)

    # try to isolate brain
    core = (dn > 0.25).astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bm = cv2.dilate(core, k2, iterations=3)

    # clean up the mask
    bm = cv2.bitwise_and(bm, m1)
    bm = cv2.erode(bm, k2, iterations=1)
    bm = ndimage.binary_fill_holes(bm > 0).astype(np.uint8) * 255

    # keep only the biggest blob
    n, l, s, _ = cv2.connectedComponentsWithStats(bm)
    if n > 1:
        big = 1 + np.argmax(s[1:, cv2.CC_STAT_AREA])
        tmp = np.zeros_like(bm)
        tmp[l == big] = 255
        bm = tmp

    # smooth out edges
    bm = cv2.GaussianBlur(bm, (5, 5), 0)
    _, bm = cv2.threshold(bm, 127, 255, cv2.THRESH_BINARY)

    # finally masked image!!
    res = cv2.bitwise_and(img, img, mask=bm)
    return res
