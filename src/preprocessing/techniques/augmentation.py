import cv2
import numpy as np
import os


def f_h(img):
    # flip left-right
    return cv2.flip(img, 1)


def f_v(img):
    # flip top-bottom
    return cv2.flip(img, 0)


def r90(img):
    # rotate 90
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def r180(img):
    # rotate 180
    return cv2.rotate(img, cv2.ROTATE_180)


# suffix -> function
AUG_MAP = {
    "flip_h":  f_h,
    "flip_v":  f_v,
    "rot90":   r90,
    "rot180":  r180,
}


def do_aug_single(img):
    # apply all 4 to one image
    return {n: fn(img) for n, fn in AUG_MAP.items()}


def run_aug(in_d, out_d, labels=("yes", "no")):
    # loop through dataset and make augmented versions
    for l in labels:
        p1 = os.path.join(in_d, l)
        p2 = os.path.join(out_d, l)
        os.makedirs(p2, exist_ok=True)

        files = [f for f in os.listdir(p1)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        if not files:
            print(f"Nothing found in {p1}")
            continue

        for f in files:
            p = os.path.join(p1, f)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error reading {p}")
                continue

            stem, ext = os.path.splitext(f)

            # save original
            cv2.imwrite(os.path.join(p2, f), img)

            # save variants
            for n, a_img in do_aug_single(img).items():
                out_name = f"{stem}_{n}{ext}"
                cv2.imwrite(os.path.join(p2, out_name), a_img)

        total = len(files) * (1 + len(AUG_MAP))
        print(f"Done {l}: {len(files)} -> {total} images")


if __name__ == "__main__":
    import sys
    # 3 levels up: techniques -> preprocessing -> src -> project root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

    run_aug("data/outputs/processed", "data/outputs/augmented", labels=("yes", "no"))
    print("finished augmentation.")
