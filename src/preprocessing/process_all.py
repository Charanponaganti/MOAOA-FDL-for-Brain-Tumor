import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import cv2
from src.preprocessing.preprocess import preprocess

input_dir = "data/brain_tumor_dataset"
output_dir = "data/processed"

os.makedirs(output_dir, exist_ok=True)

for label in ["yes", "no"]:
    in_path = os.path.join(input_dir, label)
    out_path = os.path.join(output_dir, label)
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(in_path):
        img_path = os.path.join(in_path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = preprocess(img)
        if processed is None:
            continue

        # convert back to 0-255
        processed = (processed * 255).astype("uint8")

        save_path = os.path.join(out_path, file)
        cv2.imwrite(save_path, processed)

print("Done processing")