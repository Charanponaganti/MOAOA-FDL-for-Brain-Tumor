import os
import cv2
from segmentation import get_tumor_mask
from tqdm import tqdm


# base directory
base_dir = "pre_data"

# output directories
out_yes = "outputs/masked_yes"
out_no = "outputs/masked_no"

os.makedirs(out_yes, exist_ok=True)
os.makedirs(out_no, exist_ok=True)

# classes
classes = ["yes", "no"]

for cls in classes:
    
    image_dir = os.path.join(base_dir, f"strip_{cls}")
    mask_dir = os.path.join(base_dir, f"mask_{cls}")
    
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    
    for img_name, mask_name in tqdm(zip(images, masks), total=len(images)):
        
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        try:
            tumor_mask = get_tumor_mask(img_path, mask_path)
            
            # save in correct folder
            if cls == "yes":
                save_path = os.path.join(out_yes, img_name)
            else:
                save_path = os.path.join(out_no, img_name)
            
            cv2.imwrite(save_path, tumor_mask)
            
            print(f"[{cls}] Done: {img_name}")
        
        except Exception as e:
            print(f"[{cls}] Error: {img_name} → {e}")
            
            
