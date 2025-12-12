import os
from PIL import Image
import cv2
import numpy as np
from datasets import load_dataset

# Load the dataset
ds = load_dataset("MohamedExperio/ICDAR2019")
dataset = ds["train"]

# Create output directory
output_dir_gt = "./raw/ICDAR2019/"
os.makedirs(output_dir_gt, exist_ok=True)

# Process and save each image
for idx, example in enumerate(dataset):
    # Get the PIL image and convert to NumPy array (OpenCV format)
    pil_img = example["image"]  # Already a PIL.Image in RGB
    np_img = np.array(pil_img)  # Shape: (H, W, 3), dtype uint8
    
    # Save ground truth image
    gt_filename = os.path.join(output_dir_gt, f"gt_image_{idx:04d}.jpg")
    cv2.imwrite(gt_filename, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    print(f"Saved {gt_filename}")


print("Processing complete!")