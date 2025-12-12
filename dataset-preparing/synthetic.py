# how to use: python synthetic.py <input_clean_folder>
# Example: python synthetic.py ./raw/good_sans-serif_1st

import cv2 as cv
import os
import shutil
from datetime import datetime
import sys

def main():
    
    timestamp = datetime.now().strftime("%Y%m%d")
    indir = sys.argv[1]   # folder with clean originals
    indir_basename = os.path.basename(indir)
    out_clean_dir = f"./train/{indir_basename}/clean/"
    out_broken_dir = f"./train/{indir_basename}/broken/"

    # # remove existing output directories if they exist
    # print(f"Removing existing output directories:\n{out_clean_dir}\n{out_broken_dir}")
    # shutil.rmtree(out_clean_dir, ignore_errors=True)
    # shutil.rmtree(out_broken_dir, ignore_errors=True)

    # Create output directories
    os.makedirs(out_clean_dir, exist_ok=True)
    os.makedirs(out_broken_dir, exist_ok=True)

    # Ask user before processing ground truth
    ans = input(f"Do you want to do thresholding for ground truth? (y/n): ").strip().lower()
    if ans != "y":
        print(f"Skipped thresholding for ground truth")

    # Generate ground truth image with single threshold and broken images with different thresholds
    for file in os.listdir(indir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_basename, ext = os.path.splitext(file)

            in_clean_file = os.path.join(indir, file)
            out_clean_file = os.path.join(out_clean_dir, f"{indir_basename}_{img_basename}.jpg")

            if ans == "y":
                process_image(in_clean_file, out_clean_file, r_thr=120)  # cleaned ground truth
                print(f"Ground truth image processed and saved: {out_clean_file}")
            else:
                cv.imwrite(out_clean_file, cv.imread(in_clean_file))
                print(f"Ground truth image copied: {out_clean_file}")
                
            for r_thr in range(50, 110, 10):  # 50, 60, ..., 100
                out_broken_file = os.path.join(out_broken_dir, f"{indir_basename}_{img_basename}_thr{r_thr}.jpg")
                process_image(in_clean_file, out_broken_file, r_thr)
                print(f"Broken image processed and saved: {out_broken_file}")


def process_image(indir_file, outdir_file, r_thr):
    img_bgr = cv.imread(indir_file)
    ri = img_bgr[:, :, 2]  # red channel

    # thresholding
    ri[ri < r_thr] = 0
    ri[ri >= r_thr] = 255

    cv.imwrite(outdir_file, ri)

main()
