import cv2 as cv
import os
import shutil
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y%m%d")
    indir = "./good_sans-serif_raw_20251211"   # folder with clean originals
    out_clean_dir = f"./{timestamp}/clean_{timestamp}"
    out_broken_dir = f"./{timestamp}/broken_{timestamp}"

    # remove existing output directories if they exist
    print(f"Removing existing output directories:\n{out_clean_dir}\n{out_broken_dir}")
    shutil.rmtree(out_clean_dir, ignore_errors=True)
    shutil.rmtree(out_broken_dir, ignore_errors=True)

    # Create output directories
    os.makedirs(out_clean_dir, exist_ok=True)
    os.makedirs(out_broken_dir, exist_ok=True)

    # Generate ground truth image with single threshold and broken images with different thresholds
    for file in os.listdir(indir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            in_clean_file = os.path.join(indir, file)
            out_clean_file = os.path.join(out_clean_dir, file)

            process_image(in_clean_file, out_clean_file, r_thr=120)  # cleaned ground truth
            print(f"Ground truth image saved: {out_clean_file}")

            # Generate 6 broken variants
            base_name, ext = os.path.splitext(file)
            for r_thr in range(50, 110, 10):  # 50, 60, ..., 100
                out_broken_file = os.path.join(out_broken_dir, f"{base_name}_thr{r_thr}{ext}")
                process_image(in_clean_file, out_broken_file, r_thr)
                print(f"Broken image saved: {out_broken_file}")

def process_image(indir_file, outdir_file, r_thr):
    img_bgr = cv.imread(indir_file)
    ri = img_bgr[:, :, 2]  # red channel

    # thresholding
    ri[ri < r_thr] = 0
    ri[ri >= r_thr] = 255

    cv.imwrite(outdir_file, ri)

main()
