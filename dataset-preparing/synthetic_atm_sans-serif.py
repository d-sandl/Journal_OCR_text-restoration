import cv2 as cv
import os
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y%m%d")
    indir = "./good_sans-serif_raw_20251211"   # folder with clean originals
    clean_dir = f"./{timestamp}/clean_{timestamp}"
    broken_dir = f"./{timestamp}/broken_{timestamp}"

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(broken_dir, exist_ok=True)

    # Copy clean images once
    for file in os.listdir(indir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            in_clean = os.path.join(indir, file)
            out_clean = os.path.join(clean_dir, file)
            cv.imwrite(out_clean, cv.imread(in_clean))  # save original
            print(f"Ground truth image saved: {out_clean}")

            # Generate 6 broken variants
            base_name, ext = os.path.splitext(file)
            for r_thr in range(50, 110, 10):  # 50, 60, ..., 100
                out_broken = os.path.join(broken_dir, f"{base_name}_thr{r_thr}{ext}")
                process_image(in_clean, out_broken, r_thr)

def process_image(indir_file, outdir_file, r_thr):
    img_bgr = cv.imread(indir_file)
    ri = img_bgr[:, :, 2]  # red channel

    # thresholding
    ri[ri < r_thr] = 0
    ri[ri >= r_thr] = 255

    cv.imwrite(outdir_file, ri)
    print(f"Broken image saved: {outdir_file}")

main()
