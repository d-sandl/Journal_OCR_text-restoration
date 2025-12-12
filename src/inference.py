# test_restore.py - Quick visual check on any test image
# how to use:
# python inference.py pix2pix_epoch_190_1st-train.pth
# Example: python inference.py pix2pix_epoch_190_1st-train.pth
import torch
from torchvision import transforms
from PIL import Image
import os
import shutil
import sys
from datetime import datetime
from train import UNetGenerator, Config

# ------------------ SETTINGS ------------------
config = Config()
device = config.device

# naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = sys.argv[1]
model_name_without_ext, _ = os.path.splitext(model_name)

# Load your best model (change the path if needed)
MODEL_PATH = f"../checkpoints/{model_name}"
INPUT_DIR = f"../data/test/test_input/"    # ← put any broken test image here
OUTPUT_DIR = f"../data/test/test_output/{timestamp}_{model_name_without_ext}/"
AFTER_TEST_DIR = f"../data/test/after_test_input/"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AFTER_TEST_DIR, exist_ok=True)

# ------------------ LOAD MODEL ------------------
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device)['generator'])
generator.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------ RESTORE FULL PAGE WITH SEAMLESS BLENDING ------------------
def restore_and_compare(broken_path, output_path):
    img = Image.open(broken_path).convert('RGB')
    w, h = img.size

    # Simple sliding window with overlap blending
    patch_size = 256
    step = 192  # overlap = 64
    restored_patches = []

    print(f"Restoring {os.path.basename(broken_path)}...")
    for y in range(0, max(h - patch_size + 1, 1), step):
        for x in range(0, max(w - patch_size + 1, 1), step):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)

            # Pad if near edge
            pad_r = patch_size - patch.width
            pad_b = patch_size - patch.height
            if pad_r > 0 or pad_b > 0:
                patch = transforms.Pad((0, 0, pad_r, pad_b))(patch)

            tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                out = generator(tensor)
            out = out.squeeze(0).cpu() * 0.5 + 0.5  # denormalize

            # Crop back padding
            if pad_r > 0 or pad_b > 0:
                out = out[:, :patch_size-pad_b, :patch_size-pad_r]

            restored_patches.append((out, x, y, patch.width, patch.height))

    # Create canvas
    canvas = img.copy()
    canvas = canvas.convert('RGB')
    draw = canvas.load()

    for patch_tensor, x, y, pw, ph in restored_patches:
        patch_np = (patch_tensor.permute(1,2,0).numpy() * 255).astype('uint8')
        patch_img = Image.fromarray(patch_np)
        canvas.paste(patch_img, (x, y))

    # Save side-by-side comparison
    comparison = Image.new('RGB', (w*2 + 50, h), (255, 255, 255))
    comparison.paste(img, (0, 0))
    comparison.paste(canvas, (w + 50, 0))

    # Add text
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(comparison)
    d.text((50, 20), "BROKEN", fill=(255,0,0), font=font)
    d.text((w + 100, 20), "RESTORED", fill=(0,150,0), font=font)

    comparison.save(output_path)
    print(f"Comparison saved → {output_path}")


# ------------------ RUN ON ONE OR MANY IMAGES ------------------
if __name__ == "__main__":
    # # Restore a single image
    # restore_and_compare(TEST_IMAGE, "out_" + TEST_IMAGE)

    # Or restore all broken images in a folder
    for img in os.listdir(INPUT_DIR)[:10]:
        if img.endswith(('.jpg','.png', '.jpeg')):
            src_path = os.path.join(INPUT_DIR, img)
            out_path = os.path.join(OUTPUT_DIR, img)

            # restore
            restore_and_compare(src_path, out_path)

            # move to after_test_input
            dst_path = os.path.join(AFTER_TEST_DIR, img)
            shutil.move(src_path, dst_path)
            print(f"Moved {img} → {AFTER_TEST_DIR}")