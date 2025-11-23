import os
import cv2
import numpy as np
from tqdm import tqdm
from random import sample

# ------------------------------
# Configuration
# ------------------------------
dataset_path = "D:/Reside-6K"
sets = ["train", "test"]
folders = ["hazy", "GT"]
NUM_IMAGES = 30

# ------------------------------
# Create Output Folders
# ------------------------------
for s in sets:
    for f in folders:
        os.makedirs(os.path.join(dataset_path, s, f"{f}_enhanced"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, s, "visualization_before_after"), exist_ok=True)

# ------------------------------
# Helper: Unsharp Mask for Extra Clarity
# ------------------------------
def unsharp_mask(image, strength=1.5):
    """Enhance edges and fine details."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1 + strength, gaussian, -strength, 0)
    return sharpened

# ------------------------------
# Adaptive Enhancement Function
# ------------------------------
def adaptive_enhance(img, mode="hazy"):
    """
    Enhanced adaptive processing:
      - hazy: stronger clarity, CLAHE + sharpening
      - GT: soft normalization
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)

    # ------------------------------
    # For HAZY IMAGES
    # ------------------------------
    if mode == "hazy":
        if mean_brightness < 80:
            alpha, beta = 1.9, 28
        elif mean_brightness < 130:
            alpha, beta = 1.6, 18
        else:
            alpha, beta = 1.3, 8

        enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Apply CLAHE to boost local contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        # Add unsharp masking for crispness
        enhanced = unsharp_mask(enhanced, strength=1.2)

    # ------------------------------
    # For GROUND TRUTH IMAGES
    # ------------------------------
    else:
        if mean_brightness < 80:
            alpha, beta = 1.2, 8
        elif mean_brightness < 130:
            alpha, beta = 1.1, 5
        else:
            alpha, beta = 1.05, 0

        enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return enhanced, alpha, beta, mean_brightness, contrast

# ------------------------------
# Process Limited Images
# ------------------------------
def process_and_save(input_dir, output_dir, limit=30, mode="hazy"):
    all_images = [img for img in os.listdir(input_dir)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not all_images:
        print(f"[WARN] No images found in {input_dir}")
        return []

    selected_images = sample(all_images, min(limit, len(all_images)))
    print(f"[INFO] Selected {len(selected_images)} images from {input_dir}")

    processed = []

    for img_name in tqdm(selected_images, desc=f"Enhancing {os.path.basename(input_dir)}"):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced, alpha, beta, mean_brightness, contrast = adaptive_enhance(img, mode)

        # Save enhanced version
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, img_name), enhanced_bgr)
        processed.append(img_name)

        # Log enhancement summary
        print(f"  {img_name}: mean={mean_brightness:.1f}, contrast={contrast:.1f}, alpha={alpha}, beta={beta}")

    return processed

# ------------------------------
# Visualization (Before | After)
# ------------------------------
def save_visualization(original_dir, enhanced_dir, vis_dir, processed_list):
    for img_name in processed_list:
        orig_path = os.path.join(original_dir, img_name)
        enh_path = os.path.join(enhanced_dir, img_name)

        if not os.path.exists(enh_path):
            continue

        orig = cv2.imread(orig_path)
        enh = cv2.imread(enh_path)
        if orig is None or enh is None:
            continue

        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        enh = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)

        combined = np.hstack((orig, enh))
        save_path = os.path.join(vis_dir, f"before_after_{img_name}")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

# ------------------------------
# Run for All Folders
# ------------------------------
print("=== Starting Optimized Clarity Enhancement for Hazy and GT Images ===")

for s in sets:
    for f in folders:
        mode = "hazy" if f.lower() == "hazy" else "GT"
        print(f"\n[PROCESS] {s}/{f} — Mode: {mode}, {NUM_IMAGES} images...")

        input_dir = os.path.join(dataset_path, s, f)
        output_dir = os.path.join(dataset_path, s, f"{f}_enhanced")
        vis_dir = os.path.join(dataset_path, s, "visualization_before_after")

        processed = process_and_save(input_dir, output_dir, limit=NUM_IMAGES, mode=mode)
        save_visualization(input_dir, output_dir, vis_dir, processed)

print("\n[INFO] Enhancement complete! All results saved successfully.")
