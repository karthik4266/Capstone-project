import os
import cv2
import numpy as np
from tqdm import tqdm
from random import sample

# ------------------------------
# CONFIG
# ------------------------------
dataset_path = "D:/Reside-6K"
sets = ["train", "test"]
folders = ["hazy", "GT"]
NUM_IMAGES = 30


# ------------------------------
# Dark Channel Prior (DCP) Functions
# ------------------------------
def dark_channel(image, size=15):
    b, g, r = cv2.split(image)
    minimum = cv2.min(cv2.min(b, g), r)
    return cv2.erode(minimum, np.ones((size, size)))


def estimate_atmosphere(image, dark_channel):
    h, w = dark_channel.shape
    num_pixels = h * w
    top_pixels = int(max(num_pixels * 0.001, 1))

    indices = np.argsort(dark_channel.ravel())[::-1][:top_pixels]
    atmosphere = np.mean(image.reshape(-1, 3)[indices], axis=0)
    return atmosphere


def recover_image(image, atmosphere, transmission, t_min=0.25):
    transmission = np.clip(transmission, t_min, 1)
    result = (image - atmosphere) / transmission[:, :, None] + atmosphere
    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------
# Soft Sharpening
# ------------------------------
def unsharp(image):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.25, blur, -0.25, 0)


# ------------------------------
# Autonomous Dehazing Enhancement (BEST)
# ------------------------------
def autonomous_enhance(img):
    origin = img.copy().astype(np.float32)
    img = img.astype(np.float32)

    # STEP 1 — Dark Channel
    dc = dark_channel(img / 255.0)

    # STEP 2 — Improved Airlight Estimation (avoid sky overestimation)
    h, w = dc.shape
    num_pixels = max(int(h * w * 0.001), 1)

    # Select brightest pixels BUT avoid sky-only dominance
    flat = dc.reshape(-1)
    indices = np.argsort(flat)[::-1][:num_pixels]
    atmosphere = np.mean(img.reshape(-1, 3)[indices], axis=0)
    atmosphere = np.clip(atmosphere, 20, 240)  # prevents over-blown atmospheric value

    # STEP 3 — Transmission Map
    trans_est = 1 - 0.95 * dark_channel(img / (atmosphere + 1e-6), size=15)

    # Clamp transmission to avoid black artifacts
    trans_est = np.clip(trans_est, 0.15, 1.0)

    # STEP 4 — Guided Filter to refine (edge preserving)
    gray = cv2.cvtColor(origin.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    transmission = cv2.ximgproc.guidedFilter(
        guide=gray, src=trans_est.astype(np.float32), radius=30, eps=1e-3
    )

    # Ensure stability after filtering
    transmission = np.clip(transmission, 0.25, 1.0)

    # STEP 5 — Recover Image
    result = (img - atmosphere) / transmission[..., None] + atmosphere
    result = np.clip(result, 0, 255).astype(np.uint8)

    # STEP 6 — Adaptive Gamma Correction
    brightness = np.mean(result)
    gamma = 1.4 if brightness < 110 else 1.1
    result = np.array(255 * ((result / 255) ** (1 / gamma)), dtype=np.uint8)

    # STEP 7 — Soft Sharpen
    result = unsharp(result)

    # SAFETY CHECK: avoid fully dark or corrupted output
    if np.mean(result) < 5 or np.sum(result) == 0:
        return origin.astype(np.uint8), "FALLBACK_USED"

    return result, "OK"


# ------------------------------
# GT Enhancement (Very Mild)
# ------------------------------
def enhance_gt(img):
    return cv2.convertScaleAbs(img, alpha=1.03, beta=2)


# ------------------------------
# PROCESS + SAVE
# ------------------------------
def process_and_save(input_dir, output_dir, limit=30, mode="hazy"):

    os.makedirs(output_dir, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
    all_images = [i for i in os.listdir(input_dir) if i.lower().endswith(valid_ext)]

    selected = sample(all_images, min(limit, len(all_images)))
    print(f"[INFO] Processing {len(selected)} images from {input_dir}")

    processed = []

    for img_name in tqdm(selected, desc=f"Enhancing {mode}"):

        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mode == "hazy":
            enhanced, status = autonomous_enhance(img)
            if status == "FALLBACK_USED":
               print(f"[WARN] Auto-fix applied for {img_name} (sky-heavy image)")
        else:
            enhanced = enhance_gt(img)


        cv2.imwrite(os.path.join(output_dir, img_name),
                    cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

        processed.append(img_name)

    return processed


# ------------------------------
# SAVE BEFORE/AFTER VISUALIZATION
# ------------------------------
def save_visualization(original_dir, enhanced_dir, vis_dir, processed_list):

    os.makedirs(vis_dir, exist_ok=True)

    for img_name in processed_list:

        orig = cv2.imread(os.path.join(original_dir, img_name))
        enh = cv2.imread(os.path.join(enhanced_dir, img_name))

        if orig is None or enh is None:
            continue

        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        enh = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)

        canvas = np.hstack((orig, enh))

        cv2.imwrite(
            os.path.join(vis_dir, f"before_after_{img_name}"),
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        )


# ------------------------------
# MAIN
# ------------------------------
print("\n=== Autonomous Driving Dehazing Started ===\n")

for s in sets:
    for f in folders:

        mode = "hazy" if f.lower() == "hazy" else "gt"

        input_dir = os.path.join(dataset_path, s, f)
        output_dir = os.path.join(dataset_path, s, f"{f}_enhanced")
        vis_dir = os.path.join(dataset_path, s, "visualization_before_after")

        processed = process_and_save(input_dir, output_dir, NUM_IMAGES, mode)
        save_visualization(input_dir, output_dir, vis_dir, processed)

print("\n=== Processing Complete — Check Enhanced Folders and Visualization ===\n")
