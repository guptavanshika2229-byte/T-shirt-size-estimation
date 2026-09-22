# ==========================================
# BODYM SAMPLE IMAGE DOWNLOAD
# ==========================================

import pandas as pd
import os
import subprocess


# ==========================================
# 1. LOAD COMBINED DATASET
# ==========================================

print("Loading combined dataset...")

df = pd.read_csv(
    "data/processed/image_measurement_dataset.csv"
)

print("Dataset loaded successfully!")


# ==========================================
# 2. SELECT FIRST 5 PHOTO IDs
# ==========================================

sample_photo_ids = df["photo_id"].head(5).tolist()

print("\nSample photo IDs:")

for photo_id in sample_photo_ids:
    print(photo_id)


# ==========================================
# 3. CREATE LOCAL FOLDER
# ==========================================

output_folder = "data/raw/sample_masks"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ==========================================
# 4. DOWNLOAD SAMPLE IMAGES
# ==========================================

print("\nDownloading sample images...")

for photo_id in sample_photo_ids:

    s3_path = (
        f"s3://amazon-bodym/train/mask/{photo_id}.png"
    )

    local_path = (
        f"{output_folder}/{photo_id}.png"
    )

    print("\nDownloading:")
    print(photo_id)

    result = subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            s3_path,
            local_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:

        print("✅ Download successful")

    else:

        print("❌ Download failed")
        print(result.stderr)


# ==========================================
# 5. FINISH
# ==========================================

print("\n==========================================")
print("SAMPLE IMAGE DOWNLOAD COMPLETED")
print("==========================================")

print("\nImages saved in:")
print(output_folder)