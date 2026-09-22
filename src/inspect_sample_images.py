# ==========================================
# BODYM SAMPLE IMAGE ANALYSIS
# ==========================================

import cv2
import os
import numpy as np


# ==========================================
# 1. IMAGE FOLDER
# ==========================================

image_folder = "data/raw/sample_masks"


# ==========================================
# 2. GET IMAGE FILES
# ==========================================

image_files = [
    file
    for file in os.listdir(image_folder)
    if file.endswith(".png")
]


print("Number of sample images:", len(image_files))


# ==========================================
# 3. ANALYZE EACH IMAGE
# ==========================================

for file in image_files:

    image_path = os.path.join(
        image_folder,
        file
    )

    # Load image in grayscale
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    print("\n------------------------------------------")
    print("Image:", file)

    # Check image loading
    if image is None:

        print("❌ Could not load image")

        continue

    print("✅ Image loaded")

    # Image dimensions
    height, width = image.shape

    print("Width :", width)
    print("Height:", height)

    # Find unique pixel values
    unique_values = np.unique(image)

    print("Unique pixel values:")
    print(unique_values)

    # Count white/body pixels
    white_pixels = np.sum(image > 0)

    print("Body pixels:", white_pixels)

    # Calculate body area percentage
    total_pixels = image.shape[0] * image.shape[1]

    body_percentage = (
        white_pixels / total_pixels
    ) * 100

    print(
        "Body area percentage:",
        round(body_percentage, 2),
        "%"
    )


# ==========================================
# 4. FINISH
# ==========================================

print("\n==========================================")
print("SAMPLE IMAGE ANALYSIS COMPLETED")
print("==========================================")