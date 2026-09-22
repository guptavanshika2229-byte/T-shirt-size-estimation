# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# FULL SILHOUETTE FEATURE EXTRACTION
# ==========================================

import cv2
import os
import numpy as np
import pandas as pd


# ==========================================
# 1. SETTINGS
# ==========================================

image_folder = "data/raw/train_masks"

output_file = (
    "data/processed/all_silhouette_features.csv"
)


# ==========================================
# 2. FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(image_path):

    # Load image as grayscale
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None


    # Convert to binary mask
    _, binary = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )


    # Find body pixels
    body_pixels = np.column_stack(
        np.where(binary > 0)
    )

    if len(body_pixels) == 0:
        return None


    # ======================================
    # BODY BOUNDING BOX
    # ======================================

    y_coordinates = body_pixels[:, 0]
    x_coordinates = body_pixels[:, 1]

    min_y = y_coordinates.min()
    max_y = y_coordinates.max()

    min_x = x_coordinates.min()
    max_x = x_coordinates.max()

    body_height = max_y - min_y + 1
    body_width = max_x - min_x + 1


    # ======================================
    # BODY AREA
    # ======================================

    body_area = len(body_pixels)

    total_area = (
        binary.shape[0] *
        binary.shape[1]
    )

    body_area_ratio = (
        body_area / total_area
    )


    # ======================================
    # HEIGHT / WIDTH RATIO
    # ======================================

    height_width_ratio = (
        body_height / body_width
    )


    # ======================================
    # WIDTH AT DIFFERENT BODY LEVELS
    # ======================================

    def width_at_height(relative_position):

        y = int(
            min_y +
            relative_position * body_height
        )

        # Keep y inside image
        y = min(
            max(y, 0),
            binary.shape[0] - 1
        )

        row = np.where(
            binary[y] > 0
        )[0]

        if len(row) == 0:
            return 0

        return (
            row.max() -
            row.min() +
            1
        )


    shoulder_width = width_at_height(0.25)

    chest_width = width_at_height(0.35)

    waist_width = width_at_height(0.50)

    hip_width = width_at_height(0.65)

    thigh_width = width_at_height(0.75)


    # ======================================
    # NORMALIZED WIDTH FEATURES
    # ======================================

    shoulder_width_ratio = (
        shoulder_width /
        body_height
    )

    chest_width_ratio = (
        chest_width /
        body_height
    )

    waist_width_ratio = (
        waist_width /
        body_height
    )

    hip_width_ratio = (
        hip_width /
        body_height
    )

    thigh_width_ratio = (
        thigh_width /
        body_height
    )


    # ======================================
    # RETURN FEATURES
    # ======================================

    return {

        "body_height_pixels":
            body_height,

        "body_width_pixels":
            body_width,

        "body_area":
            body_area,

        "body_area_ratio":
            body_area_ratio,

        "height_width_ratio":
            height_width_ratio,

        "shoulder_width":
            shoulder_width,

        "chest_width":
            chest_width,

        "waist_width":
            waist_width,

        "hip_width":
            hip_width,

        "thigh_width":
            thigh_width,

        "shoulder_width_ratio":
            shoulder_width_ratio,

        "chest_width_ratio":
            chest_width_ratio,

        "waist_width_ratio":
            waist_width_ratio,

        "hip_width_ratio":
            hip_width_ratio,

        "thigh_width_ratio":
            thigh_width_ratio
    }


# ==========================================
# 3. GET ALL PNG FILES
# ==========================================

print("Searching for BodyM training images...")

image_files = [
    file
    for file in os.listdir(image_folder)
    if file.lower().endswith(".png")
]

image_files.sort()

print(
    "Total PNG images found:",
    len(image_files)
)


# ==========================================
# 4. PROCESS ALL IMAGES
# ==========================================

feature_rows = []

total_images = len(image_files)

print("\nStarting feature extraction...")
print()


for index, file in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        image_folder,
        file
    )


    features = extract_features(
        image_path
    )


    if features is not None:

        # Remove .png to get photo_id
        photo_id = os.path.splitext(
            file
        )[0]

        features["photo_id"] = photo_id

        feature_rows.append(
            features
        )


    # Display progress every 100 images
    if index % 100 == 0:

        print(
            f"Processed: {index} / {total_images}"
        )


# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

features_df = pd.DataFrame(
    feature_rows
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("FEATURE EXTRACTION COMPLETED")
print("==========================================")

print(
    "Images processed:",
    total_images
)

print(
    "Feature rows created:",
    len(features_df)
)

print(
    "Number of columns:",
    len(features_df.columns)
)


# ==========================================
# 7. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")

print(
    features_df.isnull().sum()
)


# ==========================================
# 8. SAVE FEATURES
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

features_df.to_csv(
    output_file,
    index=False
)


print("\nFeatures saved successfully!")

print(
    "Saved at:",
    output_file
)


# ==========================================
# 9. DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows:")

print(
    features_df.head()
)


# ==========================================
# 10. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("FULL SILHOUETTE FEATURE EXTRACTION DONE")
print("==========================================")