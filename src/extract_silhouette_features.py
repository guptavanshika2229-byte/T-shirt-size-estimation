# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# SILHOUETTE FEATURE EXTRACTION
# ==========================================

import cv2
import os
import numpy as np
import pandas as pd


# ==========================================
# 1. IMAGE FOLDER
# ==========================================

image_folder = "data/raw/sample_masks"


# ==========================================
# 2. FUNCTION TO EXTRACT FEATURES
# ==========================================

def extract_features(image_path):

    # Load image as grayscale
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None


    # --------------------------------------
    # Convert image into binary mask
    # --------------------------------------

    _, binary = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------
    # Find body pixels
    # --------------------------------------

    body_pixels = np.column_stack(
        np.where(binary > 0)
    )


    if len(body_pixels) == 0:
        return None


    # --------------------------------------
    # Body bounding box
    # --------------------------------------

    y_coordinates = body_pixels[:, 0]
    x_coordinates = body_pixels[:, 1]


    min_y = y_coordinates.min()
    max_y = y_coordinates.max()

    min_x = x_coordinates.min()
    max_x = x_coordinates.max()


    body_height = max_y - min_y + 1
    body_width = max_x - min_x + 1


    # --------------------------------------
    # Body area
    # --------------------------------------

    body_area = len(body_pixels)


    # --------------------------------------
    # Body area ratio
    # --------------------------------------

    total_area = binary.shape[0] * binary.shape[1]

    body_area_ratio = body_area / total_area


    # --------------------------------------
    # Height / width ratio
    # --------------------------------------

    height_width_ratio = (
        body_height / body_width
    )


    # --------------------------------------
    # Width at different body levels
    # --------------------------------------

    def width_at_height(relative_position):

        y = int(
            min_y +
            relative_position * body_height
        )

        row = np.where(binary[y] > 0)[0]

        if len(row) == 0:
            return 0

        return row.max() - row.min() + 1


    shoulder_width = width_at_height(0.25)

    chest_width = width_at_height(0.35)

    waist_width = width_at_height(0.50)

    hip_width = width_at_height(0.65)

    thigh_width = width_at_height(0.75)


    # --------------------------------------
    # Width ratios
    # --------------------------------------

    shoulder_width_ratio = (
        shoulder_width / body_height
    )

    chest_width_ratio = (
        chest_width / body_height
    )

    waist_width_ratio = (
        waist_width / body_height
    )

    hip_width_ratio = (
        hip_width / body_height
    )

    thigh_width_ratio = (
        thigh_width / body_height
    )


    # --------------------------------------
    # Return features
    # --------------------------------------

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
# 3. FIND SAMPLE IMAGES
# ==========================================

image_files = [
    file
    for file in os.listdir(image_folder)
    if file.endswith(".png")
]


print(
    "Number of images:",
    len(image_files)
)


# ==========================================
# 4. EXTRACT FEATURES
# ==========================================

feature_rows = []


for file in image_files:

    image_path = os.path.join(
        image_folder,
        file
    )


    features = extract_features(
        image_path
    )


    if features is not None:

        features["photo_id"] = (
            file.replace(".png", "")
        )

        feature_rows.append(
            features
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
print("EXTRACTED SILHOUETTE FEATURES")
print("==========================================")

print(
    features_df
)


print("\nFeature names:")

print(
    features_df.columns.tolist()
)


# ==========================================
# 7. SAVE FEATURES
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


output_path = (
    "data/processed/silhouette_features_sample.csv"
)


features_df.to_csv(
    output_path,
    index=False
)


print("\nFeatures saved successfully!")

print(
    "Saved at:",
    output_path
)


# ==========================================
# 8. FINISH
# ==========================================

print("\n==========================================")
print("SILHOUETTE FEATURE EXTRACTION COMPLETED")
print("==========================================")