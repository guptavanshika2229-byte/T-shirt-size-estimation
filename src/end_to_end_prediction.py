# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# END-TO-END IMAGE PREDICTION
# ==========================================

import cv2
import numpy as np
import pandas as pd
import joblib


# ==========================================
# 1. SELECT INPUT IMAGE
# ==========================================

image_path = (
    "data/raw/sample_masks/"
    "e6f404ebda41ebe93573d3e219c88297.png"
)


# ==========================================
# 2. LOAD IMAGE
# ==========================================

print("Loading silhouette image...")

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)


if image is None:

    print("❌ Could not load image")
    exit()


print("✅ Image loaded successfully")


# ==========================================
# 3. CONVERT TO BINARY MASK
# ==========================================

_, binary = cv2.threshold(
    image,
    127,
    255,
    cv2.THRESH_BINARY
)


# ==========================================
# 4. FIND BODY PIXELS
# ==========================================

body_pixels = np.column_stack(
    np.where(binary > 0)
)


if len(body_pixels) == 0:

    print("❌ No body detected")
    exit()


y_coordinates = body_pixels[:, 0]
x_coordinates = body_pixels[:, 1]


# ==========================================
# 5. BODY BOUNDING BOX
# ==========================================

min_y = y_coordinates.min()
max_y = y_coordinates.max()

min_x = x_coordinates.min()
max_x = x_coordinates.max()


body_height = (
    max_y - min_y + 1
)

body_width = (
    max_x - min_x + 1
)


# ==========================================
# 6. BODY AREA
# ==========================================

body_area = len(body_pixels)

total_area = (
    binary.shape[0] *
    binary.shape[1]
)

body_area_ratio = (
    body_area / total_area
)


# ==========================================
# 7. HEIGHT / WIDTH RATIO
# ==========================================

height_width_ratio = (
    body_height / body_width
)


# ==========================================
# 8. WIDTH AT DIFFERENT BODY LEVELS
# ==========================================

def width_at_height(relative_position):

    y = int(
        min_y +
        relative_position * body_height
    )

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


# ==========================================
# 9. NORMALIZED WIDTH FEATURES
# ==========================================

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


# ==========================================
# 10. CREATE IMAGE FEATURE DATAFRAME
# ==========================================

feature_data = {

    "body_height_pixels": body_height,

    "body_width_pixels": body_width,

    "body_area": body_area,

    "body_area_ratio": body_area_ratio,

    "height_width_ratio": height_width_ratio,

    "shoulder_width": shoulder_width,

    "chest_width": chest_width,

    "waist_width": waist_width,

    "hip_width": hip_width,

    "thigh_width": thigh_width,

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


feature_df = pd.DataFrame(
    [feature_data]
)


# ==========================================
# 11. LOAD BODY MEASUREMENT MODEL
# ==========================================

print("\nLoading body measurement model...")

measurement_model = joblib.load(
    "models/body_measurement_model.pkl"
)

measurement_features = joblib.load(
    "models/body_measurement_features.pkl"
)

print(
    "✅ Body measurement model loaded"
)


# ==========================================
# 12. PREPARE IMAGE FEATURES
# ==========================================

X_measurement = feature_df[
    measurement_features
]


# ==========================================
# 13. PREDICT BODY MEASUREMENTS
# ==========================================

print("\nPredicting body measurements...")

predicted_measurements = (
    measurement_model.predict(
        X_measurement
    )
)


measurement_names = [

    "ankle",
    "arm-length",
    "bicep",
    "calf",
    "chest",
    "forearm",
    "height",
    "hip",
    "leg-length",
    "shoulder-breadth",
    "shoulder-to-crotch",
    "thigh",
    "waist",
    "wrist"
]


measurement_df = pd.DataFrame(
    predicted_measurements,
    columns=measurement_names
)


print("\n==========================================")
print("ESTIMATED BODY MEASUREMENTS")
print("==========================================")


for measurement in measurement_names:

    value = measurement_df[
        measurement
    ].iloc[0]

    print(
        f"{measurement:20s}: "
        f"{value:.2f} cm"
    )


# ==========================================
# 14. LOAD T-SHIRT SIZE MODEL
# ==========================================

print("\nLoading T-shirt size model...")

size_model = joblib.load(
    "models/tshirt_size_model.pkl"
)

size_features = joblib.load(
    "models/features.pkl"
)

size_label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

print(
    "✅ T-shirt size model loaded"
)


# ==========================================
# 15. PREPARE RAW MEASUREMENTS
# ==========================================

X_size = measurement_df[
    size_features
]


# ==========================================
# 16. PREDICT T-SHIRT SIZE
# ==========================================
#
# IMPORTANT:
#
# The Random Forest model was trained
# using RAW measurements.
#
# Therefore we DO NOT use StandardScaler
# before Random Forest prediction.
#
# ==========================================

prediction = size_model.predict(
    X_size
)


predicted_size = (
    size_label_encoder.inverse_transform(
        prediction
    )
)


# ==========================================
# 17. DISPLAY FINAL RESULT
# ==========================================

print("\n==========================================")
print("FINAL T-SHIRT SIZE PREDICTION")
print("==========================================")


print(
    "\nPredicted T-shirt size:",
    predicted_size[0]
)


print("\n==========================================")
print("END-TO-END PREDICTION COMPLETED")
print("==========================================")