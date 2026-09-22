# ==========================================
# DEBUG MEASUREMENT -> SIZE PIPELINE
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv"
)


# ==========================================
# 2. CREATE SIZE LABELS
# ==========================================

def assign_tshirt_size(chest):

    if chest < 86:
        return "XS"

    elif chest < 94:
        return "S"

    elif chest < 102:
        return "M"

    elif chest < 110:
        return "L"

    elif chest < 118:
        return "XL"

    else:
        return "XXL"


df["tshirt_size"] = df["chest"].apply(
    assign_tshirt_size
)


# ==========================================
# 3. CREATE SAME TEST SPLIT
# ==========================================

image_features = [

    "body_height_pixels",
    "body_width_pixels",
    "body_area",
    "body_area_ratio",
    "height_width_ratio",
    "shoulder_width",
    "chest_width",
    "waist_width",
    "hip_width",
    "thigh_width",
    "shoulder_width_ratio",
    "chest_width_ratio",
    "waist_width_ratio",
    "hip_width_ratio",
    "thigh_width_ratio"
]


X = df[image_features]

y = df["tshirt_size"]

groups = df["subject_id"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


test_data = df.iloc[test_indices].copy()


# ==========================================
# 4. LOAD MODELS
# ==========================================

print("\nLoading models...")

measurement_model = joblib.load(
    "models/body_measurement_model.pkl"
)

measurement_features = joblib.load(
    "models/body_measurement_features.pkl"
)

size_model = joblib.load(
    "models/tshirt_size_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

size_features = joblib.load(
    "models/features.pkl"
)


print("Models loaded successfully!")


# ==========================================
# 5. PREDICT BODY MEASUREMENTS
# ==========================================

print("\nPredicting measurements...")

X_test = test_data[
    measurement_features
]


predicted_measurements = (
    measurement_model.predict(
        X_test
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


predicted_df = pd.DataFrame(
    predicted_measurements,
    columns=measurement_names
)


# ==========================================
# 6. SIZE PREDICTION USING
#    PREDICTED MEASUREMENTS
# ==========================================

X_predicted_size = predicted_df[
    size_features
]


X_predicted_scaled = scaler.transform(
    X_predicted_size
)


predicted_size_numeric = (
    size_model.predict(
        X_predicted_scaled
    )
)


predicted_sizes = (
    label_encoder.inverse_transform(
        predicted_size_numeric
    )
)


# ==========================================
# 7. SIZE PREDICTION USING
#    ACTUAL MEASUREMENTS
# ==========================================

actual_measurements = test_data[
    size_features
]


actual_scaled = scaler.transform(
    actual_measurements
)


actual_size_numeric = (
    size_model.predict(
        actual_scaled
    )
)


actual_measurement_sizes = (
    label_encoder.inverse_transform(
        actual_size_numeric
    )
)


# ==========================================
# 8. ACTUAL CHEST-BASED LABELS
# ==========================================

actual_labels = (
    test_data["tshirt_size"].values
)


# ==========================================
# 9. COMPARE RESULTS
# ==========================================

predicted_accuracy = accuracy_score(
    actual_labels,
    predicted_sizes
)


actual_measurement_accuracy = accuracy_score(
    actual_labels,
    actual_measurement_sizes
)


print("\n==========================================")
print("DEBUG RESULTS")
print("==========================================")


print(
    "\nActual measurements -> Size:"
)

print(
    round(
        actual_measurement_accuracy * 100,
        2
    ),
    "%"
)


print(
    "\nPredicted measurements -> Size:"
)

print(
    round(
        predicted_accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# 10. CHECK PREDICTED SIZE DISTRIBUTION
# ==========================================

print("\n==========================================")
print("PREDICTED SIZE DISTRIBUTION")
print("==========================================")


print(
    pd.Series(
        predicted_sizes
    ).value_counts()
    .sort_index()
)


# ==========================================
# 11. DISPLAY FIRST 20 CASES
# ==========================================

print("\n==========================================")
print("FIRST 20 TEST CASES")
print("==========================================")


comparison = pd.DataFrame({

    "Actual_Size":
        actual_labels[:20],

    "Size_From_Actual_Measurements":
        actual_measurement_sizes[:20],

    "Size_From_Predicted_Measurements":
        predicted_sizes[:20],

    "Actual_Chest":
        test_data["chest"]
        .values[:20],

    "Predicted_Chest":
        predicted_df["chest"]
        .values[:20],

    "Actual_Waist":
        test_data["waist"]
        .values[:20],

    "Predicted_Waist":
        predicted_df["waist"]
        .values[:20]

})


print(
    comparison
)


print("\n==========================================")
print("DEBUG COMPLETED")
print("==========================================")