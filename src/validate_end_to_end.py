# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# END-TO-END PIPELINE VALIDATION
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading ML dataset...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv"
)

print("Dataset loaded successfully!")

print(
    "Dataset shape:",
    df.shape
)


# ==========================================
# 2. CREATE T-SHIRT SIZE LABELS
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


df["tshirt_size"] = (
    df["chest"].apply(
        assign_tshirt_size
    )
)


# ==========================================
# 3. IMAGE FEATURES
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


# ==========================================
# 4. SUBJECT-LEVEL TEST SPLIT
# ==========================================

print("\nCreating subject-level test set...")

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


test_data = df.iloc[
    test_indices
].copy()


print("\nTesting images:")
print(
    len(test_data)
)


print("Testing subjects:")
print(
    test_data["subject_id"].nunique()
)


# ==========================================
# 5. LOAD BODY MEASUREMENT MODEL
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
# 6. PREDICT BODY MEASUREMENTS
# ==========================================

print("\nPredicting body measurements...")

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


predicted_measurements_df = pd.DataFrame(
    predicted_measurements,
    columns=measurement_names,
    index=test_data.index
)


print(
    "✅ Body measurements predicted"
)


# ==========================================
# 7. LOAD T-SHIRT SIZE MODEL
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
# 8. PREPARE RAW PREDICTED MEASUREMENTS
# ==========================================

X_size = predicted_measurements_df[
    size_features
]


# ==========================================
# 9. PREDICT T-SHIRT SIZE
# ==========================================
#
# IMPORTANT:
# Random Forest was trained on RAW
# measurements, therefore no scaler.
#
# ==========================================

print("\nPredicting final T-shirt sizes...")

predicted_size_numeric = (
    size_model.predict(
        X_size
    )
)


predicted_sizes = (
    size_label_encoder.inverse_transform(
        predicted_size_numeric
    )
)


actual_sizes = (
    test_data["tshirt_size"].values
)


print(
    "✅ T-shirt sizes predicted"
)


# ==========================================
# 10. EXACT ACCURACY
# ==========================================

accuracy = accuracy_score(
    actual_sizes,
    predicted_sizes
)


print("\n==========================================")
print("END-TO-END EXACT ACCURACY")
print("==========================================")


print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

labels = [
    "XS",
    "S",
    "M",
    "L",
    "XL",
    "XXL"
]


print("\n==========================================")
print("END-TO-END CLASSIFICATION REPORT")
print("==========================================")


print(
    classification_report(
        actual_sizes,
        predicted_sizes,
        labels=labels,
        zero_division=0
    )
)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

print("\n==========================================")
print("END-TO-END CONFUSION MATRIX")
print("==========================================")


matrix = confusion_matrix(
    actual_sizes,
    predicted_sizes,
    labels=labels
)


print("\nLabels:")
print(labels)


print("\nMatrix:")
print(matrix)


# ==========================================
# 13. ±1 SIZE ACCURACY
# ==========================================

size_order = {

    "XS": 0,
    "S": 1,
    "M": 2,
    "L": 3,
    "XL": 4,
    "XXL": 5
}


within_one_size = []


for actual, predicted in zip(
    actual_sizes,
    predicted_sizes
):

    actual_number = (
        size_order[actual]
    )

    predicted_number = (
        size_order[predicted]
    )

    difference = abs(
        actual_number -
        predicted_number
    )

    within_one_size.append(
        difference <= 1
    )


within_one_accuracy = (
    np.mean(
        within_one_size
    )
)


print("\n==========================================")
print("WITHIN ±1 SIZE ACCURACY")
print("==========================================")


print(
    "Accuracy:",
    round(
        within_one_accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# 14. SIZE DISTRIBUTION
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
# 15. SAVE RESULTS
# ==========================================

results = pd.DataFrame({

    "photo_id":
        test_data["photo_id"].values,

    "subject_id":
        test_data["subject_id"].values,

    "actual_size":
        actual_sizes,

    "predicted_size":
        predicted_sizes

})


results.to_csv(
    "data/processed/"
    "end_to_end_validation_results.csv",
    index=False
)


print("\nValidation results saved to:")

print(
    "data/processed/"
    "end_to_end_validation_results.csv"
)


# ==========================================
# 16. FINISH
# ==========================================

print("\n==========================================")
print("END-TO-END VALIDATION COMPLETED")
print("==========================================")