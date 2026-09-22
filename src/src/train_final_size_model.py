# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# FINAL SIZE MODEL TRAINING
# ==========================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading ML dataset...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


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


df["tshirt_size"] = df["chest"].apply(
    assign_tshirt_size
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
# 4. SUBJECT-LEVEL SPLIT
# ==========================================

print("\nCreating subject-level train/test split...")

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


train_data = df.iloc[
    train_indices
].copy()


test_data = df.iloc[
    test_indices
].copy()


print("\nTraining images:")
print(len(train_data))

print("Testing images:")
print(len(test_data))

print("\nTraining subjects:")
print(
    train_data["subject_id"].nunique()
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

print("Body measurement model loaded!")


# ==========================================
# 6. PREDICT MEASUREMENTS FOR TRAINING DATA
# ==========================================

print("\nPredicting training measurements...")

X_train_image = train_data[
    measurement_features
]


predicted_train_measurements = (
    measurement_model.predict(
        X_train_image
    )
)


# ==========================================
# 7. PREDICT MEASUREMENTS FOR TEST DATA
# ==========================================

print("Predicting testing measurements...")

X_test_image = test_data[
    measurement_features
]


predicted_test_measurements = (
    measurement_model.predict(
        X_test_image
    )
)


print("Measurement prediction completed!")


# ==========================================
# 8. CREATE MEASUREMENT DATAFRAMES
# ==========================================

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


train_measurements = pd.DataFrame(
    predicted_train_measurements,
    columns=measurement_names
)


test_measurements = pd.DataFrame(
    predicted_test_measurements,
    columns=measurement_names
)


# ==========================================
# 9. TRAIN FINAL SIZE MODEL
# ==========================================

print("\n==========================================")
print("TRAINING FINAL SIZE MODEL")
print("==========================================")


size_features = [

    "ankle",
    "arm-length",
    "bicep",
    "calf",
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


X_train_size = train_measurements[
    size_features
]


y_train = train_data[
    "tshirt_size"
]


X_test_size = test_measurements[
    size_features
]


y_test = test_data[
    "tshirt_size"
]


print("\nTraining final Random Forest...")


final_model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


final_model.fit(
    X_train_size,
    y_train
)


print("Final model training completed!")


# ==========================================
# 10. TEST FINAL MODEL
# ==========================================

print("\nMaking predictions...")


predictions = final_model.predict(
    X_test_size
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==========================================")
print("FINAL MODEL VALIDATION")
print("==========================================")


print(
    "Exact accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# 11. PREDICTED SIZE DISTRIBUTION
# ==========================================

print("\n==========================================")
print("PREDICTED SIZE DISTRIBUTION")
print("==========================================")


print(
    pd.Series(
        predictions
    ).value_counts()
    .sort_index()
)


# ==========================================
# 12. SAVE FINAL MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


model_path = (
    "models/final_tshirt_size_model.pkl"
)


features_path = (
    "models/final_tshirt_size_features.pkl"
)


joblib.dump(
    final_model,
    model_path
)


joblib.dump(
    size_features,
    features_path
)


print("\n==========================================")
print("FINAL MODEL SAVED")
print("==========================================")


print(
    "Model:",
    model_path
)


print(
    "Features:",
    features_path
)


# ==========================================
# 13. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("FINAL SIZE MODEL TRAINING COMPLETED")
print("==========================================")