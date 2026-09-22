# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# BODY MEASUREMENT MODEL TRAINING
# ==========================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD ML DATASET
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
# 2. DEFINE IMAGE FEATURES
# ==========================================

features = [
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


# ==========================================
# 3. DEFINE BODY MEASUREMENT TARGETS
# ==========================================

targets = [
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


print("\nNumber of input features:")
print(len(features))

print("\nNumber of target measurements:")
print(len(targets))


# ==========================================
# 4. CREATE INPUT AND TARGET DATA
# ==========================================

X = df[features]

y = df[targets]

groups = df["subject_id"]


print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ==========================================
# 5. SUBJECT-LEVEL TRAIN/TEST SPLIT
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


X_train = X.iloc[train_indices]

X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]

y_test = y.iloc[test_indices]


print("\nTraining samples:")
print(len(X_train))

print("Testing samples:")
print(len(X_test))


print("\nTraining subjects:")
print(
    df.iloc[train_indices]["subject_id"].nunique()
)

print("Testing subjects:")
print(
    df.iloc[test_indices]["subject_id"].nunique()
)


# ==========================================
# 6. TRAIN RANDOM FOREST REGRESSOR
# ==========================================

print("\n==========================================")
print("TRAINING RANDOM FOREST")
print("==========================================")


base_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model = MultiOutputRegressor(
    base_model
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


print("Model training completed!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test
)


predictions_df = pd.DataFrame(
    predictions,
    columns=targets
)


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")


overall_mae = mean_absolute_error(
    y_test,
    predictions
)


overall_r2 = r2_score(
    y_test,
    predictions
)


print("\nOverall Mean Absolute Error:")
print(
    round(overall_mae, 3)
)


print("\nOverall R² Score:")
print(
    round(overall_r2, 3)
)


# ==========================================
# 9. INDIVIDUAL MEASUREMENT PERFORMANCE
# ==========================================

print("\n==========================================")
print("INDIVIDUAL MEASUREMENT PERFORMANCE")
print("==========================================")


for index, target in enumerate(targets):

    actual = y_test.iloc[:, index]

    predicted = predictions[:, index]


    mae = mean_absolute_error(
        actual,
        predicted
    )


    r2 = r2_score(
        actual,
        predicted
    )


    print(
        f"{target:20s} "
        f"MAE = {mae:.2f} cm   "
        f"R² = {r2:.3f}"
    )


# ==========================================
# 10. SHOW SAMPLE PREDICTIONS
# ==========================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")


sample_results = pd.DataFrame(
    {
        "Actual Chest": y_test["chest"].values[:10],
        "Predicted Chest": predictions[:10, targets.index("chest")],

        "Actual Waist": y_test["waist"].values[:10],
        "Predicted Waist": predictions[:10, targets.index("waist")],

        "Actual Hip": y_test["hip"].values[:10],
        "Predicted Hip": predictions[:10, targets.index("hip")]
    }
)


print(
    sample_results
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


model_path = (
    "models/body_measurement_model.pkl"
)


features_path = (
    "models/body_measurement_features.pkl"
)


joblib.dump(
    model,
    model_path
)


joblib.dump(
    features,
    features_path
)


print("\n==========================================")
print("MODEL SAVED")
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
# 12. FINISH
# ==========================================

print("\n==========================================")
print("BODY MEASUREMENT MODEL TRAINING COMPLETED")
print("==========================================")