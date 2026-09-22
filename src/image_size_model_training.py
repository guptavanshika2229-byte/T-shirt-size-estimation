# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# IMAGE → T-SHIRT SIZE MODEL
# ==========================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
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
# 2. CREATE T-SHIRT SIZE LABEL
# ==========================================

print("\nCreating T-shirt size labels...")


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
    df["chest"].apply(assign_tshirt_size)
)


print("\nT-shirt size distribution:")

print(
    df["tshirt_size"]
    .value_counts()
    .sort_index()
)


# ==========================================
# 3. SELECT IMAGE FEATURES
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


X = df[features]

y = df["tshirt_size"]

groups = df["subject_id"]


print("\nNumber of image features:")
print(len(features))


print("\nInput shape:")
print(X.shape)


# ==========================================
# 4. SUBJECT-LEVEL TRAIN/TEST SPLIT
# ==========================================

print(
    "\nCreating subject-level train/test split..."
)


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
    df.iloc[train_indices]
    ["subject_id"]
    .nunique()
)


print("Testing subjects:")

print(
    df.iloc[test_indices]
    ["subject_id"]
    .nunique()
)


# ==========================================
# 5. TRAIN RANDOM FOREST CLASSIFIER
# ==========================================

print("\n==========================================")

print("TRAINING RANDOM FOREST CLASSIFIER")

print("==========================================")


model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


print("\nTraining model...")


model.fit(

    X_train,

    y_train
)


print("Model training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")


predictions = model.predict(

    X_test
)


# ==========================================
# 7. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(

    y_test,

    predictions
)


print("\n==========================================")

print("MODEL ACCURACY")

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
# 8. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")

print("CLASSIFICATION REPORT")

print("==========================================")


print(

    classification_report(

        y_test,

        predictions,

        labels=[
            "XS",
            "S",
            "M",
            "L",
            "XL",
            "XXL"
        ],

        zero_division=0
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("\n==========================================")

print("CONFUSION MATRIX")

print("==========================================")


labels = [

    "XS",

    "S",

    "M",

    "L",

    "XL",

    "XXL"
]


matrix = confusion_matrix(

    y_test,

    predictions,

    labels=labels
)


print("\nLabels:")

print(labels)


print("\nMatrix:")

print(matrix)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

print("\n==========================================")

print("FEATURE IMPORTANCE")

print("==========================================")


importance = pd.DataFrame(

    {

        "feature": features,

        "importance":
            model.feature_importances_

    }

)


importance = importance.sort_values(

    by="importance",

    ascending=False
)


print(

    importance
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs(

    "models",

    exist_ok=True
)


model_path = (

    "models/image_tshirt_size_model.pkl"

)


features_path = (

    "models/image_tshirt_size_features.pkl"

)


joblib.dump(

    model,

    model_path
)


joblib.dump(

    features,

    features_path
)


# ==========================================
# 12. SAVE CLASS LABELS
# ==========================================

labels_path = (

    "models/tshirt_size_classes.pkl"

)


joblib.dump(

    labels,

    labels_path
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


print(

    "Classes:",

    labels_path
)


# ==========================================
# 13. FINAL MESSAGE
# ==========================================

print("\n==========================================")

print("IMAGE T-SHIRT SIZE MODEL TRAINING COMPLETED")

print("==========================================")