# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# Model Training
# ==========================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading BodyM dataset...")

df = pd.read_csv("data/raw/measurements.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. CHECK AND CLEAN DATA
# ==========================================

print("\nChecking missing values:")

print(df.isnull().sum())


# Remove rows containing missing values
df = df.dropna()

print("\nDataset shape after cleaning:", df.shape)


# Remove duplicate rows
df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)


# ==========================================
# 3. CREATE T-SHIRT SIZE LABEL
# ==========================================

print("\nCreating T-shirt size labels...")


def assign_tshirt_size(chest):
    """
    Assign a T-shirt size based on chest circumference.

    These ranges are project-defined prototype ranges.
    They are not official BodyM labels.
    """

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


df["tshirt_size"] = df["chest"].apply(assign_tshirt_size)


print("\nT-shirt size distribution:")
print(df["tshirt_size"].value_counts().sort_index())


# ==========================================
# 4. SELECT FEATURES
# ==========================================

features = [
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

target = "tshirt_size"


X = df[features]
y = df[target]


print("\nFeatures used for training:")
print(features)

print("\nTarget variable:")
print(target)


# ==========================================
# 5. CONVERT SIZE LABELS INTO NUMBERS
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nSize classes:")
print(label_encoder.classes_)


# ==========================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 8. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# Make predictions
logistic_predictions = logistic_model.predict(X_test_scaled)


# Calculate accuracy
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print("\nLogistic Regression Accuracy:")
print(logistic_accuracy)


# ==========================================
# 9. TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)


# Make predictions
random_forest_predictions = random_forest_model.predict(X_test)


# Calculate accuracy
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)


print("\nRandom Forest Accuracy:")
print(random_forest_accuracy)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        random_forest_predictions,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        random_forest_predictions
    )
)


# ==========================================
# 12. COMPARE MODELS
# ==========================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    "Logistic Regression:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(random_forest_accuracy * 100, 2),
    "%"
)


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

if random_forest_accuracy >= logistic_accuracy:

    best_model = random_forest_model
    best_model_name = "Random Forest"

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"


print("\nBest model:", best_model_name)


# ==========================================
# 14. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/tshirt_size_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

joblib.dump(
    features,
    "models/features.pkl"
)


print("\nModel files saved successfully!")

print("Saved files:")
print("models/tshirt_size_model.pkl")
print("models/scaler.pkl")
print("models/label_encoder.pkl")
print("models/features.pkl")


print("\n==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")