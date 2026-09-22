import pandas as pd
import os


print("Loading silhouette features...")

features = pd.read_csv(
    "data/processed/all_silhouette_features.csv"
)

print("Silhouette features loaded!")
print("Shape:", features.shape)


print("\nLoading BodyM measurement mapping...")

measurements = pd.read_csv(
    "data/processed/image_measurement_dataset.csv"
)

print("Measurement mapping loaded!")
print("Shape:", measurements.shape)


print("\nMerging datasets...")

ml_dataset = features.merge(
    measurements,
    on="photo_id",
    how="inner"
)


print("\n==========================================")
print("ML DATASET CREATED")
print("==========================================")

print("Shape:", ml_dataset.shape)

print("\nColumns:")
print(ml_dataset.columns.tolist())


print("\nMissing values:")

print(
    ml_dataset.isnull().sum()
)


print("\nFirst 5 rows:")

print(
    ml_dataset.head()
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


output_path = (
    "data/processed/ml_dataset.csv"
)


ml_dataset.to_csv(
    output_path,
    index=False
)


print("\nML dataset saved successfully!")

print(
    "Saved at:",
    output_path
)


print("\n==========================================")
print("DATA PREPARATION COMPLETED")
print("==========================================")